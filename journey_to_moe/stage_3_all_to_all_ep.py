# Here we will implement basic all2all communication with process groups and MoE

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from stage_2_all_to_all_group import All2All, all2all
import math
from torch.profiler import profile, record_function, ProfilerActivity

IGNORE_PRINT = True


def print0(s):
    if IGNORE_PRINT:
        return
    if dist.get_rank() == 0:
        print(s)


class ExpertLayer(nn.Module):
    def __init__(self, hidden_dim, ffn_dim):
        super().__init__()
        self.w1 = nn.Linear(hidden_dim, ffn_dim)
        self.w2 = nn.Linear(ffn_dim, hidden_dim)

    def forward(self, x):
        return self.w2(F.gelu(self.w1(x)))


class Router(nn.Module):
    def __init__(self, hidden_dim, num_experts, top_k):
        super().__init__()
        self.router = nn.Linear(hidden_dim, num_experts, bias=False)
        self.num_experts = num_experts
        self.top_k = top_k
        # init small randn
        torch.nn.init.normal_(
            self.router.weight, mean=0.0, std=0.01 / math.sqrt(hidden_dim)
        )

    def forward(self, hidden_states):

        router_logits = self.router(hidden_states)
        probs = F.softmax(router_logits, dim=-1)
        print0(f"Router logits: {router_logits}")
        print0(f"Router probs: {probs}")
        probs, indices = torch.topk(probs, self.top_k, dim=-1)

        return probs, indices


class MoELayer(nn.Module):
    def __init__(self, hidden_dim, ffn_dim, num_experts, top_k):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim

        # Create router
        self.router = Router(hidden_dim, num_experts, top_k=top_k)

        # Create experts
        self.expert = ExpertLayer(hidden_dim, ffn_dim)

        # Create All2All handler
        self.a2a = All2All(num_experts)

        self.top_k = top_k
        self.capacity = 512

    def _permute_tokens(self, tokens, expert_indices, probs):
        # Get indices of tokens going to each expert

        token_indices_per_expert = (
            []
        )  # this will will be list of lists, keeping track of indices of tokens going to each expert
        token_probs_per_expert = (
            []
        )  # in the above order, keeping track of probabilities of tokens going to each expert

        for expert_idx in range(self.num_experts):
            this_expert_indices = []
            this_expert_probs = []

            for i in range(self.top_k):
                this_expert_indice = torch.where(expert_indices[:, i] == expert_idx)[0]
                this_expert_prob = probs[this_expert_indice, i]

                this_expert_indices.append(this_expert_indice)
                this_expert_probs.append(this_expert_prob)

            this_expert_indices = torch.cat(this_expert_indices)
            this_expert_probs = torch.cat(this_expert_probs)

            if len(this_expert_indices) > self.capacity:
                to_select = torch.randperm(len(this_expert_indices))[: self.capacity]
                this_expert_indices = this_expert_indices[to_select]
                this_expert_probs = this_expert_probs[to_select]

            token_indices_per_expert.append(this_expert_indices)
            token_probs_per_expert.append(this_expert_probs)

        # permutation tells us which token goes to which expert. Notice that we have -1s in the permutation.
        for i in range(self.num_experts):
            print0(f"Expert {i} indices: {token_indices_per_expert[i]}")

        permuted_tokens = []
        for i in range(self.num_experts):
            expert_tokens = tokens[token_indices_per_expert[i]]
            if len(expert_tokens) < self.capacity:
                expert_tokens = torch.cat(
                    [
                        expert_tokens,
                        torch.zeros(
                            (self.capacity - len(expert_tokens), self.hidden_dim),
                            device=tokens.device,
                            dtype=tokens.dtype,
                        )
                        * -1,
                    ]
                )

            permuted_tokens.append(expert_tokens)

        permuted_tokens = torch.cat(permuted_tokens)

        return permuted_tokens, token_indices_per_expert, token_probs_per_expert

    def _unpermute_tokens(
        self, buffer, tokens, token_indices_per_expert, token_probs_per_expert
    ):
        # Create reverse permutation. Now each token goes back to its original permutation.
        new_tokens = torch.zeros_like(buffer)
        for i in range(self.num_experts):
            offset = self.capacity * i
            print0(f"token_indices_per_expert[i]: {token_indices_per_expert[i].shape}")
            print0(
                f"tokens[offset : offset + len(token_indices_per_expert[i])]: {tokens[offset : offset + len(token_indices_per_expert[i])].shape}"
            )
            print0(f"token_probs_per_expert[i]: {token_probs_per_expert[i].shape}")
            new_tokens[token_indices_per_expert[i]] += (
                tokens[offset : offset + len(token_indices_per_expert[i])]
                * token_probs_per_expert[i][:, None]
            )

        return new_tokens

    def forward(self, hidden_states):
        # Get routing probabilities and map
        original_shape = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
        probs, expert_indices = self.router(hidden_states)
        print0(f"Expert indices: {expert_indices}")
        print0(f"Probs: {probs.shape}")

        # Permute tokens locally based on routing
        permuted_tokens, token_indices_per_expert, token_probs_per_expert = (
            self._permute_tokens(hidden_states, expert_indices, probs)
        )

        # All2All communication to get tokens to correct experts
        expert_tokens = all2all(permuted_tokens, self.a2a)

        # now all tokens are in the correct experts. We can now process them
        expert_outputs = self.expert(expert_tokens)

        # All2All communication to return tokens
        local_tokens = all2all(expert_outputs, self.a2a)

        # Unpermute tokens back to original order
        output = self._unpermute_tokens(
            hidden_states,
            local_tokens,
            token_indices_per_expert,
            token_probs_per_expert,
        )

        return output.reshape(original_shape)


# Example usage: torchrun --nproc_per_node=8 stage_3_all_to_all_ep.py

if __name__ == "__main__":

    dist.init_process_group("nccl")
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    torch.cuda.set_device(rank)

    # MoE params
    hidden_dim = 1024
    num_experts = 8  # This will create 4 expert groups
    batch_size = 32
    seq_len = 1024
    top_k = 2

    moe = MoELayer(hidden_dim, hidden_dim * 4, num_experts, top_k=top_k).cuda()
    print(next(moe.parameters()).device)

    x = torch.randn(batch_size * seq_len, hidden_dim, device="cuda", requires_grad=True)
    x.retain_grad()

    # Forward and backward pass

    with profile(
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler("./log/stage3_ep"),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        for i in range(10):
            prof.step()
            out = moe(x)
            loss = out.mean()
            loss.backward(retain_graph=True)

    print0(f"input: {x[:3, :3]}")
    print0(f"out: {out[:3, :3]}")

    if rank == 0:
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {out.shape}")

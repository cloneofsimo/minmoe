# Here we will implement basic all2all communication with process groups and MoE

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from stage_2_all_to_all_group import All2All, all2all
import math
from torch.profiler import profile, record_function, ProfilerActivity

IGNORE_PRINT = False

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


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

        torch.nn.init.normal_(
            self.router.weight, mean=0.0, std=0.01 / math.sqrt(hidden_dim)
        )

    def forward(self, hidden_states):

        router_logits = self.router(hidden_states)
        probs = F.softmax(router_logits, dim=-1)

        probs, indices = torch.topk(probs, self.top_k, dim=-1)

        return probs, indices


class MoELayer(nn.Module):
    def __init__(self, hidden_dim, ffn_dim, num_experts, top_k, a2a):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim

        # Create router
        self.router = Router(hidden_dim, num_experts, top_k=top_k)

        # Create experts
        self.expert = ExpertLayer(hidden_dim, ffn_dim)

        # Create All2All handler
        self.a2a = a2a

        self.top_k = top_k
        self.capacity = 512

    def _permute_tokens(self, tokens, expert_indices, probs):
        # Get indices of tokens going to each expert

        expert_indices = expert_indices.reshape(-1)
        expert_indices_values, sorted_token_indices = torch.sort(expert_indices, dim=0)
        probs = probs.reshape(-1)
        # find where the values change
        value_change_indices = torch.where(
            expert_indices_values[1:] != expert_indices_values[:-1]
        )[0].tolist()
        value_change_indices = [0] + value_change_indices + [len(expert_indices)]
        # print(value_change_indices)
        # get the indices of the tokens that are going to each expert
        token_indices_per_expert = [
            sorted_token_indices[value_change_indices[i] : value_change_indices[i + 1]][
                : self.capacity
            ]
            for i in range(len(value_change_indices) - 1)
        ]
        token_probs_per_expert = [
            probs[token_indices_per_expert[i]]
            for i in range(len(token_indices_per_expert))
        ]
        token_indices_per_expert = [
            token_indices_per_expert[i].to(tokens.device) // self.top_k
            for i in range(len(token_indices_per_expert))
        ]

        # Pre-allocate full tensor instead of building list
        permuted_tokens = torch.full(
            (self.num_experts * self.capacity, self.hidden_dim),
            0,
            device=tokens.device,
            dtype=tokens.dtype,
        )

        for i in range(self.num_experts):
            # Get indices for this expert
            expert_indices = token_indices_per_expert[i].to(tokens.device)
            num_tokens = len(expert_indices)
            offset = i * self.capacity
            # Copy actual tokens
            permuted_tokens[offset : offset + num_tokens] = tokens[expert_indices]

        return permuted_tokens, token_indices_per_expert, token_probs_per_expert

    def _unpermute_tokens(
        self, buffer, tokens, token_indices_per_expert, token_probs_per_expert
    ):
        # Create reverse permutation. Now each token goes back to its original place.
        # We would notice that prob is still differentiable.
        new_tokens = torch.zeros_like(buffer)
        for i in range(self.num_experts):
            offset = self.capacity * i
            new_tokens[token_indices_per_expert[i]] += tokens[
                offset : offset + len(token_indices_per_expert[i])
            ] * token_probs_per_expert[i][:, None].to(buffer.device)

        return new_tokens

    def forward(self, hidden_states):
        # Get routing probabilities and map
        original_shape = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
        probs, expert_indices = self.router(hidden_states)

        permuted_tokens, token_indices_per_expert, token_probs_per_expert = (
            self._permute_tokens(hidden_states, expert_indices, probs)
        )

        # All2All communication to get tokens to correct experts
        expert_tokens = all2all(permuted_tokens, self.a2a)

        # now all tokens are in the correct experts. We can now process them
        expert_outputs = self.expert(expert_tokens)

        # All2All communication to return tokens
        local_tokens = all2all(expert_outputs, self.a2a)
        print(f"local_tokens: {local_tokens.shape}")

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

    hidden_dim = 8192
    num_experts = 8
    batch_size = 4
    seq_len = 1024
    top_k = 2

    a2a = All2All(num_experts)

    moe = MoELayer(hidden_dim, hidden_dim * 4, num_experts, top_k=top_k, a2a=a2a).cuda()
    print(next(moe.parameters()).device)
    # print total param
    total_params = sum(p.numel() for p in moe.parameters())
    print(f"Total params: {total_params}")

    x = torch.randn(batch_size * seq_len, hidden_dim, device="cuda", requires_grad=True)
    x.retain_grad()

    with profile(
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=10, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(
            "./log/stage3_ep_opt_8"
        ),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        for i in range(20):
            prof.step()
            out = moe(x)
            loss = out.mean()
            loss.backward(retain_graph=True)

    if rank == 0:
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {out.shape}")

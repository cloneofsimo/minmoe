import torch
import torch.nn as nn
import torch.nn.functional as F
from stage_3_all_to_all_ep import MoELayer, print0, All2All


class MoEGPTBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, num_experts, top_k, ffn_dim, a2a):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.moe = MoELayer(hidden_dim, ffn_dim, num_experts, top_k, a2a)

    def forward(self, x, attention_mask=None):
        residual = x
        x = self.ln1(x)
        x, _ = self.attn(x, x, x)
        x = x + residual

        residual = x
        x = self.ln2(x)
        x = self.moe(x)

        return x + residual


class MoEGPT(nn.Module):
    def __init__(
        self,
        vocab_size,
        hidden_dim,
        num_layers,
        num_heads,
        num_experts,
        top_k,
        max_seq_len=1024,
    ):
        super().__init__()

        a2a = All2All(num_experts)
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.position_embedding = nn.Parameter(torch.zeros(max_seq_len, hidden_dim))

        ffn_dim = hidden_dim * 4
        self.blocks = nn.ModuleList(
            [
                MoEGPTBlock(hidden_dim, num_heads, num_experts, top_k, ffn_dim, a2a)
                for _ in range(num_layers)
            ]
        )

        self.ln_f = nn.LayerNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids, attention_mask=None):
        x = self.token_embedding(input_ids)
        x = x + self.position_embedding[: input_ids.size(1)]

        for block in self.blocks:
            x = block(x, attention_mask)

        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits


def train_step(model, optimizer, batch):
    input_ids = batch["input_ids"]
    labels = batch["labels"]
    attention_mask = batch["attention_mask"]

    logits = model(input_ids, attention_mask)

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return {"loss": loss.item()}


import time
import torch.distributed as dist

if __name__ == "__main__":

    dist.init_process_group("nccl")
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    torch.cuda.set_device(rank)

    vocab_size = 512
    hidden_dim = 4096
    num_layers = 8
    num_heads = 8
    num_experts = 8
    top_k = 2

    model = MoEGPT(
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        num_experts=num_experts,
        top_k=top_k,
    ).cuda()

    optimizer = torch.optim.AdamW(model.token_embedding.parameters(), lr=1e-4)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total_params}")

    batch = {
        "input_ids": torch.randint(0, vocab_size, (4, 1024), device="cuda"),
        "labels": torch.randint(0, vocab_size, (4, 1024), device="cuda"),
        "attention_mask": torch.ones(4, 1024, device="cuda", dtype=torch.bool),
    }

    idx = 0
    total_steps = 8
    taken_times = []
    for _ in range(total_steps):
        t0 = time.time()

        metrics = train_step(model, optimizer, batch)
        time_taken = time.time() - t0
        taken_times.append(time_taken)

    sorted_times = sorted(taken_times)[1:-1]
    average_time = sum(sorted_times) / len(sorted_times)
    min_time = min(sorted_times)
    max_time = max(sorted_times)
    std_time = torch.tensor(sorted_times).std().item()

    print0(f"\nTraining Statistics:")
    print0(f"Average step time: {average_time:.3f}s")
    print0(f"Min step time: {min_time:.3f}s")
    print0(f"Max step time: {max_time:.3f}s")
    print0(f"Std step time: {std_time:.3f}s")

    print0(f"Time taken: {average_time}")
    mfu = 6 * total_params * 4096 / (average_time * 5 * 10**14)
    print0(f"MFU: {mfu * 100 :.2f}%")

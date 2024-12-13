import torch
import torch.nn as nn
import torch.nn.functional as F
from stage_3_all_to_all_ep import MoELayer, print0


class MoEGPTBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, num_experts, top_k, ffn_dim):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.moe = MoELayer(hidden_dim, ffn_dim, num_experts, top_k)

    def forward(self, x, attention_mask=None):
        # Self attention
        residual = x
        x = self.ln1(x)
        x, _ = self.attn(x, x, x)
        x = x + residual

        print0(f"x-after-attn: {x.std()}")
        # MoE FFN
        residual = x
        x = self.ln2(x)
        x = self.moe(x)
        print0(f"x-after-moe: {x.std()}")
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
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.position_embedding = nn.Parameter(torch.zeros(max_seq_len, hidden_dim))

        ffn_dim = hidden_dim * 4
        self.blocks = nn.ModuleList(
            [
                MoEGPTBlock(hidden_dim, num_heads, num_experts, top_k, ffn_dim)
                for _ in range(num_layers)
            ]
        )

        self.ln_f = nn.LayerNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids, attention_mask=None):
        # Get embeddings
        x = self.token_embedding(input_ids)
        x = x + self.position_embedding[: input_ids.size(1)]

        # Transform through layers
        for block in self.blocks:
            x = block(x, attention_mask)

        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits


def train_step(model, optimizer, batch):
    input_ids = batch["input_ids"]
    labels = batch["labels"]
    attention_mask = batch["attention_mask"]

    # Forward pass
    logits = model(input_ids, attention_mask)

    # Compute main loss
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return {"loss": loss.item()}


# Example usage
# torchrun --nproc_per_node=8 stage_4_gpt.py
if __name__ == "__main__":
    import torch.distributed as dist

    dist.init_process_group("nccl")
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    torch.cuda.set_device(rank)

    # Model params
    vocab_size = 50257  # GPT-2 vocab size
    hidden_dim = 768
    num_layers = 12
    num_heads = 12
    num_experts = 8
    top_k = 2

    # Create model
    model = MoEGPT(
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        num_experts=num_experts,
        top_k=top_k,
    ).cuda()

    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Dummy batch for demonstration
    batch = {
        "input_ids": torch.randint(0, vocab_size, (8, 512), device="cuda"),
        "labels": torch.randint(0, vocab_size, (8, 512), device="cuda"),
        "attention_mask": torch.ones(8, 512, device="cuda", dtype=torch.bool),
    }

    # Training step
    metrics = train_step(model, optimizer, batch)
    print0(f"Training metrics: {metrics}")

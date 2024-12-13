import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms


def setup():
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert world_size == 8, "Need exactly 8 GPUs"
    torch.cuda.set_device(rank)
    return rank, world_size


class DistributedMoE(nn.Module):
    def __init__(self, hidden_size=128, ffn_ratio=4):
        super().__init__()
        self.hidden_size = hidden_size
        self.rank = dist.get_rank()
        self.router = nn.Linear(hidden_size, 8, bias=False)
        self.local_expert = nn.Sequential(
            nn.Linear(hidden_size, ffn_ratio * hidden_size),
            nn.GELU(),
            nn.Linear(ffn_ratio * hidden_size, hidden_size),
        )

    def forward(self, x):
        B, S, H = x.shape
        x_flat = x.reshape(-1, H)

        # Route tokens
        scores = self.router(x_flat)  # [B*S, 8]
        probs = F.softmax(scores, dim=-1)  # [B*S, 8]

        # Get assignments for current expert
        expert_probs = probs[:, self.rank]  # [B*S]
        selected = expert_probs > 0

        # Get tokens for this expert
        expert_tokens = x_flat[selected]  # [num_selected, H]
        expert_weights = expert_probs[selected]  # [num_selected]

        # Process if we have tokens
        if len(expert_tokens) > 0:
            expert_output = self.local_expert(expert_tokens)
        else:
            expert_output = torch.zeros(0, H, device=x.device)

        # Share sizes first
        local_count = torch.tensor([len(expert_tokens)], device=x.device)
        all_counts = [torch.zeros_like(local_count) for _ in range(8)]
        dist.all_gather(all_counts, local_count)

        # Initialize output
        output_flat = torch.zeros_like(x_flat)

        # Process each expert
        for eid in range(8):
            count = all_counts[eid].item()
            if count == 0:
                continue

            # Get tokens for this expert
            curr_probs = probs[:, eid]
            curr_selected = curr_probs >= 0
            curr_weights = curr_probs[curr_selected]

            # Get output
            if eid == self.rank:
                curr_output = expert_output
            else:
                curr_output = torch.zeros(count, H, device=x.device)

            # Share output
            dist.broadcast(curr_output, eid)

            # Add to total output
            output_flat[curr_selected] = curr_output * curr_weights.unsqueeze(-1)

        return output_flat.view(B, S, H)


class MoEModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Linear(49, 128)  # For 7x7 MNIST patches
        self.moe = DistributedMoE(128)
        self.classifier = nn.Linear(128, 10)

    def forward(self, x):
        B = x.shape[0]
        # Convert image to patches
        patches = x.unfold(2, 7, 7).unfold(3, 7, 7)
        patches = patches.reshape(B, -1, 49)  # [B, num_patches, 49]

        # Process
        h = self.emb(patches)  # [B, num_patches, H]
        h = self.moe(h)  # [B, num_patches, H]
        h = h.mean(dim=1)  # [B, H]
        out = self.classifier(h)  # [B, 10]
        return out


import click

import wandb


@click.command()
@click.option("--lr", type=float, default=1e-3)
def train(lr):
    # Setup
    rank, _ = setup()
    torch.manual_seed(42 + rank)
    if rank == 0:
        wandb.init(project="moe-mnist", config={"lr": lr, "moe": "ref"})
    # Data
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    dataset = datasets.MNIST("data", train=True, download=True, transform=transform)
    sampler = DistributedSampler(dataset)
    loader = DataLoader(dataset, batch_size=32, sampler=sampler)

    # Model & optimizer
    model = MoEModel().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Train
    for epoch in range(3):
        sampler.set_epoch(epoch)
        for idx, (data, target) in enumerate(loader):
            data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()

            output = model(data)
            loss = F.cross_entropy(output, target)

            loss.backward()
            for p in model.parameters():
                if p.grad is not None:
                    dist.all_reduce(p.grad)
            optimizer.step()

            if idx % 100 == 0 and rank == 0:
                print(f"Epoch {epoch}, Step {idx}, Loss {loss.item():.4f}")
                wandb.log({"loss": loss.item()})


if __name__ == "__main__":
    # torchrun --nproc_per_node=8 train.py --lr=$lr
    train()

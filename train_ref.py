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
    torch.cuda.set_device(rank)
    return rank, world_size


class FFN(nn.Module):
    def __init__(self, hidden_size=128, ffn_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, ffn_dim), nn.GELU(), nn.Linear(ffn_dim, hidden_size)
        )

    def forward(self, x):
        return self.net(x)


class DistributedFFN(nn.Module):
    """Standard FFN - matches structure of DistributedMoE but uses single FFN"""

    def __init__(self, hidden_size=128):
        super().__init__()
        self.hidden_size = hidden_size
        self.rank = dist.get_rank()

        # No router needed
        self.ffn = FFN(hidden_size)

    def forward(self, x):
        B, S, H = x.shape
        x_flat = x.reshape(-1, H)

        # Just process all tokens through FFN
        output = self.ffn(x_flat)

        return output.view(B, S, H)


class ReferenceModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Linear(49, 128)  # For 7x7 MNIST patches
        self.ffn = DistributedFFN(128)
        self.classifier = nn.Linear(128, 10)

    def forward(self, x):
        B = x.shape[0]
        # Convert image to patches - same as MoE model
        patches = x.unfold(2, 7, 7).unfold(3, 7, 7)
        patches = patches.reshape(B, -1, 49)  # [B, num_patches, 49]

        # Process
        h = self.emb(patches)  # [B, num_patches, H]
        h = self.ffn(h)  # [B, num_patches, H]
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
    model = ReferenceModel().cuda()
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
            # Still need gradient sync in distributed training
            for p in model.parameters():
                if p.grad is not None:
                    dist.all_reduce(p.grad)
            optimizer.step()

            if idx % 100 == 0 and rank == 0:
                print(f"Epoch {epoch}, Step {idx}, Loss {loss.item():.4f}")
                wandb.log({"loss": loss.item()})


if __name__ == "__main__":
    train()

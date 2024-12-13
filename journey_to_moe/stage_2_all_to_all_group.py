import torch
import torch.distributed as dist


class All2All:

    def __init__(self, group_size):
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.group_size = group_size

        assert (
            self.world_size % group_size == 0
        ), "Group size must divide world size evenly"

        self.num_groups = self.world_size // group_size
        self.group_id = self.rank // group_size

        # Create process groups
        self.groups = []
        for i in range(self.num_groups):
            ranks = list(range(i * group_size, (i + 1) * group_size))
            self.groups.append(dist.new_group(ranks))

        self.group = self.groups[self.group_id]

    def forward(self, input_tensor):
        # Use all_to_all_single instead of all_to_all
        output = torch.empty_like(input_tensor)
        dist.all_to_all_single(output, input_tensor, group=self.group)
        return output

    def backward(self, grad_output):
        # Use all_to_all_single for backward pass too
        grad_input = torch.empty_like(grad_output)
        dist.all_to_all_single(grad_input, grad_output, group=self.group)
        return grad_input


class All2AllFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_tensor, all2all_handler):
        ctx.all2all_handler = all2all_handler
        return all2all_handler.forward(input_tensor)

    @staticmethod
    def backward(ctx, grad_output):
        return ctx.all2all_handler.backward(grad_output), None


def all2all(x, all2all_handler):
    return All2AllFunction.apply(x, all2all_handler)


# Example usage: torchrun --nproc_per_node=8 stage_2_all_to_all_group.py

if __name__ == "__main__":

    dist.init_process_group("nccl")
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    torch.cuda.set_device(rank)

    # Set group size - must divide world_size evenly
    group_size = 4  # This will create 4 groups of 2 processes each

    a2a = All2All(group_size)

    x = torch.ones((8, 2), device=rank, requires_grad=True) * (rank % group_size)
    x.retain_grad()

    for i in range(10):
        out = all2all(x, a2a)
        # Now loss is relative to rank within group
        group_rank = rank % group_size
        loss = 0.5 * ((out - group_rank) ** 2).sum()
        loss.backward()

    if rank == 0:
        print(out)
        print(f"Rank {rank}, \nOut: {out}, \nGrad: {x.grad}")

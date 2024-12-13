import torch
import torch.distributed as dist
from torch.profiler import profile, record_function, ProfilerActivity


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

        # Pre-allocate a tensor for outputs (just a placeholder; will be resized as needed)
        self.cached_output = None
        self.cached_grad_input = None

    def forward(self, input_tensor):
        # Allocate/reuse output tensor to avoid overhead
        if (
            self.cached_output is None
            or self.cached_output.size() != input_tensor.size()
        ):
            self.cached_output = torch.empty_like(input_tensor)
        dist.all_to_all_single(self.cached_output, input_tensor, group=self.group)
        return self.cached_output

    def backward(self, grad_output):
        # Allocate/reuse grad_input tensor
        if (
            self.cached_grad_input is None
            or self.cached_grad_input.size() != grad_output.size()
        ):
            self.cached_grad_input = torch.empty_like(grad_output)
        dist.all_to_all_single(self.cached_grad_input, grad_output, group=self.group)
        return self.cached_grad_input


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


if __name__ == "__main__":
    dist.init_process_group("nccl")
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    torch.cuda.set_device(rank)

    group_size = 4
    a2a = All2All(group_size)

    # Larger input might give more meaningful profiling data
    x = torch.ones((8, 2), device=rank, requires_grad=True) * (rank % group_size)
    x.retain_grad()

    # Example Profiling Run
    with profile(
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler("./log/stage2_1"),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:

        for i in range(10):
            prof.step()

            out = all2all(x, a2a)
            # Compute the loss relative to group rank
            group_rank = rank % group_size
            loss = 0.5 * ((out - group_rank) ** 2).sum()

            loss.backward(retain_graph=True)

    if rank == 0:
        print("Output:", out)
        print("Grad:", x.grad)
        # Print profiler summary
        # print(prof.key_averages().table(sort_by="self_cuda_time_total"))

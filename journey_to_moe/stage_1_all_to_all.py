# Here we will implement basic all2all communication, eventually build up to a MoE layer.
# Notice that all of this is not at all optimized, nor do I intend it to be.


import torch
import torch.distributed as dist


class All2AllFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_tensor, world_size):

        ctx.world_size = world_size

        chunks = [t.contiguous() for t in input_tensor.chunk(world_size, dim=0)]

        output_tensors = [torch.empty_like(chunk) for chunk in chunks]

        dist.all_to_all(output_tensors, chunks)

        output = torch.cat(output_tensors, dim=0)
        return output

    @staticmethod
    def backward(ctx, grad_output):

        world_size = ctx.world_size

        grad_chunks = [t.contiguous() for t in grad_output.chunk(world_size, dim=0)]

        grad_output_tensors = [torch.empty_like(chunk) for chunk in grad_chunks]

        dist.all_to_all(grad_output_tensors, grad_chunks)

        grad_input = torch.cat(grad_output_tensors, dim=0)

        return grad_input, None


def all2all(x, world_size):
    return All2AllFunction.apply(x, world_size)


# Example usage: torchrun --nproc_per_node=8 stage_1_all_to_all.py

if __name__ == "__main__":

    dist.init_process_group("nccl")
    world_size = dist.get_world_size()

    rank = dist.get_rank()
    torch.cuda.set_device(rank)

    x = torch.ones((16, 2), device=rank, requires_grad=True) * rank
    x.retain_grad()

    for i in range(10):
        out = all2all(x, world_size)
        loss = 0.5 * ((out - rank) ** 2).sum()
        # can you guess what the gradient will be?
        loss.backward(retain_graph=True)

    print(x.grad)
    # Think carefully!

import os
import torch
import torch.distributed as dist

def init_process_group(backend, use_cuda):
    """init torch distributed process group
    """

    if 'SLURM_NTASKS' not in os.environ:
        world_size = int(os.getenv("WORLD_SIZE", 1))
        world_rank = int(os.getenv("RANK", 0))
    else:
        world_size = int(os.getenv("SLURM_NTASKS", 1))
        world_rank = int(os.getenv("SLURM_PROCID", 0))
    dist.init_process_group(backend=backend, rank=world_rank, world_size=world_size)

    if use_cuda:
        if 'LOCAL_RANK' in os.environ:
            local_rank = int(os.getenv("LOCAL_RANK"))
        else:
            return world_rank % torch.cuda.device_count()
    else:
        local_rank = 0

    return local_rank, world_rank, world_size

# TODO: add async_op option
def all_reduce(input, group):
    input = input.contiguous()
    dist.all_reduce(input, group = group)
    return input

def split(input, group, dim, shapes):
    """Split the tensor along dim."""
    # Bypass the function if we are using only 1 GPU or if
    # communicator is not initialized
    size = dist.get_world_size(group)
    if size == 1:
        return input

    # Split along  dimension.
    if shapes is None:
        shapes = compute_split_shapes(input.shape[dim], size)
    input_list = list(torch.split(input, shapes, dim=dim))

    # Note: torch.split does not create contiguous tensors by default.
    rank = dist.get_rank(group)
    output = input_list[rank].contiguous()

    return output


def all_gather(input, group, dim, shapes):
    """
    Gather tensors and concatinate along the dimension dim_.
    """
    size = dist.get_world_size(group)
    if (shapes is not None) and (len(shapes) != size):
        raise ValueError(f"Error: passed shapes of size {len(shapes)} not equal to {size}")
    if dim >= input.dim():  # gathering along dim that doesnt exist
        raise ValueError(
            f"Error: Gathering along {dim} for a tensor of size {input.dim()}"
        )

    # Bypass the function if we are using only 1 GPU or if
    # communicator is not initialized
    if size == 1:
        return input

    input = input.contiguous()
    input_shape = list(input.shape)
    if shapes is not None:
        rank = dist.get_rank(group)
        if input_shape[dim] != shapes[rank]:
            raise ValueError(
                f"Error: local tensor size {input_shape[dim]} does not match shapes[{rank}] = {shapes[rank]}"
            )

        if len(set(shapes)) != 1:
            padded_shape = list(input_shape)
            padded_shape[dim] = max(shapes)
            padded = torch.zeros(padded_shape, dtype=input.dtype, device=input.device)
            padded.narrow(dim, 0, input_shape[dim]).copy_(input)
            input_list = [torch.empty_like(padded) for _ in range(size)]
            dist.all_gather(input_list, padded, group = group)
            return torch.cat(
                [tensor.narrow(dim, 0, shapes[src]) for src, tensor in enumerate(input_list)],
                dim=dim,
            ).contiguous()

        input_list = []
        for src in range(size):
            input_shape[dim] = shapes[src]
            input_list.append(
                torch.empty(input_shape, dtype=input.dtype, device=input.device)
            )
    else:
        # assume equal shape on all ranks
        input_list = [torch.empty_like(input) for _ in range(size)]

    dist.all_gather(input_list, input, group = group)
    output = torch.cat(input_list, dim=dim).contiguous()

    return output

def reduce_scatter(input, group, dim, shapes):
    """Reduce the tensor across ranks, then split the result along dim."""
    reduced = all_reduce(input, group)
    return split(reduced, group, dim, shapes)


def all_to_all_repartition(input, group, source_dim, dest_dim, source_shapes, dest_shapes):
    size = dist.get_world_size(group)
    rank = dist.get_rank(group)
    if size == 1:
        return input

    if source_shapes is None or dest_shapes is None:
        return None
    if len(set(source_shapes)) != 1 or len(set(dest_shapes)) != 1:
        return None
    if input.shape[source_dim] != source_shapes[rank]:
        return None
    if input.shape[dest_dim] != sum(dest_shapes):
        return None

    input_list = [chunk.contiguous() for chunk in torch.split(input.contiguous(), dest_shapes[0], dim=dest_dim)]
    output_list = [torch.empty_like(input_list[0]) for _ in range(size)]
    try:
        dist.all_to_all(output_list, input_list, group=group)
    except RuntimeError:
        return None
    return torch.cat(output_list, dim=source_dim).contiguous()


def roll_shards(input, group, shard_shift):
    size = dist.get_world_size(group)
    if size == 1 or shard_shift % size == 0:
        return input

    rank = dist.get_rank(group)
    send_rank = (rank + shard_shift) % size
    recv_rank = (rank - shard_shift) % size
    send_peer = dist.get_global_rank(group, send_rank)
    recv_peer = dist.get_global_rank(group, recv_rank)
    output = torch.empty_like(input)
    ops = [
        dist.P2POp(dist.isend, input.contiguous(), send_peer, group),
        dist.P2POp(dist.irecv, output, recv_peer, group),
    ]
    for request in dist.batch_isend_irecv(ops):
        request.wait()
    return output


def resolve_split_shapes(shapes, shard_dim, axis_name, group=None):
    if shapes is None:
        return None

    if isinstance(shapes, dict):
        if shard_dim not in shapes:
            raise ValueError(f"Missing split shapes for mesh dimension {shard_dim!r}")
        shapes = shapes[shard_dim]

        if isinstance(shapes, dict):
            if axis_name not in shapes:
                raise ValueError(
                    f"Missing split shapes for axis {axis_name!r} on mesh dimension {shard_dim!r}"
                )
            shapes = shapes[axis_name]

    if group is not None:
        size = dist.get_world_size(group)
        if len(shapes) != size:
            raise ValueError(f"Error: passed shapes of size {len(shapes)} not equal to {size}")

    return shapes

# helper routine to compute uneven splitting in balanced way:
def compute_split_shapes(size, num_chunks):
    if num_chunks == 1:
        return [size]

    # first, check if we can split using div-up to balance the load:
    chunk_size = (size + num_chunks - 1) // num_chunks
    last_chunk_size = max(0, size - chunk_size * (num_chunks - 1))
    if last_chunk_size == 0:
        # in this case, the last shard would be empty, split with floor instead:
        chunk_size = size // num_chunks
        last_chunk_size = size - chunk_size * (num_chunks - 1)

    # generate sections list
    sections = [chunk_size for _ in range(num_chunks - 1)] + [last_chunk_size]

    return sections

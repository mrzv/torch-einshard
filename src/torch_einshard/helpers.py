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
    if input.shape[source_dim] != source_shapes[rank]:
        return None
    if input.shape[dest_dim] != sum(dest_shapes):
        return None

    source_offsets = [sum(source_shapes[:i]) for i in range(size)]
    dest_offsets = [sum(dest_shapes[:i]) for i in range(size)]
    output_shape = list(input.shape)
    output_shape[source_dim] = sum(source_shapes)
    output_shape[dest_dim] = dest_shapes[rank]
    output = torch.empty(output_shape, dtype=input.dtype, device=input.device)
    ops = []
    recv_targets = []

    for peer in range(size):
        send_chunk = input.narrow(dest_dim, dest_offsets[peer], dest_shapes[peer]).contiguous()
        recv_chunk = output.narrow(source_dim, source_offsets[peer], source_shapes[peer])
        if peer == rank:
            recv_chunk.copy_(send_chunk)
            continue

        global_peer = dist.get_global_rank(group, peer)
        ops.append(dist.P2POp(dist.isend, send_chunk, global_peer, group))
        recv_buffer = torch.empty_like(recv_chunk)
        recv_targets.append((recv_buffer, recv_chunk))
        ops.append(dist.P2POp(dist.irecv, recv_buffer, global_peer, group))

    for request in dist.batch_isend_irecv(ops):
        request.wait()
    for recv_buffer, recv_chunk in recv_targets:
        recv_chunk.copy_(recv_buffer)
    return output.contiguous()


def owner_swap(input, mesh, source_shard_dims, dest_shard_dims, output_shape):
    device_mesh = getattr(mesh, "device_mesh", mesh)
    mesh_tensor = device_mesh.mesh
    dim_names = device_mesh.mesh_dim_names
    name_to_dim = {name: i for i, name in enumerate(dim_names)}
    rank = dist.get_rank()
    coord = (mesh_tensor == rank).nonzero()[0].tolist()

    send_coord = list(coord)
    recv_coord = list(coord)
    for source_shard_dim, dest_shard_dim in zip(source_shard_dims, dest_shard_dims):
        source_mesh_dim = name_to_dim[source_shard_dim]
        dest_mesh_dim = name_to_dim[dest_shard_dim]
        send_coord[dest_mesh_dim] = coord[source_mesh_dim]
        recv_coord[source_mesh_dim] = coord[dest_mesh_dim]

    send_rank = int(mesh_tensor[tuple(send_coord)].item())
    recv_rank = int(mesh_tensor[tuple(recv_coord)].item())
    if send_rank == rank and recv_rank == rank:
        return input.contiguous()

    output = torch.empty(output_shape, dtype=input.dtype, device=input.device)
    ops = [
        dist.P2POp(dist.isend, input.contiguous(), send_rank),
        dist.P2POp(dist.irecv, output, recv_rank),
    ]
    for request in dist.batch_isend_irecv(ops):
        request.wait()
    return output.contiguous()


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


def roll_sharded(input, group, dim, shift, shapes):
    size = dist.get_world_size(group)
    rank = dist.get_rank(group)
    if shapes is None or len(shapes) != size:
        return None

    total_size = sum(shapes)
    shift = shift % total_size
    if size == 1:
        return torch.roll(input, shifts=shift, dims=dim)
    if shift == 0:
        return input
    if input.shape[dim] != shapes[rank]:
        return None

    offsets = [sum(shapes[:i]) for i in range(size)]
    ends = [offsets[i] + shapes[i] for i in range(size)]

    def rank_for(global_offset):
        for i, end in enumerate(ends):
            if global_offset < end:
                return i
        return size - 1

    def chunks_for(src_rank):
        chunks = []
        src_size = shapes[src_rank]
        start = 0
        while start < src_size:
            dest_global = (offsets[src_rank] + start + shift) % total_size
            dest_rank = rank_for(dest_global)
            dest_offset = dest_global - offsets[dest_rank]
            length = min(src_size - start, ends[dest_rank] - dest_global)
            if length == 0:
                break
            chunks.append((start, length, dest_rank, dest_offset))
            start += length
        return chunks

    output = torch.empty_like(input)
    ops = []
    recv_targets = []
    for start, length, dest_rank, _ in chunks_for(rank):
        if dest_rank == rank:
            continue
        global_peer = dist.get_global_rank(group, dest_rank)
        ops.append(dist.P2POp(dist.isend, input.narrow(dim, start, length).contiguous(), global_peer, group))

    for src_rank in range(size):
        for start, length, dest_rank, dest_offset in chunks_for(src_rank):
            if dest_rank != rank:
                continue
            recv_chunk = output.narrow(dim, dest_offset, length)
            if src_rank == rank:
                recv_chunk.copy_(input.narrow(dim, start, length))
                continue
            global_peer = dist.get_global_rank(group, src_rank)
            recv_buffer = torch.empty_like(recv_chunk)
            recv_targets.append((recv_buffer, recv_chunk))
            ops.append(dist.P2POp(dist.irecv, recv_buffer, global_peer, group))

    for request in dist.batch_isend_irecv(ops):
        request.wait()
    for recv_buffer, recv_chunk in recv_targets:
        recv_chunk.copy_(recv_buffer)
    return output.contiguous()


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


def compute_split_shapes_for_factors(size, num_chunks, factor):
    result = compute_split_shapes(size // factor, num_chunks)
    if factor == 1:
        return result

    result = [section * factor for section in result]
    result[-1] += size % factor
    return result

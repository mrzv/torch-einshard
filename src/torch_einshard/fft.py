import math
import warnings
from collections.abc import Mapping

import torch
import torch.distributed as dist

from .distributed import distributed_1d
from .families import cached_expand_axis_families, expand_family_mapping
from .grammar import parse_sharding
from .mappings import allgather_forward_split_backward, split_forward_allgather_backward
from .helpers import resolve_split_shapes
from .sharding import Axis, AxisGroup, Axes, EllipsisAxis, TensorSpec


def _axes(spec):
    return spec.axes if hasattr(spec, "axes") else spec


def _axis_dim(axes, name):
    for dim, axis in enumerate(axes):
        if axis.name == name:
            return dim
    raise ValueError(f"Axis {name!r} is not present")


def _validate_spec(spec, *, label):
    if spec.partials:
        raise ValueError("einfft does not support partial tensor specs")

    names = []
    for axis in _axes(spec):
        if isinstance(axis, EllipsisAxis):
            raise NotImplementedError(f"einfft does not support ellipsis axes in {label}")
        if isinstance(axis, AxisGroup):
            raise NotImplementedError(f"einfft does not support factored axes in {label}")
        if axis.name in names:
            raise ValueError(f"Axis {axis.name!r} appears more than once in {label}")
        names.append(axis.name)
    return names


def _normalize_transform_axes(axes):
    if isinstance(axes, dict):
        return dict(axes)
    if isinstance(axes, (str, bytes)):
        raise TypeError("axes must be a mapping or iterable of axis names, not a string")
    try:
        return {source: source for source in axes}
    except TypeError as error:
        raise TypeError("axes must be a mapping or iterable of axis names") from error


def _expand_transform_axes(axes, families):
    if isinstance(axes, dict):
        return expand_family_mapping(axes, families, label="FFT axis")
    if isinstance(axes, (str, bytes)):
        return axes
    if not families:
        return axes

    result = []
    for axis in axes:
        result.extend(families.get(axis, (axis,)))
    return result


def _group(mesh, shard_dim):
    return mesh[shard_dim].get_group()


def _all_to_all_same_shape(input, group, build_send, build_output):
    size = dist.get_world_size(group)
    rank = dist.get_rank(group)
    output, recv_targets = build_output()
    ops = []

    for peer in range(size):
        send_chunk = build_send(peer).contiguous()
        recv_chunk = recv_targets[peer]
        if peer == rank:
            recv_chunk.copy_(send_chunk)
            continue

        global_peer = dist.get_global_rank(group, peer)
        ops.append(dist.P2POp(dist.isend, send_chunk, global_peer, group))
        recv_buffer = torch.empty_like(recv_chunk)
        ops.append(dist.P2POp(dist.irecv, recv_buffer, global_peer, group))
        recv_targets[peer] = (recv_buffer, recv_chunk)

    if ops:
        for request in dist.batch_isend_irecv(ops):
            request.wait()
    for target in recv_targets:
        if isinstance(target, tuple):
            recv_buffer, recv_chunk = target
            recv_chunk.copy_(recv_buffer)
    return output.contiguous()


def _starts(shapes):
    result = []
    offset = 0
    for shape in shapes:
        result.append(offset)
        offset += shape
    return result


def _owner(starts, shapes, index):
    for rank, (start, shape) in enumerate(zip(starts, shapes)):
        if start <= index < start + shape:
            return rank
    raise ValueError(f"Global index {index} is outside the sharded axis")


def _empty_axis_like(input, dim, size):
    shape = list(input.shape)
    shape[dim] = size
    return torch.empty(*shape, dtype=input.dtype, device=input.device)


def _prefix_repartition_no_autograd(input, group, dim, source_shapes, dest_shapes):
    rank = dist.get_rank(group)
    size = dist.get_world_size(group)
    source_starts = _starts(source_shapes)
    dest_starts = _starts(dest_shapes)
    source_start = source_starts[rank]
    dest_start = dest_starts[rank]
    source_end = source_start + source_shapes[rank]
    dest_end = dest_start + dest_shapes[rank]
    source_total = sum(source_shapes)

    z = input.movedim(dim, -1).contiguous()
    batch_shape = z.shape[:-1]
    z = z.reshape(-1, source_shapes[rank])
    batch = z.shape[0]

    send_chunks = []
    send_split_sizes = []
    recv_lengths = []
    recv_offsets = []
    recv_split_sizes = []

    for peer in range(size):
        peer_start = dest_starts[peer]
        peer_end = peer_start + dest_shapes[peer]
        start = max(source_start, peer_start)
        end = min(source_end, peer_end, source_total)
        length = max(0, end - start)
        send_chunks.append(z[:, start - source_start:start - source_start + length].reshape(-1))
        send_split_sizes.append(batch * length)

    for peer in range(size):
        peer_start = source_starts[peer]
        peer_end = peer_start + source_shapes[peer]
        start = max(dest_start, peer_start)
        end = min(dest_end, peer_end, source_total)
        length = max(0, end - start)
        recv_lengths.append(length)
        recv_offsets.append(start - dest_start)
        recv_split_sizes.append(batch * length)

    send = torch.cat(send_chunks) if send_chunks else z.new_empty(0)
    recv = z.new_empty(sum(recv_split_sizes))
    dist.all_to_all_single(recv, send, recv_split_sizes, send_split_sizes, group=group)

    output = z.new_zeros(batch, dest_shapes[rank])
    offset = 0
    for length, recv_offset, split_size in zip(recv_lengths, recv_offsets, recv_split_sizes):
        if split_size:
            output[:, recv_offset:recv_offset + length].copy_(recv[offset:offset + split_size].reshape(batch, length))
        offset += split_size
    return output.reshape(*batch_shape, dest_shapes[rank]).movedim(-1, dim).contiguous()


class _PrefixRepartition(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, group, dim, source_shapes, dest_shapes):
        ctx.group = group
        ctx.dim = dim
        ctx.source_shapes = source_shapes
        ctx.dest_shapes = dest_shapes
        return _prefix_repartition_no_autograd(input, group, dim, source_shapes, dest_shapes)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = _prefix_repartition_no_autograd(
            grad_output,
            ctx.group,
            ctx.dim,
            ctx.dest_shapes,
            ctx.source_shapes,
        )
        return grad_input, None, None, None, None


def _prefix_repartition(input, group, dim, source_shapes, dest_shapes):
    return _PrefixRepartition.apply(input, group, dim, source_shapes, dest_shapes)


def _fft_unscaled(input, *, dim, inverse):
    if inverse:
        return torch.fft.ifft(input, dim=dim) * input.shape[dim]
    return torch.fft.fft(input, dim=dim)


def _apply_fft_norm(input, *, n, inverse, norm):
    if norm == "ortho":
        return input / math.sqrt(n)
    if inverse:
        if norm in (None, "backward"):
            return input / n
        if norm == "forward":
            return input
    else:
        if norm in (None, "backward"):
            return input
        if norm == "forward":
            return input / n
    raise ValueError(f"Invalid FFT norm {norm!r}")


def _distributed_fft_1d_no_autograd(input, group, dim, *, inverse, norm):
    size = dist.get_world_size(group)
    rank = dist.get_rank(group)
    local_size = input.shape[dim]
    chunk = local_size // size
    total_size = local_size * size

    z = input.movedim(dim, -1).contiguous()
    batch_shape = z.shape[:-1]
    z = z.reshape(-1, local_size)
    batch = z.shape[0]

    def build_m_output():
        output = torch.empty(batch, size, chunk, dtype=z.dtype, device=z.device)
        return output, [output[:, peer, :] for peer in range(size)]

    z = _all_to_all_same_shape(
        z,
        group,
        lambda peer: z[:, peer * chunk:(peer + 1) * chunk],
        build_m_output,
    )
    z = _fft_unscaled(z, dim=1, inverse=inverse)

    m = torch.arange(rank * chunk, (rank + 1) * chunk, device=z.device)
    k2 = torch.arange(size, device=z.device)
    sign = 1 if inverse else -1
    phase = sign * 2j * torch.pi * k2[:, None] * m[None, :] / total_size
    z = z * torch.exp(phase).to(dtype=z.dtype)

    def build_k2_output():
        output = torch.empty(batch, size, chunk, dtype=z.dtype, device=z.device)
        return output, [output[:, peer, :] for peer in range(size)]

    z = _all_to_all_same_shape(
        z,
        group,
        lambda peer: z[:, peer, :],
        build_k2_output,
    )
    z = z.reshape(batch, local_size)
    z = _fft_unscaled(z, dim=1, inverse=inverse)

    def build_output():
        output = torch.empty(batch, size, chunk, dtype=z.dtype, device=z.device)
        return output, [output[:, peer, :] for peer in range(size)]

    z = _all_to_all_same_shape(
        z,
        group,
        lambda peer: z[:, peer * chunk:(peer + 1) * chunk],
        build_output,
    )
    z = z.permute(0, 2, 1).reshape(batch, local_size)
    z = _apply_fft_norm(z, n=total_size, inverse=inverse, norm=norm)
    return z.reshape(*batch_shape, local_size).movedim(-1, dim).contiguous()


def _adjoint_fft(input, group, dim, shapes, *, inverse, norm):
    if norm == "ortho":
        return _distributed_fft_1d_no_autograd(input, group, dim, inverse=not inverse, norm="ortho")
    if inverse:
        adjoint_norm = "forward" if norm in (None, "backward") else None
        return _distributed_fft_1d_no_autograd(input, group, dim, inverse=False, norm=adjoint_norm)

    adjoint_norm = None if norm == "forward" else "forward"
    return _distributed_fft_1d_no_autograd(input, group, dim, inverse=True, norm=adjoint_norm)


class _DistributedFFT1D(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, group, dim, shapes, inverse, norm):
        ctx.group = group
        ctx.dim = dim
        ctx.shapes = shapes
        ctx.inverse = inverse
        ctx.norm = norm
        return _distributed_fft_1d_no_autograd(input, group, dim, inverse=inverse, norm=norm)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = _adjoint_fft(
            grad_output,
            ctx.group,
            ctx.dim,
            ctx.shapes,
            inverse=ctx.inverse,
            norm=ctx.norm,
        )
        return grad_input, None, None, None, None, None


def _distributed_fft_1d(input, group, dim, shapes, *, inverse, norm):
    return _DistributedFFT1D.apply(input, group, dim, shapes, inverse, norm)


def _distributed_fft_axis_plan(x, source, target, input_axis, output_axis, input_axes, shapes, mesh):
    if input_axis.local() or output_axis.local():
        return None
    if input_axis.shard_dim != output_axis.shard_dim:
        return None

    comm = _group(mesh, input_axis.shard_dim)
    split_shapes = resolve_split_shapes(shapes, input_axis.shard_dim, input_axis.name, comm)
    output_shapes = resolve_split_shapes(shapes, output_axis.shard_dim, output_axis.name, comm)
    size = dist.get_world_size(comm)
    rank = dist.get_rank(comm)
    dim = _axis_dim(input_axes, source)
    if split_shapes is None:
        split_shapes = [x.shape[dim]] * size
    if output_shapes is None:
        output_shapes = split_shapes
    if split_shapes != output_shapes or len(set(split_shapes)) != 1:
        return None

    local_size = split_shapes[rank]
    if x.shape[dim] != local_size or local_size != split_shapes[0] or local_size % size != 0:
        return None
    return comm, dim, split_shapes, sum(split_shapes)


def _fast_fft_plan(x, transform_axes, input_by_name, output_by_name, input_axes, shapes, mesh, *, require_complex=True):
    if require_complex and not torch.is_complex(x):
        return None

    plan = []
    sharded_dims = set()
    for source, target in transform_axes.items():
        dim = _axis_dim(input_axes, source)
        input_axis = input_by_name[source]
        output_axis = output_by_name[target]
        if not output_axis.local():
            if output_axis.shard_dim in sharded_dims:
                return None
            sharded_dims.add(output_axis.shard_dim)
        if input_axis.local():
            plan.append(("local", source, target, dim, x.shape[dim]))
            continue
        if mesh is None or output_axis.local() or input_axis.shard_dim != output_axis.shard_dim:
            return None
        axis_plan = _distributed_fft_axis_plan(
            x,
            source,
            target,
            input_axis,
            output_axis,
            input_axes,
            shapes,
            mesh,
        )
        if axis_plan is None:
            return None
        comm, dim, split_shapes, global_size = axis_plan
        plan.append(("distributed", source, target, dim, global_size, comm, split_shapes))
    return plan


def _apply_fast_fft_plan(z, current_axes, output_by_name, plan, *, inverse, norm):
    total_size = 1
    per_axis_norm = "forward" if inverse else None
    for item in plan:
        total_size *= item[4]
        if item[0] == "local":
            _, source, target, dim, _ = item
            z = _fft_unscaled(z, dim=dim, inverse=inverse)
            current_axes[dim] = Axis(target)
        else:
            _, source, target, dim, _, comm, split_shapes = item
            z = _distributed_fft_1d(z, comm, dim, split_shapes, inverse=inverse, norm=per_axis_norm)
            current_axes[dim] = Axis(target, output_by_name[target].shard_dim)
    return _apply_fft_norm(z, n=total_size, inverse=inverse, norm=norm)


def _fast_real_fft_plan(x, transform_axes, input_by_name, output_by_name, input_axes, shapes, mesh, *, inverse, signal_sizes):
    half_source = next(reversed(transform_axes))
    half_target = transform_axes[half_source]
    half_input = input_by_name[half_source]
    half_output = output_by_name[half_target]
    if not inverse and not half_input.local() and not half_output.local() and half_input.shard_dim == half_output.shard_dim:
        if not _is_supported_real_fft_input(x):
            return None
        if mesh is None:
            return None
        comm = _group(mesh, half_input.shard_dim)
        full_shapes = resolve_split_shapes(shapes, half_input.shard_dim, half_input.name, comm)
        size = dist.get_world_size(comm)
        rank = dist.get_rank(comm)
        dim = _axis_dim(input_axes, half_source)
        if full_shapes is None:
            full_shapes = [x.shape[dim]] * size
        if len(set(full_shapes)) != 1:
            return None
        local_size = full_shapes[rank]
        if x.shape[dim] != local_size or local_size % size != 0:
            return None
        global_size = sum(full_shapes)
        half_shapes = resolve_split_shapes(shapes, half_output.shard_dim, half_output.name, comm)
        if half_shapes is None:
            return None
        if sum(half_shapes) != global_size // 2 + 1:
            return None

        complex_axes = {source: target for source, target in transform_axes.items() if source != half_source}
        if any(output_by_name[target].shard_dim == half_output.shard_dim for target in complex_axes.values()):
            return None
        complex_plan = []
        if complex_axes:
            complex_plan = _fast_fft_plan(
                x,
                complex_axes,
                input_by_name,
                output_by_name,
                input_axes,
                shapes,
                mesh,
                require_complex=False,
            )
            if complex_plan is None:
                return None
        return ("sharded_half_forward", half_source, half_target, dim, global_size, comm, full_shapes, half_shapes, complex_plan)

    if not half_input.local() or not half_output.local():
        return None
    if not inverse and not _is_supported_real_fft_input(x):
        return None

    complex_axes = {source: target for source, target in transform_axes.items() if source != half_source}
    if not complex_axes:
        return None

    plan = _fast_fft_plan(
        x,
        complex_axes,
        input_by_name,
        output_by_name,
        input_axes,
        shapes,
        mesh,
        require_complex=False,
    )
    if plan is None:
        return None

    if inverse and signal_sizes is not None:
        if not isinstance(signal_sizes, Mapping):
            return plan
        plan_sizes = {item[1]: item[4] for item in plan}
        for source, target in complex_axes.items():
            requested_size = signal_sizes.get(target, signal_sizes.get(source))
            if requested_size is None:
                continue
            axis_size = plan_sizes.get(source, x.shape[_axis_dim(input_axes, source)])
            if requested_size != axis_size:
                return None

    return plan


def _complex_dtype(dtype):
    if dtype in (torch.float64, torch.complex128):
        return torch.complex128
    return torch.complex64


def _is_supported_real_fft_input(x):
    return x.dtype in (torch.float32, torch.float64)


def _apply_fast_real_fft_plan(
    z,
    current_axes,
    output_by_name,
    transform_axes,
    plan,
    *,
    inverse,
    norm,
    signal_sizes,
):
    half_source = next(reversed(transform_axes))
    half_target = transform_axes[half_source]
    if isinstance(plan, tuple) and plan[0] == "sharded_half_forward":
        _, _, _, half_dim, half_full_size, comm, full_shapes, half_shapes, complex_plan = plan
        total_size = half_full_size
        for item in complex_plan:
            total_size *= item[4]

        z = z.to(_complex_dtype(z.dtype))
        z = _distributed_fft_1d(z, comm, half_dim, full_shapes, inverse=False, norm=None)
        z = _prefix_repartition(z, comm, half_dim, full_shapes, half_shapes)
        current_axes[half_dim] = Axis(half_target, output_by_name[half_target].shard_dim)

        for item in complex_plan:
            if item[0] == "local":
                _, source, target, dim, _ = item
                z = _fft_unscaled(z, dim=dim, inverse=False)
                current_axes[dim] = Axis(target)
            else:
                _, source, target, dim, _, axis_comm, split_shapes = item
                z = _distributed_fft_1d(z, axis_comm, dim, split_shapes, inverse=False, norm=None)
                current_axes[dim] = Axis(target, output_by_name[target].shard_dim)

        return _apply_fft_norm(z, n=total_size, inverse=False, norm=norm)

    sizes = _real_signal_sizes(transform_axes, current_axes, z, signal_sizes) if inverse else None
    half_size = sizes[-1] if inverse else None
    plan_sizes = {item[1]: item[4] for item in plan}
    total_size = math.prod(
        plan_sizes.get(source, sizes[i] if inverse else z.shape[_axis_dim(current_axes, source)])
        for i, source in enumerate(transform_axes)
    )
    per_axis_norm = "forward" if inverse else None

    if inverse:
        for item in plan:
            if item[0] == "local":
                _, source, target, dim, _ = item
                z = _fft_unscaled(z, dim=dim, inverse=True)
                current_axes[dim] = Axis(target)
            else:
                _, source, target, dim, _, comm, split_shapes = item
                z = _distributed_fft_1d(z, comm, dim, split_shapes, inverse=True, norm=per_axis_norm)
                current_axes[dim] = Axis(target, output_by_name[target].shard_dim)

        half_dim = _axis_dim(current_axes, half_source)
        z = torch.fft.irfft(z, n=half_size, dim=half_dim) * half_size
        current_axes[half_dim] = Axis(half_target)
    else:
        half_dim = _axis_dim(current_axes, half_source)
        z = torch.fft.rfft(z, dim=half_dim)
        current_axes[half_dim] = Axis(half_target)

        for item in plan:
            if item[0] == "local":
                _, source, target, dim, _ = item
                z = _fft_unscaled(z, dim=dim, inverse=False)
                current_axes[dim] = Axis(target)
            else:
                _, source, target, dim, _, comm, split_shapes = item
                z = _distributed_fft_1d(z, comm, dim, split_shapes, inverse=False, norm=per_axis_norm)
                current_axes[dim] = Axis(target, output_by_name[target].shard_dim)

    return _apply_fft_norm(z, n=total_size, inverse=inverse, norm=norm)


def _warn_slow_path():
    warnings.warn(
        "einfft is using the gather/FFT/split fallback for a sharded transform axis; "
        "this is correct but materializes the full transform axis on every rank.",
        RuntimeWarning,
        stacklevel=3,
    )


def _fast_path_name(fast_path, *, real):
    if fast_path is None:
        return "fallback"
    if real and isinstance(fast_path, tuple) and fast_path[0] == "sharded_half_forward":
        return "distributed-rfft-sharded-half"
    if real:
        return "distributed-rfft-local-half"
    return "distributed-fftn"


def _fallback_reason(x, transform_axes, input_by_name, output_by_name, input_axes, shapes, mesh, *, real, inverse, signal_sizes):
    if real:
        half_source = next(reversed(transform_axes))
        half_target = transform_axes[half_source]
        half_input = input_by_name[half_source]
        half_output = output_by_name[half_target]
        if not inverse and not _is_supported_real_fft_input(x):
            return "forward real FFT fast path requires float32 or float64 input"
        if inverse and (not half_input.local() or not half_output.local()):
            return "inverse real FFT with a sharded half-spectrum axis is not optimized yet"
        if half_input.local() != half_output.local():
            return "real FFT half-spectrum axis must be local on both sides or sharded on the same mesh dimension"
        if not half_input.local() and half_input.shard_dim != half_output.shard_dim:
            return "real FFT half-spectrum axis must stay on the same mesh dimension"
        if inverse and signal_sizes is not None and isinstance(signal_sizes, Mapping):
            for source, target in list(transform_axes.items())[:-1]:
                requested_size = signal_sizes.get(target, signal_sizes.get(source))
                if requested_size is not None:
                    dim = _axis_dim(input_axes, source)
                    actual_size = None
                    if input_by_name[source].local():
                        actual_size = x.shape[dim]
                    elif mesh is not None:
                        axis = input_by_name[source]
                        comm = _group(mesh, axis.shard_dim)
                        split_shapes = resolve_split_shapes(shapes, axis.shard_dim, axis.name, comm)
                        if split_shapes is not None:
                            actual_size = sum(split_shapes)
                        else:
                            actual_size = x.shape[dim] * dist.get_world_size(comm)
                    if actual_size is not None and requested_size != actual_size:
                        return "inverse real FFT non-half-axis signal_sizes request padding or cropping"
    if mesh is None and any(not input_by_name[source].local() or not output_by_name[target].local() for source, target in transform_axes.items()):
        return "distributed FFT layouts require mesh"
    if not real and not torch.is_complex(x):
        return "complex FFT fast path requires complex input"
    for source, target in transform_axes.items():
        input_axis = input_by_name[source]
        output_axis = output_by_name[target]
        if not input_axis.local() and output_axis.local():
            return "sharded transform axes must stay sharded for the fast path"
        if input_axis.local() and not output_axis.local():
            return "local transform axes cannot become sharded in the fast path"
        if not input_axis.local() and input_axis.shard_dim != output_axis.shard_dim:
            return "sharded transform axes must stay on the same mesh dimension"
        if not input_axis.local() and mesh is not None:
            comm = _group(mesh, input_axis.shard_dim)
            split_shapes = resolve_split_shapes(shapes, input_axis.shard_dim, input_axis.name, comm)
            if split_shapes is not None:
                rank = dist.get_rank(comm)
                if len(set(split_shapes)) != 1:
                    return "fast path requires equal shard sizes"
                if split_shapes[rank] % dist.get_world_size(comm) != 0:
                    return "fast path requires local shard size divisible by mesh size"
    return "layout is not supported by an optimized FFT path"


def _shared_transform_mesh_dim_reason(transform_axes, input_by_name, output_by_name):
    for source, target in transform_axes.items():
        axis = input_by_name[source]
        if axis.local():
            axis = output_by_name[target]
        if not axis.local():
            for other_name, other_axis in input_by_name.items():
                if other_name == source or other_axis.local():
                    continue
                if other_axis.shard_dim == axis.shard_dim:
                    return "einfft transform axes cannot share a mesh dimension with another input or output axis"

        output_axis = output_by_name[target]
        if output_axis.local():
            continue
        for other_name, other_axis in output_by_name.items():
            if other_name == target or other_axis.local():
                continue
            if other_axis.shard_dim == output_axis.shard_dim:
                return "einfft transform axes cannot share a mesh dimension with another input or output axis"
    return None


def _real_signal_sizes(transform_axes, current_axes, x, signal_sizes):
    if signal_sizes is None:
        signal_sizes = {}
    elif not isinstance(signal_sizes, Mapping):
        raise TypeError("signal_sizes must be a mapping from axis names to sizes")

    result = []
    for source, target in transform_axes.items():
        if target in signal_sizes:
            result.append(signal_sizes[target])
        elif source in signal_sizes:
            result.append(signal_sizes[source])
        else:
            result.append(None)

    if result[-1] is None:
        dim = _axis_dim(current_axes, next(reversed(transform_axes)))
        result[-1] = 2 * (x.shape[dim] - 1)
    for i, size in enumerate(result):
        if size is None:
            dim = _axis_dim(current_axes, list(transform_axes)[i])
            result[i] = x.shape[dim]
    return tuple(result)


def einfft(
    shard,
    x,
    axes,
    *,
    inverse=False,
    norm=None,
    mesh=None,
    shapes=None,
    families=None,
    real=False,
    signal_sizes=None,
    explain=False,
):
    """Apply a named-axis FFT with sharding-aware gather/split fallback."""
    shard, _ = cached_expand_axis_families(shard, families=families)
    axes = _expand_transform_axes(axes, families)
    transform_axes = _normalize_transform_axes(axes)
    input_spec, other_spec, output_spec = parse_sharding(shard)
    if other_spec is not None:
        raise ValueError("einfft expects a unary sharding expression")

    input_names = _validate_spec(input_spec, label="input")
    output_names = _validate_spec(output_spec, label="output")
    if len(input_names) != x.dim():
        raise ValueError("Input tensor rank does not match the input spec")
    if len(input_names) != len(output_names):
        raise ValueError("Input and output specs must have the same rank")

    if not transform_axes:
        if sorted(input_names) != sorted(output_names):
            raise ValueError("Output axes must match input axes when no FFT axes are requested")
        if explain:
            layout_is_noop = [axis.name for axis in _axes(input_spec)] == output_names and _axes(input_spec) == _axes(output_spec)
            return {
                "fast_path": False,
                "path": "no-op-layout" if layout_is_noop else "layout-only",
                "reason": "no transform axes requested" if layout_is_noop else "no transform axes requested; only layout changes would run",
                "axes": {},
            }
        return distributed_1d((input_spec, None, output_spec), x, mesh=mesh, shapes=shapes)

    for source, target in transform_axes.items():
        if source not in input_names:
            raise ValueError(f"FFT source axis {source!r} is not present in the input")
        if target not in output_names:
            raise ValueError(f"FFT output axis {target!r} is not present in the output")
    if len(set(transform_axes)) != len(transform_axes):
        raise ValueError("FFT source axes must be distinct")
    if len(set(transform_axes.values())) != len(transform_axes):
        raise ValueError("FFT output axes must be distinct")

    expected_output_names = [transform_axes.get(name, name) for name in input_names]
    if sorted(expected_output_names) != sorted(output_names):
        raise ValueError("Output axes must match input axes with FFT axes renamed")

    input_by_name = {axis.name: axis for axis in _axes(input_spec)}
    output_by_name = {axis.name: axis for axis in _axes(output_spec)}
    for name in input_names:
        if name in transform_axes:
            continue
        if input_by_name[name] != output_by_name[name]:
            raise ValueError("einfft output must preserve non-FFT axis sharding")

    current_axes = Axes(Axis(axis.name, axis.shard_dim) for axis in _axes(input_spec))
    z = x

    shared_mesh_reason = _shared_transform_mesh_dim_reason(transform_axes, input_by_name, output_by_name)
    if shared_mesh_reason is not None:
        if explain:
            return {
                "fast_path": False,
                "path": "unsupported-layout",
                "reason": shared_mesh_reason,
                "axes": dict(transform_axes),
            }
        raise NotImplementedError(shared_mesh_reason)

    fast_path = None
    if real:
        fast_path = _fast_real_fft_plan(
            z,
            transform_axes,
            input_by_name,
            output_by_name,
            current_axes,
            shapes,
            mesh,
            inverse=inverse,
            signal_sizes=signal_sizes,
        )
    else:
        fast_path = _fast_fft_plan(
            z,
            transform_axes,
            input_by_name,
            output_by_name,
            current_axes,
            shapes,
            mesh,
        )
    if fast_path is not None:
        if explain:
            return {
                "fast_path": True,
                "path": _fast_path_name(fast_path, real=real),
                "reason": None,
                "axes": dict(transform_axes),
            }
        if real:
            z = _apply_fast_real_fft_plan(
                z,
                current_axes,
                output_by_name,
                transform_axes,
                fast_path,
                inverse=inverse,
                norm=norm,
                signal_sizes=signal_sizes,
            )
        else:
            z = _apply_fast_fft_plan(z, current_axes, output_by_name, fast_path, inverse=inverse, norm=norm)
        current_spec = TensorSpec(current_axes)
        if [axis.name for axis in current_axes] == output_names and current_axes == _axes(output_spec):
            return z
        return distributed_1d((current_spec, None, output_spec), z, mesh=mesh, shapes=shapes)

    if explain:
        return {
            "fast_path": False,
            "path": "fallback",
            "reason": _fallback_reason(
                z,
                transform_axes,
                input_by_name,
                output_by_name,
                current_axes,
                shapes,
                mesh,
                real=real,
                inverse=inverse,
                signal_sizes=signal_sizes,
            ),
            "axes": dict(transform_axes),
        }

    if any(
        not input_by_name[source].local() or (real and not output_by_name[target].local())
        for source, target in transform_axes.items()
    ):
        _warn_slow_path()

    fft_dims = []
    for source in transform_axes:
        dim = _axis_dim(current_axes, source)
        fft_dims.append(dim)
        axis = current_axes[dim]
        if axis.local():
            continue
        if mesh is None:
            raise ValueError("Distributed einfft operations require mesh")
        comm = _group(mesh, axis.shard_dim)
        split_shapes = resolve_split_shapes(shapes, axis.shard_dim, axis.name, comm)
        z = allgather_forward_split_backward(z, comm, dim, split_shapes)
        current_axes[dim] = Axis(axis.name)

    if real:
        if inverse:
            z = torch.fft.irfftn(
                z,
                s=_real_signal_sizes(transform_axes, current_axes, z, signal_sizes),
                dim=tuple(fft_dims),
                norm=norm,
            )
        else:
            z = torch.fft.rfftn(z, dim=tuple(fft_dims), norm=norm)
    else:
        fft = torch.fft.ifftn if inverse else torch.fft.fftn
        z = fft(z, dim=tuple(fft_dims), norm=norm)

    for source, target in transform_axes.items():
        dim = _axis_dim(current_axes, source)
        output_axis = output_by_name[target]
        current_axes[dim] = Axis(target)
        if output_axis.local():
            continue
        if mesh is None:
            raise ValueError("Distributed einfft operations require mesh")
        comm = _group(mesh, output_axis.shard_dim)
        split_shapes = resolve_split_shapes(shapes, output_axis.shard_dim, output_axis.name, comm)
        z = split_forward_allgather_backward(z, comm, dim, split_shapes)
        current_axes[dim] = Axis(target, output_axis.shard_dim)

    current_spec = TensorSpec(current_axes)
    if [axis.name for axis in current_axes] == output_names and current_axes == _axes(output_spec):
        return z

    return distributed_1d(
        (current_spec, None, output_spec),
        z,
        mesh=mesh,
        shapes=shapes,
    )

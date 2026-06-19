import math
import warnings

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


def _fast_fft_plan(x, transform_axes, input_by_name, output_by_name, input_axes, shapes, mesh):
    if not torch.is_complex(x):
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


def _warn_slow_path():
    warnings.warn(
        "einfft is using the gather/FFT/split fallback for a sharded transform axis; "
        "this is correct but materializes the full transform axis on every rank.",
        RuntimeWarning,
        stacklevel=3,
    )


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
):
    """Apply a named-axis complex FFT with sharding-aware gather/split fallback."""
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
        z = _apply_fast_fft_plan(z, current_axes, output_by_name, fast_path, inverse=inverse, norm=norm)
        current_spec = TensorSpec(current_axes)
        if [axis.name for axis in current_axes] == output_names and current_axes == _axes(output_spec):
            return z
        return distributed_1d((current_spec, None, output_spec), z, mesh=mesh, shapes=shapes)

    if any(not input_by_name[source].local() for source in transform_axes):
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

import string
import torch

from .sharding import AxisGroup, EllipsisAxis


def _axes(spec):
    return spec.axes if hasattr(spec, "axes") else spec


def _flat_axes(spec):
    axes = _axes(spec)
    return axes.flat() if hasattr(axes, "flat") else axes


def _has_groups(spec):
    return any(isinstance(axis, AxisGroup) for axis in _axes(spec))


def _append_axis(equation, axis, shard_to_einsum, idx, name_only):
    if isinstance(axis, EllipsisAxis):
        return equation + "...", idx
    if name_only:
        axis = axis.name
    if axis not in shard_to_einsum:
        shard_to_einsum[axis] = string.ascii_lowercase[idx]
        idx += 1
    return equation + shard_to_einsum[axis], idx


def _resolve_group_shape(dim_size, group, sizes):
    known = []
    unknown = []
    product = 1
    for axis in group:
        size = None if sizes is None else sizes.get(axis.name)
        if size is None:
            unknown.append(axis)
            known.append(None)
        else:
            product *= size
            known.append(size)

    if len(unknown) > 1:
        names = ", ".join(axis.name for axis in unknown)
        raise ValueError(f"Cannot infer multiple factor sizes for {names}")
    if len(unknown) == 1:
        if dim_size % product != 0:
            raise ValueError(f"Cannot infer factor size for {unknown[0].name!r} from dimension {dim_size}")
        known[known.index(None)] = dim_size // product
    elif product != dim_size:
        raise ValueError(f"Factor sizes multiply to {product}, but tensor dimension is {dim_size}")
    return known


def _expand_groups(input, spec, sizes):
    if not _has_groups(spec):
        return input

    shape = []
    dim = 0
    fixed_dims = sum(1 for axis in _axes(spec) if not isinstance(axis, EllipsisAxis))
    ellipsis_dims = input.dim() - fixed_dims
    for axis in _axes(spec):
        if isinstance(axis, EllipsisAxis):
            shape.extend(input.shape[dim:dim + ellipsis_dims])
            dim += ellipsis_dims
        elif isinstance(axis, AxisGroup):
            shape.extend(_resolve_group_shape(input.shape[dim], axis, sizes))
            dim += 1
        else:
            shape.append(input.shape[dim])
            dim += 1
    return input.reshape(shape)


def _pack_groups(output, spec):
    if not _has_groups(spec):
        return output

    shape = []
    dim = 0
    fixed_dims = sum(len(axis) if isinstance(axis, AxisGroup) else 1 for axis in _axes(spec) if not isinstance(axis, EllipsisAxis))
    ellipsis_dims = output.dim() - fixed_dims
    for axis in _axes(spec):
        if isinstance(axis, EllipsisAxis):
            shape.extend(output.shape[dim:dim + ellipsis_dims])
            dim += ellipsis_dims
        elif isinstance(axis, AxisGroup):
            size = 1
            for _ in axis:
                size *= output.shape[dim]
                dim += 1
            shape.append(size)
        else:
            shape.append(output.shape[dim])
            dim += 1
    return output.reshape(shape)

# local contraction: translate the formula to einsum
def einsum(shard, *xs, name_only = False, sizes = None):
    if not name_only:
        assert not getattr(shard[0], "partials", ()), "Local einsum does not support partial inputs"
        if shard[1] is not None:
            assert not getattr(shard[1], "partials", ()), "Local einsum does not support partial inputs"

    shard_to_einsum = {}
    idx = 0
    equation = ""

    if not name_only:
        xs = [_expand_groups(xs[0], shard[0], sizes)] + list(xs[1:])
        if shard[1] is not None:
            xs[1] = _expand_groups(xs[1], shard[1], sizes)

    for i,n in enumerate(_flat_axes(shard[0])):
        equation, idx = _append_axis(equation, n, shard_to_einsum, idx, name_only)

    if shard[1] is not None:
        equation += ','
        for i,n in enumerate(_flat_axes(shard[1])):
            equation, idx = _append_axis(equation, n, shard_to_einsum, idx, name_only)
    equation += '->'
    for i,n in enumerate(_flat_axes(shard[2])):
        if isinstance(n, EllipsisAxis):
            equation += "..."
            continue
        key = n.name if name_only else n
        assert key in shard_to_einsum, f"Output dimension {n} must be present in input"
        equation += shard_to_einsum[key]

    output = torch.einsum(equation, *xs)
    if name_only:
        return output
    return _pack_groups(output, shard[2])

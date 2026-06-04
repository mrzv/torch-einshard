import string
import torch

from .sharding import AxisGroup


def _axes(spec):
    return spec.axes if hasattr(spec, "axes") else spec


def _flat_axes(spec):
    axes = _axes(spec)
    return axes.flat() if hasattr(axes, "flat") else axes


def _has_groups(spec):
    return any(isinstance(axis, AxisGroup) for axis in _axes(spec))


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
    for axis in _axes(spec):
        if isinstance(axis, AxisGroup):
            shape.extend(_resolve_group_shape(input.shape[dim], axis, sizes))
        else:
            shape.append(input.shape[dim])
        dim += 1
    return input.reshape(shape)


def _pack_groups(output, spec):
    if not _has_groups(spec):
        return output

    shape = []
    dim = 0
    for axis in _axes(spec):
        if isinstance(axis, AxisGroup):
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
        if name_only:
            n = n.name
        if n not in shard_to_einsum:
            shard_to_einsum[n] = string.ascii_lowercase[idx]
            idx += 1
        equation += shard_to_einsum[n]

    if shard[1] is not None:
        equation += ','
        for i,n in enumerate(_flat_axes(shard[1])):
            if name_only:
                n = n.name
            if n not in shard_to_einsum:
                shard_to_einsum[n] = string.ascii_lowercase[idx]
                idx += 1
            equation += shard_to_einsum[n]
    equation += '->'
    for i,n in enumerate(_flat_axes(shard[2])):
        if name_only:
            n = n.name
        assert n in shard_to_einsum, f"Output dimension {n} must be present in input"
        equation += shard_to_einsum[n]

    output = torch.einsum(equation, *xs)
    if name_only:
        return output
    return _pack_groups(output, shard[2])

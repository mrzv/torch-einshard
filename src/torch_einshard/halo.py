import torch
import torch.nn.functional as F
import torch.distributed as dist

from .families import cached_expand_axis_families, expand_family_mapping
from .grammar import parse_sharding
from .helpers import resolve_split_shapes
from .mappings import allgather_forward_reducescatter_backward
from .sharding import EllipsisAxis


def _axes(spec):
    return spec.axes if hasattr(spec, "axes") else spec


def _normalize_halo(value):
    if isinstance(value, int):
        if value < 0:
            raise ValueError("Halo widths must be non-negative")
        return value, value

    left, right = value
    if left < 0 or right < 0:
        raise ValueError("Halo widths must be non-negative")
    return left, right


def _pad_dim(x, dim, left, right, *, boundary, fill):
    if left == 0 and right == 0:
        return x

    if boundary == "periodic":
        if x.shape[dim] == 0:
            raise ValueError("Cannot periodic-pad an empty axis")
        pieces = []
        if left:
            index = torch.arange(x.shape[dim] - left, x.shape[dim], device=x.device).remainder(x.shape[dim])
            pieces.append(torch.index_select(x, dim, index))
        pieces.append(x)
        if right:
            index = torch.arange(right, device=x.device).remainder(x.shape[dim])
            pieces.append(torch.index_select(x, dim, index))
        return torch.cat(pieces, dim=dim).contiguous()

    if boundary == "replicate":
        if x.shape[dim] == 0:
            raise ValueError("Cannot replicate-pad an empty axis")
        index = torch.arange(-left, x.shape[dim] + right, device=x.device).clamp(0, x.shape[dim] - 1)
        return torch.index_select(x, dim, index).contiguous()

    padding = [0, 0] * x.dim()
    padding[2 * (x.dim() - dim - 1)] = left
    padding[2 * (x.dim() - dim - 1) + 1] = right
    if boundary == "constant":
        return F.pad(x, padding, mode="constant", value=fill)
    raise ValueError(f"Unsupported halo boundary {boundary!r}")


def _axis_dim(axes, axis_name):
    for dim, axis in enumerate(axes):
        if isinstance(axis, EllipsisAxis):
            raise NotImplementedError("einhalo does not support ellipsis axes")
        if axis.name == axis_name:
            return dim
    raise ValueError(f"Axis {axis_name!r} is not present in the tensor spec")


def _validate_unique_axis_names(axes):
    seen = set()
    for axis in axes:
        if isinstance(axis, EllipsisAxis):
            raise NotImplementedError("einhalo does not support ellipsis axes")
        if not hasattr(axis, "name"):
            raise NotImplementedError("einhalo does not support factored axes")
        if axis.name in seen:
            raise ValueError(f"Axis {axis.name!r} appears more than once")
        seen.add(axis.name)


def _take_periodic_range(x, dim, start, length):
    size = x.shape[dim]
    index = torch.arange(start, start + length, device=x.device).remainder(size)
    return torch.index_select(x, dim, index)


def _halo_sharded_axis(x, axis, dim, left, right, *, mesh, shapes, boundary, fill):
    if mesh is None:
        raise ValueError("Sharded einhalo axes require mesh")

    group = mesh[axis.shard_dim].get_group()
    size = dist.get_world_size(group)
    rank = dist.get_rank(group)
    split_shapes = resolve_split_shapes(shapes, axis.shard_dim, axis.name, group)
    if split_shapes is None:
        split_shapes = [x.shape[dim]] * size
    if x.shape[dim] != split_shapes[rank]:
        raise ValueError(
            f"Local tensor size {x.shape[dim]} does not match split shape {split_shapes[rank]} "
            f"for axis {axis.name!r}"
        )

    full = allgather_forward_reducescatter_backward(x, group, dim, split_shapes)
    total = sum(split_shapes)
    if boundary == "periodic" and total == 0:
        raise ValueError("Cannot periodic-pad an empty axis")
    start = sum(split_shapes[:rank])
    local = split_shapes[rank]

    if boundary == "periodic":
        return _take_periodic_range(full, dim, start - left, local + left + right).contiguous()

    if boundary not in ("constant", "replicate"):
        raise ValueError(f"Unsupported halo boundary {boundary!r}")

    take_start = max(0, start - left)
    take_end = min(total, start + local + right)
    result = full.narrow(dim, take_start, take_end - take_start)
    pad_left = take_start - (start - left)
    pad_right = (start + local + right) - take_end
    return _pad_dim(result, dim, pad_left, pad_right, boundary=boundary, fill=fill)


def _halo_tensor(spec, x, halos, *, mesh=None, shapes=None, boundary="constant", fill=0):
    axes = _axes(spec)
    _validate_unique_axis_names(axes)
    axis_names = {axis.name for axis in axes}
    unknown = set(halos) - axis_names
    if unknown:
        raise ValueError(f"Halo axis {sorted(unknown)[0]!r} is not present in the tensor spec")
    z = x
    for axis in axes:
        if axis.name not in halos:
            continue
        left, right = _normalize_halo(halos[axis.name])
        if left == 0 and right == 0:
            continue

        dim = _axis_dim(axes, axis.name)
        if axis.local():
            z = _pad_dim(z, dim, left, right, boundary=boundary, fill=fill)
        else:
            z = _halo_sharded_axis(
                z,
                axis,
                dim,
                left,
                right,
                mesh=mesh,
                shapes=shapes,
                boundary=boundary,
                fill=fill,
            )
    return z


def einhalo(shard, x, halos, *, boundary="constant", fill=0, mesh=None, shapes=None, families=None):
    shard, _ = cached_expand_axis_families(shard, families=families)
    halos = expand_family_mapping(halos, families, label="Halo")
    spec = parse_sharding(f"{shard} -> {shard}")[0]
    return _halo_tensor(spec, x, halos, mesh=mesh, shapes=shapes, boundary=boundary, fill=fill)


def einwindow(
    shard,
    x,
    windows,
    radius,
    *,
    boundary="constant",
    fill=0,
    mesh=None,
    shapes=None,
    families=None,
):
    shard, _ = cached_expand_axis_families(shard, families=families)
    windows = expand_family_mapping(windows, families, label="Window")
    radius = expand_family_mapping(radius, families, label="Radius")
    input_spec, other_spec, output_spec = parse_sharding(shard)
    if other_spec is not None:
        raise ValueError("einwindow expects a unary sharding expression")
    if input_spec.partials or output_spec.partials:
        raise ValueError("einwindow does not support partial tensor specs")

    input_axes = _axes(input_spec)
    output_axes = _axes(output_spec)
    _validate_unique_axis_names(input_axes)
    _validate_unique_axis_names(output_axes)
    input_names = [axis.name for axis in input_axes]
    output_names = [axis.name for axis in output_axes]
    if any(isinstance(axis, EllipsisAxis) for axis in input_axes) or any(isinstance(axis, EllipsisAxis) for axis in output_axes):
        raise NotImplementedError("einwindow does not support ellipsis axes")

    missing = [axis_name for axis_name in windows if axis_name not in input_names]
    if missing:
        raise ValueError(f"Window source axis {missing[0]!r} is not present in the input")
    missing_radius = [axis_name for axis_name in windows if axis_name not in radius]
    if missing_radius:
        raise ValueError(f"Missing radius for window source axis {missing_radius[0]!r}")
    extra_radius = set(radius) - set(windows)
    if extra_radius:
        raise ValueError(f"Radius axis {sorted(extra_radius)[0]!r} has no matching window")

    expected_names = input_names + list(windows.values())
    if len(set(expected_names)) != len(expected_names):
        raise ValueError("Window axes must be distinct from input axes and each other")
    if sorted(expected_names) != sorted(output_names):
        raise ValueError("einwindow output axes must be exactly the input axes plus the requested window axes")

    output_by_name = {axis.name: axis for axis in output_axes}
    for axis in input_axes:
        if output_by_name[axis.name] != axis:
            raise ValueError("einwindow output must preserve input axis sharding")
    for window_axis in windows.values():
        if not output_by_name[window_axis].local():
            raise ValueError("einwindow window axes must be local")

    halos = {axis_name: radius[axis_name] for axis_name in windows}
    z = _halo_tensor(input_spec, x, halos, mesh=mesh, shapes=shapes, boundary=boundary, fill=fill)
    current_names = list(input_names)

    for axis in input_axes:
        if axis.name not in windows:
            continue
        left, right = _normalize_halo(radius[axis.name])
        window_axis = windows[axis.name]
        if window_axis in current_names:
            raise ValueError(f"Window axis {window_axis!r} conflicts with an existing axis")

        dim = current_names.index(axis.name)
        z = z.unfold(dim, left + right + 1, 1)
        current_names.append(window_axis)

    permutation = [current_names.index(name) for name in output_names]
    return z.permute(permutation).contiguous()

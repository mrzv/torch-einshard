import torch

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

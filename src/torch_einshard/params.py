from dataclasses import dataclass, field

import torch.distributed as dist
import torch

from .grammar import parse_sharding
from .helpers import all_reduce, compute_split_shapes_for_factors
from .sharding import AxisGroup, EllipsisAxis


PARAM_SPEC_ATTR = "einshard_spec"


def _tuple(value):
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    result = tuple(value)
    if any(not isinstance(item, str) for item in result):
        raise TypeError("Parameter metadata entries must be strings")
    return result


def _duplicates(values):
    seen = set()
    result = []
    for value in values:
        if value in seen and value not in result:
            result.append(value)
        seen.add(value)
    return result


def _parse_axes(spec):
    input_spec, other, output_spec = parse_sharding(f"{spec} -> {spec}")
    if other is not None or input_spec.partials or output_spec.partials:
        raise ValueError("Parameter specs must contain only axis layout notation")
    return input_spec.axes


@dataclass(frozen=True)
class ParamSpec:
    layout: str
    shared: tuple[str, ...] = ()
    reduce: tuple[str, ...] = ()
    axes: object = field(init=False)

    def __init__(self, layout, *, shared=(), reduce=()):
        axes = _parse_axes(layout)
        shared = _tuple(shared)
        reduce = _tuple(reduce)
        all_shard_dims = axes.all_shard_dims()
        duplicate_shard_dims = _duplicates(all_shard_dims)
        if duplicate_shard_dims:
            dims = ", ".join(sorted(duplicate_shard_dims))
            raise ValueError(f"Parameter specs cannot shard multiple axes over the same mesh dimension: {dims}")
        shard_dims = set(all_shard_dims)
        for name in shared:
            overlap = shard_dims.intersection(name.split("-"))
            if overlap:
                dims = ", ".join(sorted(overlap))
                raise ValueError(f"Shared parameter metadata overlaps with sharded axis dimensions: {dims}")
        object.__setattr__(self, "layout", layout)
        object.__setattr__(self, "axes", axes)
        object.__setattr__(self, "shared", shared)
        object.__setattr__(self, "reduce", reduce)

    def __repr__(self):
        args = [repr(self.layout)]
        if self.shared:
            args.append(f"shared={self.shared!r}")
        if self.reduce:
            args.append(f"reduce={self.reduce!r}")
        return f"ParamSpec({', '.join(args)})"


@dataclass(frozen=True)
class ParamShardMetadata:
    global_shape: tuple[int, ...]
    local_slices: tuple[slice, ...]
    local_shape: tuple[int, ...]
    shard_dims: tuple[str, ...]


def _group(mesh, name):
    try:
        return mesh[name].get_group()
    except (KeyError, RuntimeError) as error:
        if "-" in name and not hasattr(mesh, "device_mesh"):
            raise ValueError(
                f"Compound mesh group {name!r} requires es.wrap_mesh(mesh)"
            ) from error
        raise ValueError(f"Mesh does not contain parameter metadata group {name!r}") from error


def sync_param_(param, spec, mesh):
    for name in spec.shared:
        group = _group(mesh, name)
        if dist.get_world_size(group) == 1:
            continue
        dist.broadcast(param.data, src=dist.get_global_rank(group, 0), group=group)
    return param


def reduce_grad_(param, spec, mesh):
    if param.grad is None:
        return param
    for name in spec.reduce:
        param.grad = all_reduce(param.grad, _group(mesh, name))
    return param


def set_param_spec(param, spec):
    setattr(param, PARAM_SPEC_ATTR, spec)
    return param


def get_param_spec(param):
    return getattr(param, PARAM_SPEC_ATTR, None)


def iter_param_specs(module):
    for name, param in module.named_parameters():
        spec = get_param_spec(param)
        if spec is not None:
            yield name, param, spec


def _require_spec(param_or_spec):
    if isinstance(param_or_spec, ParamSpec):
        return param_or_spec
    spec = get_param_spec(param_or_spec)
    if spec is None:
        raise ValueError("Parameter does not have an attached ParamSpec")
    return spec


def param_shard_dims(param_or_spec):
    spec = _require_spec(param_or_spec)
    return tuple(spec.axes.all_shard_dims())


def _mesh_rank_and_size(mesh, shard_dim):
    if not dist.is_initialized():
        raise RuntimeError("Parameter shard metadata requires an initialized process group")
    device_mesh = getattr(mesh, "device_mesh", mesh)
    if not hasattr(device_mesh, "mesh_dim_names") or not hasattr(device_mesh, "mesh"):
        raise TypeError("mesh must be a PyTorch DeviceMesh or torch_einshard.wrap_mesh(mesh)")
    if shard_dim in device_mesh.mesh_dim_names:
        mesh_dim = device_mesh.mesh_dim_names.index(shard_dim)
        rank = dist.get_rank()
        coord = (device_mesh.mesh == rank).nonzero()[0].tolist()
        return coord[mesh_dim], device_mesh.mesh.shape[mesh_dim]

    if "-" not in shard_dim:
        raise ValueError(f"Mesh does not contain sharded parameter dimension {shard_dim!r}")
    if not hasattr(mesh, "device_mesh"):
        raise ValueError(f"Compound sharded parameter dimension {shard_dim!r} requires es.wrap_mesh(mesh)")

    group = mesh[shard_dim].get_group()
    return dist.get_rank(group), dist.get_world_size(group)


def _axis_shard_dim(axis):
    if isinstance(axis, EllipsisAxis):
        raise ValueError("Parameter shard metadata does not support ellipsis axes")
    if isinstance(axis, AxisGroup):
        shard_dims = axis.axes.all_shard_dims()
        if shard_dims:
            raise NotImplementedError(
                "Parameter shard metadata does not support sharded factored-axis groups"
            )
        return ""
    return axis.shard_dim


def _axis_name(axis):
    if isinstance(axis, EllipsisAxis):
        raise ValueError("Parameter shard metadata does not support ellipsis axes")
    if isinstance(axis, AxisGroup):
        raise NotImplementedError("Parameter shard metadata does not support factored-axis groups")
    return axis.name


def _factor_for(axis, factors):
    if not factors:
        return 1
    name = _axis_name(axis)
    return int(factors.get(name, 1))


def param_local_slices(param_or_spec, global_shape, mesh, factors=None):
    spec = _require_spec(param_or_spec)
    global_shape = tuple(int(size) for size in global_shape)
    if len(global_shape) != len(spec.axes):
        raise ValueError(
            f"Global shape rank {len(global_shape)} does not match ParamSpec rank {len(spec.axes)}"
        )

    slices = []
    for axis, size in zip(spec.axes, global_shape):
        shard_dim = _axis_shard_dim(axis)
        if not shard_dim:
            slices.append(slice(None))
            continue

        rank, chunks = _mesh_rank_and_size(mesh, shard_dim)
        sections = compute_split_shapes_for_factors(size, chunks, _factor_for(axis, factors))
        start = sum(sections[:rank])
        slices.append(slice(start, start + sections[rank]))
    return tuple(slices)


def param_local_shape(param_or_spec, global_shape, mesh, factors=None):
    slices = param_local_slices(param_or_spec, global_shape, mesh, factors=factors)
    return _shape_from_slices(global_shape, slices)


def _shape_from_slices(global_shape, slices):
    shape = []
    for size, local_slice in zip(global_shape, slices):
        start = 0 if local_slice.start is None else local_slice.start
        stop = size if local_slice.stop is None else local_slice.stop
        shape.append(stop - start)
    return tuple(shape)


def param_shard_metadata(param_or_spec, global_shape, mesh, factors=None):
    spec = _require_spec(param_or_spec)
    global_shape = tuple(int(size) for size in global_shape)
    local_slices = param_local_slices(spec, global_shape, mesh, factors=factors)
    return ParamShardMetadata(
        global_shape=global_shape,
        local_slices=local_slices,
        local_shape=_shape_from_slices(global_shape, local_slices),
        shard_dims=param_shard_dims(spec),
    )


def sync_module_params_(module, mesh):
    for param in module.parameters():
        spec = get_param_spec(param)
        if spec is not None:
            sync_param_(param, spec, mesh)
    return module


def reduce_module_grads_(module, mesh):
    for param in module.parameters():
        spec = get_param_spec(param)
        if spec is not None:
            reduce_grad_(param, spec, mesh)
    return module


def register_grad_reduction_hook_(
    ddp_model,
    mesh,
    ddp_group="dp",
    combined_reduce_group=None,
    combined_reduce=None,
):
    combined_reduce = _tuple(combined_reduce)
    if combined_reduce_group is not None and not combined_reduce:
        raise ValueError("combined_reduce must be provided with combined_reduce_group")

    def bucket_views(bucket):
        offset = 0
        views = []
        buffer = bucket.buffer()
        for param in bucket.parameters():
            view = buffer.narrow(0, offset, param.numel()).view_as(param)
            views.append((param, view, get_param_spec(param)))
            offset += param.numel()
        return views

    def can_use_combined_reduce(views):
        if combined_reduce_group is None:
            return False
        if not views:
            return False
        expected = set(combined_reduce)
        for _, _, spec in views:
            if spec is None or set(spec.reduce) != expected:
                return False
        return True

    def reduce_by_param_specs(buffer, views):
        groups = sorted({
            name
            for _, _, spec in views
            for name in (spec.reduce if spec is not None else ())
        })
        for name in groups:
            grad_views = [
                view
                for _, view, spec in views
                if spec is not None and name in spec.reduce
            ]
            if not grad_views:
                continue
            coalesced = torch.cat([view.reshape(-1) for view in grad_views])
            dist.all_reduce(coalesced, group=_group(mesh, name))
            offset = 0
            for view in grad_views:
                view.copy_(coalesced.narrow(0, offset, view.numel()).view_as(view))
                offset += view.numel()
        return buffer

    def reduction_hook(state, bucket):
        buffer = bucket.buffer()
        group = _group(mesh, ddp_group) if ddp_group is not None else None
        group_size = dist.get_world_size(group)
        views = bucket_views(bucket)

        if can_use_combined_reduce(views):
            combined_group = _group(mesh, combined_reduce_group)
            combined_group_size = dist.get_world_size(combined_group)
            if combined_group_size == 1 and group_size == 1:
                future = torch.futures.Future()
                future.set_result(buffer)
            elif combined_group_size > 1:
                future = dist.all_reduce(buffer, group=combined_group, async_op=True).get_future()
            else:
                future = None

            if future is not None:
                def finish_combined(future):
                    if group_size > 1:
                        buffer.div_(group_size)
                    return buffer

                return future.then(finish_combined)

        if group_size == 1:
            future = torch.futures.Future()
            future.set_result(buffer)
        else:
            future = dist.all_reduce(buffer, group=group, async_op=True).get_future()

        def finish(future):
            if group_size > 1:
                buffer.div_(group_size)

            return reduce_by_param_specs(buffer, views)

        return future.then(finish)

    ddp_model.register_comm_hook(None, reduction_hook)
    return ddp_model

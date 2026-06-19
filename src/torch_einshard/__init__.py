from .einsum import einsum
from .conv import einconv
from .fft import einfft
from .grammar import parse_sharding, Axes
from .distributed import distributed_1d
from .families import cached_expand_axis_families
from .halo import einhalo, einwindow
from .mesh import CompoundDeviceMesh, wrap_mesh
from .params import (
    ParamShardMetadata,
    ParamSpec,
    get_param_spec,
    iter_param_specs,
    param_local_shape,
    param_local_slices,
    param_shard_dims,
    param_shard_metadata,
    reduce_grad_,
    reduce_module_grads_,
    register_grad_reduction_hook_,
    set_param_spec,
    sync_module_params_,
    sync_param_,
)
from .roll import einroll
from .sharding import AxisGroup


def _axes(spec):
    return spec.axes if hasattr(spec, "axes") else spec

def _flat_axes(spec):
    axes = _axes(spec)
    return axes.flat() if hasattr(axes, "flat") else axes

def _has_groups(spec):
    return any(isinstance(axis, AxisGroup) for axis in _axes(spec))

def all_local(shard):
    for s in shard:
        if s is None: continue
        if not s.local(): return False
    return True

def local_operation(shard):
    if any(getattr(s, "partials", ()) for s in shard if s is not None):
        return False

    if shard[1] is None:
        # X -> Z
        return Axes(set(_flat_axes(shard[0])) ^ set(_flat_axes(shard[2]))).local()
    else:
        # X,Y -> Z
        input0_axes = _flat_axes(shard[0])
        input1_axes = _flat_axes(shard[1])
        output_axes = _flat_axes(shard[2])
        input_axes = list(input0_axes) + list(input1_axes)

        for output_axis in output_axes:
            if output_axis in input_axes:
                continue
            if any(axis.name == output_axis.name for axis in input_axes):
                return False

        input0_by_name = {axis.name: axis for axis in input0_axes}
        input1_by_name = {axis.name: axis for axis in input1_axes}
        for name in set(input0_by_name) & set(input1_by_name):
            input0_axis = input0_by_name[name]
            input1_axis = input1_by_name[name]
            if not input0_axis.local() or not input1_axis.local():
                return False
            if input0_axis != input1_axis:
                return False

        return True

def einshard(shard, *xs, mesh = None, shapes = None, sizes = None, families = None):
    shard, sizes = cached_expand_axis_families(shard, sizes, families)
    shard = parse_sharding(shard)

    if local_operation(shard):
        return einsum(shard, *xs, sizes=sizes)

    if any(_has_groups(s) for s in shard if s is not None):
        raise NotImplementedError("Factored axes are currently supported only for local reshape operations")

    if mesh is None:
        raise ValueError("Distributed einshard operations require mesh")

    return distributed_1d(shard, *xs, mesh = mesh, shapes = shapes)

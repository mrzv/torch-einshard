from .einsum import einsum
from .grammar import parse_sharding, Axes
from .distributed import distributed_1d
from .families import expand_axis_families
from .mesh import CompoundDeviceMesh, wrap_mesh
from .params import ParamSpec, get_param_spec, reduce_grad_, reduce_module_grads_, register_grad_reduction_hook_, set_param_spec, sync_module_params_, sync_param_
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
        return Axes(set(_flat_axes(shard[0])) & set(_flat_axes(shard[1]))).local()

def einshard(shard, *xs, mesh = None, shapes = None, sizes = None, families = None):
    shard, sizes = expand_axis_families(shard, sizes, families)
    shard = parse_sharding(shard)

    if local_operation(shard):
        return einsum(shard, *xs, sizes=sizes)

    if any(_has_groups(s) for s in shard if s is not None):
        raise NotImplementedError("Factored axes are currently supported only for local reshape operations")

    return distributed_1d(shard, *xs, mesh = mesh, shapes = shapes)

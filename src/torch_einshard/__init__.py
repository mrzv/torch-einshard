from .einsum import einsum
from .grammar import sharding, Axes
from .distributed import distributed_1d
from .roll import einroll


def _axes(spec):
    return spec.axes if hasattr(spec, "axes") else spec

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
        return Axes(set(_axes(shard[0])) ^ set(_axes(shard[2]))).local()
    else:
        # X,Y -> Z
        return Axes(set(_axes(shard[0])) & set(_axes(shard[1]))).local()

def einshard(shard, *xs, mesh = None, shapes = None):
    shard = sharding(shard).map()

    if local_operation(shard):
        return einsum(shard, *xs)

    return distributed_1d(shard, *xs, mesh = mesh, shapes = shapes)

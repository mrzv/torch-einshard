from .einsum import einsum
from .grammar import sharding

def all_local(shard):
    for s in shard:
        if s is None: continue
        for a in s:
            if not a.local():
                return False
    return True

def einshard(shard, *xs):
    shard = sharding(shard).map()

    if all_local(shard):
        return einsum(shard, *xs)

    return NotImplemented

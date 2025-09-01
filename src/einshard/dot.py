import torch

from .grammar import sharding

def contract(shard, x, y):
    shard = sharding(shard).map()

    shard0 = set(shard[0])
    shard1 = set(shard[1])
    shard2 = set(shard[2])
    intersection = shard0 & shard1

    xdims, ydims, zdims = [],[],[]

    for i,n in enumerate(shard[0]):
        if n in intersection:
            assert n not in shard2, "Diagonal is not yet supported"
            xdims.append(i)
        else:
            zdims.append(n)
    for i,n in enumerate(shard[1]):
        if n in intersection:
            assert n not in shard2, "Diagonal is not yet supported"
            ydims.append(i)
        else:
            zdims.append(n)

    # contract
    z = torch.tensordot(x,y, dims = (xdims, ydims))

    # permute
    z = torch.permute(z, tuple([zdims.index(n) for n in shard[2]]))

    return z

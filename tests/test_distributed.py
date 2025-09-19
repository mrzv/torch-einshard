#!/usr/bin/env -S uv run

import pytest

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

import torch_einshard as es

use_cuda = False     # TODO: make a command-line argument
initialized = False

def init():
    global initialized, device, local_rank, world_rank, world_size

    # pytest reinitializes for some reason, so need to check for this
    if initialized:
        return (device,local_rank,world_rank,world_size)
    initialized = True

    local_rank, world_rank, world_size = es.helpers.init_process_group('gloo', use_cuda = use_cuda)

    if use_cuda:
        device = f"cuda:{local_rank}"
        torch.cuda.set_device(local_rank)
        torch.cuda.manual_seed(333)
    else:
        device = "cpu"
    torch.manual_seed(333)

    return (device,local_rank,world_rank,world_size)

def factors(n):
    return set(
        factor for i in range(1, int(n**0.5) + 1) if n % i == 0
        for factor in (i, n//i)
    )

def test_distributed_1d_2():
    device,local_rank,world_rank,world_size = init()

    for x in factors(world_size):
        mesh = init_device_mesh(device, (x, world_size // x), mesh_dim_names=("dp", "sp"))

        # distributed
        x = torch.randn(8,5)
        y = torch.randn(5,10)
        z = es.einshard('a/sp b/dp, b / dp c -> a/sp c', x, y, mesh = mesh)

        zz = torch.einsum('a b, b c -> a c', x, y)
        es.helpers.all_reduce(zz, mesh['dp'].get_group())
        assert torch.norm(z - zz) == 0.
        print(world_rank, x.shape,y.shape,z.shape)

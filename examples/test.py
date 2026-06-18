#!/usr/bin/env -S uv run

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

import torch_einshard as es

from icecream import ic

use_cuda = False     # TODO: make a command-line argument
initialized = False

def init():
    global initialized, device, local_rank, world_rank, world_size

    # pytest reinitializes for some reason, so need to check for this
    if initialized:
        return (device,local_rank,world_rank,world_size)
    initialized = True

    backend = 'nccl' if use_cuda else 'gloo'
    local_rank, world_rank, world_size = es.helpers.init_process_group(backend, use_cuda = use_cuda)

    if use_cuda:
        device = f"cuda:{local_rank}"
        torch.cuda.set_device(local_rank)
        torch.cuda.manual_seed(333)
    else:
        device = "cpu"
    torch.manual_seed(333)

    return (device,local_rank,world_rank,world_size)

device,local_rank,world_rank,world_size = init()

# output only on rank=0
if world_rank != 0:
    ic.disable()

assert world_size == 12, "World size needs to be 12 for mesh shape (2, 3, 2)"

mesh = init_device_mesh(device, (2,3,2), mesh_dim_names=("a", "b", "c"))

print(mesh)
print(mesh.get_all_groups())

print(mesh["b","c"])
print(mesh["b","c"].get_all_groups())

print(mesh["b"])
print(mesh["b"].get_all_groups())

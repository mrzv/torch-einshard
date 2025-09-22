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

assert world_size >= 4 and world_size % 4 == 0, "World size needs to be divisible by 4 (for dp = 4)"

mesh = init_device_mesh(device, (4, world_size // 4), mesh_dim_names=("dp", "sp"))

# distributed
x = torch.randn(8,5, requires_grad = True)
y = torch.randn(5,10, requires_grad = True)
z = es.einshard('a/sp b/dp, b / dp c -> a/sp c', x, y, mesh = mesh)

loss = z.norm()
ic(loss)
loss.backward()


# Gather x and y (across dp)
xx = x.detach().clone().requires_grad_(True)
x_all = es.einshard('a/sp b/dp -> a/sp b', xx,    mesh = mesh)
# x_all = es.einshard('a/sp b    -> a    b', x_all, mesh = mesh)
x_all.retain_grad()

yy = y.detach().clone().requires_grad_(True)
y_all = es.einshard('b/dp c -> b c', yy, mesh = mesh)

# Multiply the matrix locally
z_all = es.einshard('a/sp b, b c -> a/sp c', x_all, y_all)
loss_all = z_all.norm()
ic(loss_all)
loss_all.backward()
ic((x.grad - xx.grad).norm())

# Split the grads and compare
x_all_grad = x_all.grad.detach().clone()
ic(x_all_grad.shape)
x_all_grad = es.einshard('a/sp b    -> a/sp    b/dp', x_all_grad, mesh = mesh)
ic(x_all_grad.shape)

ic(x_all.shape, x_all_grad.shape)
ic((x.grad - x_all_grad).norm())

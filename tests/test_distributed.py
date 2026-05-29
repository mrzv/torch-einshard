#!/usr/bin/env -S uv run

import pytest
from icecream import ic

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

def factors(n):
    return set(
        factor for i in range(1, int(n**0.5) + 1) if n % i == 0
        for factor in (i, n//i)
    )

def test_distributed_1d_2():
    device,local_rank,world_rank,world_size = init()
    # output only on rank=0
    if world_rank != 0:
        ic.disable()

    for x in factors(world_size):
        mesh = init_device_mesh(device, (x, world_size // x), mesh_dim_names=("dp", "sp"))

        # distributed
        x = torch.randn(8,5, requires_grad = True)
        y = torch.randn(5,10, requires_grad = True)
        z = es.einshard('a/sp b/dp, b / dp c -> a/sp c', x, y, mesh = mesh)

        loss = z.norm()
        ic(loss)
        loss.backward()

        # compute locally and manually all_reduce
        zz = es.einshard('al bl, bl c -> al c', x, y)
        es.helpers.all_reduce(zz, mesh['dp'].get_group())
        ic(torch.norm(z - zz))
        assert torch.allclose(z, zz)
        print(world_rank, x.shape,y.shape,z.shape)

        # Gather x and y (across dp)
        xx = x.detach().clone().requires_grad_(True)
        x_all = es.einshard('a/sp b/dp -> a/sp b', xx,    mesh = mesh)
        x_all.retain_grad()

        yy = y.detach().clone().requires_grad_(True)
        y_all = es.einshard('b/dp c -> b c', yy, mesh = mesh)

        # Multiply the matrix locally
        z_all = es.einshard('a/sp b, b c -> a/sp c', x_all, y_all)
        loss_all = z_all.norm()
        ic(torch.norm(z - z_all))
        assert torch.allclose(z, z_all)
        loss_all.backward()
        ic((x.grad - xx.grad).norm())
        assert torch.allclose(x.grad, xx.grad)

        # Split the grads and compare
        x_all_grad = x_all.grad.detach().clone()
        ic(x_all_grad.shape)
        x_all_grad = es.einshard('a/sp b    -> a/sp    b/dp', x_all_grad, mesh = mesh)
        ic(x_all_grad.shape)

        ic(x_all.shape, x_all_grad.shape)
        ic((x.grad - x_all_grad).norm())
        assert torch.allclose(x.grad, x_all_grad)

def test_distributed_1d_1_multi_axis_split_gather():
    device,local_rank,world_rank,world_size = init()
    # output only on rank=0
    if world_rank != 0:
        ic.disable()

    for mesh_rows in factors(world_size):
        mesh = init_device_mesh(device, (mesh_rows, world_size // mesh_rows), mesh_dim_names=("dp", "sp"))

        x = torch.randn(16,24, requires_grad = True)
        shapes = {
            "sp": es.helpers.compute_split_shapes(x.shape[0], dist.get_world_size(mesh["sp"].get_group())),
            "dp": es.helpers.compute_split_shapes(x.shape[1], dist.get_world_size(mesh["dp"].get_group())),
        }

        z = es.einshard('a b -> a/sp b/dp', x, mesh = mesh, shapes = shapes)
        assert z.shape == (
            shapes["sp"][dist.get_rank(mesh["sp"].get_group())],
            shapes["dp"][dist.get_rank(mesh["dp"].get_group())],
        )

        zz = es.einshard('a/sp b/dp -> a b', z, mesh = mesh, shapes = shapes)
        assert torch.allclose(x, zz)

        loss = (z ** 2).sum()
        loss.backward()
        assert torch.allclose(x.grad, 2 * x)

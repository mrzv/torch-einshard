import torch
import torch.distributed as dist

import torch_einshard as es
from torch_einshard.mappings import (
    allgather_forward_split_backward,
    allreduce_forward_identity_backward,
    identity_forward_allreduce_backward,
    split_forward_allgather_backward,
)

from conftest import assert_close


def test_allreduce_forward_identity_backward(dist_env, mesh_1d):
    group = mesh_1d["dp"].get_group()
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)

    x = torch.full((2, 3), float(rank + 1), requires_grad=True)
    z = allreduce_forward_identity_backward(x, group)

    expected = torch.full_like(x, float(world_size * (world_size + 1) // 2))
    assert_close(z, expected)

    z.sum().backward()
    assert_close(x.grad, torch.ones_like(x))


def test_identity_forward_allreduce_backward(dist_env, mesh_1d):
    group = mesh_1d["dp"].get_group()
    world_size = dist.get_world_size(group)

    x = torch.randn(4, 5, requires_grad=True)
    z = identity_forward_allreduce_backward(x, group)

    assert_close(z, x)

    z.sum().backward()
    assert_close(x.grad, torch.ones_like(x) * world_size)


def test_split_forward_allgather_backward(dist_env, mesh_1d):
    group = mesh_1d["dp"].get_group()
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)
    rows = world_size * 2
    shapes = es.helpers.compute_split_shapes(rows, world_size)

    x = torch.randn(rows, 3, requires_grad=True)
    z = split_forward_allgather_backward(x, group, 0, shapes)

    expected = torch.split(x.detach(), shapes, dim=0)[rank]
    assert_close(z, expected)

    z.sum().backward()
    assert_close(x.grad, torch.ones_like(x))


def test_allgather_forward_split_backward(dist_env, mesh_1d):
    group = mesh_1d["dp"].get_group()
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)
    rows = world_size * 2
    shapes = es.helpers.compute_split_shapes(rows, world_size)

    full = torch.arange(rows * 3, dtype=torch.float32).reshape(rows, 3)
    x = torch.split(full, shapes, dim=0)[rank].clone().requires_grad_(True)
    z = allgather_forward_split_backward(x, group, 0, shapes)

    assert_close(z, full)

    z.sum().backward()
    assert_close(x.grad, torch.ones_like(x))

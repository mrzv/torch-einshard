import torch
import torch.distributed as dist

import torch_einshard as es

from conftest import assert_close


def test_single_axis_distributed_roll(dist_env, mesh_1d):
    group = mesh_1d["dp"].get_group()
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)
    rows = world_size * 3
    shapes = es.helpers.compute_split_shapes(rows, world_size)

    full = torch.randn(rows, 5)
    x = torch.split(full, shapes, dim=0)[rank].clone().requires_grad_(True)

    z = es.einroll("a/dp b", x, {"a": 2}, mesh=mesh_1d, shapes=shapes)

    expected = torch.split(torch.roll(full, shifts=2, dims=0), shapes, dim=0)[rank]
    assert_close(z, expected)

    (z ** 2).sum().backward()
    assert_close(x.grad, 2 * x)


def test_multi_axis_distributed_roll(dist_env, mesh_2d):
    sp_group = mesh_2d["sp"].get_group()
    dp_group = mesh_2d["dp"].get_group()
    sp_rank = dist.get_rank(sp_group)
    dp_rank = dist.get_rank(dp_group)
    shapes = {
        "sp": es.helpers.compute_split_shapes(12, dist.get_world_size(sp_group)),
        "dp": es.helpers.compute_split_shapes(16, dist.get_world_size(dp_group)),
    }

    full = torch.randn(12, 16)
    x = torch.split(full, shapes["sp"], dim=0)[sp_rank]
    x = torch.split(x, shapes["dp"], dim=1)[dp_rank].contiguous().requires_grad_(True)

    z = es.einroll("a/sp b/dp", x, {"a": -2, "b": 3}, mesh=mesh_2d, shapes=shapes)

    expected = torch.roll(full, shifts=(-2, 3), dims=(0, 1))
    expected = torch.split(expected, shapes["sp"], dim=0)[sp_rank]
    expected = torch.split(expected, shapes["dp"], dim=1)[dp_rank]
    assert_close(z, expected)

import pytest
import torch
import torch.distributed as dist

import torch_einshard as es

from conftest import assert_close


def test_local_roll_axis_family():
    x = torch.randn(2, 3, 4, 5, 6, 7)

    z = es.einroll(
        "b t *spatial c",
        x,
        {"spatial": (1, -2, 3)},
        families={"spatial": ("h", "w", "d")},
    )

    assert_close(z, torch.roll(x, shifts=(1, -2, 3), dims=(2, 3, 4)))


def test_distributed_einroll_requires_mesh():
    x = torch.randn(4, 3)

    with pytest.raises(ValueError, match="Distributed einroll operations require mesh"):
        es.einroll("a/dp b", x, {"a": 1})


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


def test_single_axis_distributed_roll_negative_shift(dist_env, mesh_1d):
    group = mesh_1d["dp"].get_group()
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)
    rows = world_size * 3
    shapes = es.helpers.compute_split_shapes(rows, world_size)

    full = torch.randn(rows, 5)
    x = torch.split(full, shapes, dim=0)[rank].clone().requires_grad_(True)

    z = es.einroll("a/dp b", x, {"a": -2}, mesh=mesh_1d, shapes=shapes)

    expected = torch.split(torch.roll(full, shifts=-2, dims=0), shapes, dim=0)[rank]
    assert_close(z, expected)

    z.sum().backward()
    assert_close(x.grad, torch.ones_like(x))


def test_single_axis_distributed_roll_shard_aligned(dist_env, mesh_1d):
    group = mesh_1d["dp"].get_group()
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)
    rows = world_size * 3
    shapes = es.helpers.compute_split_shapes(rows, world_size)

    full = torch.randn(rows, 5)
    x = torch.split(full, shapes, dim=0)[rank].clone().requires_grad_(True)

    z = es.einroll("a/dp b", x, {"a": shapes[rank]}, mesh=mesh_1d, shapes=shapes)

    expected = torch.split(torch.roll(full, shifts=shapes[rank], dims=0), shapes, dim=0)[rank]
    assert_close(z, expected)

    z.sum().backward()
    assert_close(x.grad, torch.ones_like(x))


def test_single_axis_distributed_roll_uneven_shards(dist_env, mesh_1d):
    group = mesh_1d["dp"].get_group()
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)
    rows = world_size * 3 + 1
    shapes = es.helpers.compute_split_shapes(rows, world_size)

    full = torch.randn(rows, 5)
    x = torch.split(full, shapes, dim=0)[rank].clone().requires_grad_(True)

    z = es.einroll("a/dp b", x, {"a": -3}, mesh=mesh_1d, shapes=shapes)

    expected = torch.split(torch.roll(full, shifts=-3, dims=0), shapes, dim=0)[rank]
    assert_close(z, expected)

    z.sum().backward()
    assert_close(x.grad, torch.ones_like(x))


def test_single_axis_distributed_roll_without_shapes_warns(dist_env, mesh_1d):
    group = mesh_1d["dp"].get_group()
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)
    rows = world_size * 3
    shapes = es.helpers.compute_split_shapes(rows, world_size)

    full = torch.randn(rows, 5)
    x = torch.split(full, shapes, dim=0)[rank].clone().requires_grad_(True)

    with pytest.warns(RuntimeWarning, match="gather/roll/split fallback"):
        z = es.einroll("a/dp b", x, {"a": 2}, mesh=mesh_1d)

    expected = torch.split(torch.roll(full, shifts=2, dims=0), shapes, dim=0)[rank]
    assert_close(z, expected)


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


def test_multi_axis_distributed_roll_axis_family(dist_env, mesh_2d):
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

    z = es.einroll(
        "*spatial",
        x,
        {"spatial": (-2, 3)},
        mesh=mesh_2d,
        shapes=shapes,
        families={"spatial": ("a/sp", "b/dp")},
    )

    expected = torch.roll(full, shifts=(-2, 3), dims=(0, 1))
    expected = torch.split(expected, shapes["sp"], dim=0)[sp_rank]
    expected = torch.split(expected, shapes["dp"], dim=1)[dp_rank]
    assert_close(z, expected)


def test_multi_axis_distributed_roll_uneven_shards(dist_env, mesh_2d):
    sp_group = mesh_2d["sp"].get_group()
    dp_group = mesh_2d["dp"].get_group()
    sp_rank = dist.get_rank(sp_group)
    dp_rank = dist.get_rank(dp_group)
    shapes = {
        "sp": es.helpers.compute_split_shapes(13, dist.get_world_size(sp_group)),
        "dp": es.helpers.compute_split_shapes(17, dist.get_world_size(dp_group)),
    }

    full = torch.randn(13, 17)
    x = torch.split(full, shapes["sp"], dim=0)[sp_rank]
    x = torch.split(x, shapes["dp"], dim=1)[dp_rank].contiguous().requires_grad_(True)

    z = es.einroll("a/sp b/dp", x, {"a": 5, "b": -4}, mesh=mesh_2d, shapes=shapes)

    expected = torch.roll(full, shifts=(5, -4), dims=(0, 1))
    expected = torch.split(expected, shapes["sp"], dim=0)[sp_rank]
    expected = torch.split(expected, shapes["dp"], dim=1)[dp_rank]
    assert_close(z, expected)

    z.sum().backward()
    assert_close(x.grad, torch.ones_like(x))

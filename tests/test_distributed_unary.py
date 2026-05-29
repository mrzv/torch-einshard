import torch
import torch.distributed as dist

import torch_einshard as es

from conftest import assert_close


def test_single_axis_split_gather_round_trip(dist_env, mesh_1d):
    group = mesh_1d["dp"].get_group()
    rank = dist.get_rank(group)
    rows = dist.get_world_size(group) * 2
    shapes = es.helpers.compute_split_shapes(rows, dist.get_world_size(group))

    x = torch.randn(rows, 3, requires_grad=True)
    z = es.einshard("a b -> a/dp b", x, mesh=mesh_1d, shapes=shapes)

    assert z.shape == (shapes[rank], 3)

    zz = es.einshard("a/dp b -> a b", z, mesh=mesh_1d, shapes=shapes)
    assert_close(zz, x)

    (z ** 2).sum().backward()
    assert_close(x.grad, 2 * x)


def test_multi_axis_split_gather_round_trip(dist_env, mesh_2d):
    sp_group = mesh_2d["sp"].get_group()
    dp_group = mesh_2d["dp"].get_group()
    shapes = {
        "sp": es.helpers.compute_split_shapes(16, dist.get_world_size(sp_group)),
        "dp": es.helpers.compute_split_shapes(24, dist.get_world_size(dp_group)),
    }

    x = torch.randn(16, 24, requires_grad=True)
    z = es.einshard("a b -> a/sp b/dp", x, mesh=mesh_2d, shapes=shapes)

    assert z.shape == (
        shapes["sp"][dist.get_rank(sp_group)],
        shapes["dp"][dist.get_rank(dp_group)],
    )

    zz = es.einshard("a/sp b/dp -> a b", z, mesh=mesh_2d, shapes=shapes)
    assert_close(zz, x)

    (z ** 2).sum().backward()
    assert_close(x.grad, 2 * x)


def test_split_with_output_permutation(dist_env, mesh_1d):
    group = mesh_1d["dp"].get_group()
    rank = dist.get_rank(group)
    rows = dist.get_world_size(group) * 2
    shapes = es.helpers.compute_split_shapes(rows, dist.get_world_size(group))

    x = torch.randn(3, rows, requires_grad=True)
    z = es.einshard("b a -> a/dp b", x, mesh=mesh_1d, shapes=shapes)

    expected = torch.split(x.detach(), shapes, dim=1)[rank].transpose(0, 1).contiguous()
    assert_close(z, expected)


def test_repartition_between_axes(dist_env, mesh_1d):
    group = mesh_1d["dp"].get_group()
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)
    rows = world_size * 2
    cols = world_size * 3
    row_shapes = es.helpers.compute_split_shapes(rows, world_size)
    col_shapes = es.helpers.compute_split_shapes(cols, world_size)

    full = torch.randn(rows, cols, requires_grad=True)
    x = torch.split(full.detach(), row_shapes, dim=0)[rank].clone().requires_grad_(True)

    z = es.einshard(
        "a/dp b -> a b/dp",
        x,
        mesh=mesh_1d,
        shapes={"dp": {"a": row_shapes, "b": col_shapes}},
    )

    expected = torch.split(full.detach(), col_shapes, dim=1)[rank]
    assert_close(z, expected)


def test_repartition_between_mesh_dimensions(dist_env, mesh_2d):
    dp_group = mesh_2d["dp"].get_group()
    sp_group = mesh_2d["sp"].get_group()
    dp_rank = dist.get_rank(dp_group)
    sp_rank = dist.get_rank(sp_group)
    dp_size = dist.get_world_size(dp_group)
    sp_size = dist.get_world_size(sp_group)
    rows = dp_size * sp_size * 2
    dp_shapes = es.helpers.compute_split_shapes(rows, dp_size)
    sp_shapes = es.helpers.compute_split_shapes(rows, sp_size)

    full = torch.randn(rows, 3, requires_grad=True)
    x = torch.split(full.detach(), dp_shapes, dim=0)[dp_rank].clone().requires_grad_(True)

    z = es.einshard(
        "a/dp b -> a/sp b",
        x,
        mesh=mesh_2d,
        shapes={"dp": {"a": dp_shapes}, "sp": {"a": sp_shapes}},
    )

    expected = torch.split(full.detach(), sp_shapes, dim=0)[sp_rank]
    assert_close(z, expected)


def test_partial_to_full(dist_env, mesh_tp):
    group = mesh_tp["tp"].get_group()
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)

    x = torch.full((2, 3), float(rank + 1), requires_grad=True)
    z = es.einshard("a b // tp -> a b", x, mesh=mesh_tp)

    expected = torch.full_like(x, float(world_size * (world_size + 1) // 2))
    assert_close(z, expected)

    z.sum().backward()
    assert_close(x.grad, torch.ones_like(x))


def test_partial_to_shard(dist_env, mesh_tp):
    group = mesh_tp["tp"].get_group()
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)
    rows = world_size * 2
    shapes = es.helpers.compute_split_shapes(rows, world_size)

    x = torch.full((rows, 3), float(rank + 1), requires_grad=True)
    z = es.einshard("a b // tp -> a/tp b", x, mesh=mesh_tp, shapes=shapes)

    reduced = torch.full_like(x, float(world_size * (world_size + 1) // 2))
    expected = torch.split(reduced, shapes, dim=0)[rank]
    assert_close(z, expected)

    z.sum().backward()
    assert_close(x.grad, torch.ones_like(x))


def test_shard_to_partial(dist_env, mesh_tp):
    group = mesh_tp["tp"].get_group()
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)
    rows = world_size * 2
    shapes = es.helpers.compute_split_shapes(rows, world_size)

    full = torch.arange(rows * 3, dtype=torch.float32).reshape(rows, 3)
    x = torch.split(full, shapes, dim=0)[rank].clone().requires_grad_(True)
    z = es.einshard("a/tp b -> a b // tp", x, mesh=mesh_tp, shapes=shapes)

    assert_close(z, full)

    z.sum().backward()
    assert_close(x.grad, torch.ones_like(x) * world_size)


def test_scalar_multiple_partials_to_full(dist_env, mesh_2d):
    world_rank = dist.get_rank()
    world_size = dist.get_world_size()

    x = torch.tensor(float(world_rank + 1), requires_grad=True)
    z = es.einshard("loss // (sp,dp) -> loss", x, mesh=mesh_2d)

    expected = torch.tensor(float(world_size * (world_size + 1) // 2))
    assert_close(z, expected)

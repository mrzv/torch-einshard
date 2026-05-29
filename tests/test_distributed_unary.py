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

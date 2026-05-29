import torch

import torch_einshard as es

from conftest import assert_close


def test_sharded_contraction_matches_manual_allreduce(dist_env, mesh_2d):
    x = torch.randn(8, 5, requires_grad=True)
    y = torch.randn(5, 10, requires_grad=True)

    z = es.einshard("a/sp b/dp, b/dp c -> a/sp c", x, y, mesh=mesh_2d)

    expected = es.einshard("al bl, bl c -> al c", x, y)
    es.helpers.all_reduce(expected, mesh_2d["dp"].get_group())
    assert_close(z, expected)


def test_sharded_contraction_matches_gathered_reference(dist_env, mesh_2d):
    x = torch.randn(8, 5, requires_grad=True)
    y = torch.randn(5, 10, requires_grad=True)

    z = es.einshard("a/sp b/dp, b/dp c -> a/sp c", x, y, mesh=mesh_2d)
    loss = z.norm()
    loss.backward()

    xx = x.detach().clone().requires_grad_(True)
    x_all = es.einshard("a/sp b/dp -> a/sp b", xx, mesh=mesh_2d)
    x_all.retain_grad()

    yy = y.detach().clone().requires_grad_(True)
    y_all = es.einshard("b/dp c -> b c", yy, mesh=mesh_2d)

    z_all = es.einshard("a/sp b, b c -> a/sp c", x_all, y_all)
    assert_close(z_all, z)

    z_all.norm().backward()
    assert_close(xx.grad, x.grad)

    split_grad = es.einshard("a/sp b -> a/sp b/dp", x_all.grad.detach().clone(), mesh=mesh_2d)
    assert_close(split_grad, x.grad)

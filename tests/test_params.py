import torch
import torch.distributed as dist

import torch_einshard as es

from conftest import assert_close


def test_param_spec_parses_layout_and_metadata():
    spec = es.ParamSpec("o/tp c", shared="sp1-sp2", reduce=("sp1-sp2",))

    assert spec.layout == "o/tp c"
    assert spec.axes[0].name == "o"
    assert spec.axes[0].shard_dim == "tp"
    assert spec.axes[1].name == "c"
    assert spec.shared == ("sp1-sp2",)
    assert spec.reduce == ("sp1-sp2",)


def test_param_spec_rejects_shared_sharded_axis_overlap():
    try:
        es.ParamSpec("o/tp c", shared="tp")
    except ValueError as error:
        assert "overlaps" in str(error)
    else:
        raise AssertionError("Expected overlapping shared and sharded metadata to fail")


def test_sync_param_broadcasts_shared_values(dist_env, mesh_2d):
    mesh = es.wrap_mesh(mesh_2d)
    spec = es.ParamSpec("o c", shared="dp-sp")
    param = torch.nn.Parameter(torch.full((2, 3), float(dist.get_rank() + 1)))

    es.sync_param_(param, spec, mesh)

    assert_close(param, torch.ones_like(param))


def test_reduce_grad_allreduces_reduce_groups(dist_env, mesh_2d):
    mesh = es.wrap_mesh(mesh_2d)
    spec = es.ParamSpec("o c", reduce="dp-sp")
    param = torch.nn.Parameter(torch.zeros(2, 3))
    param.grad = torch.full_like(param, float(dist.get_rank() + 1))
    world_size = dist.get_world_size()

    es.reduce_grad_(param, spec, mesh)

    expected = float(world_size * (world_size + 1) // 2)
    assert_close(param.grad, torch.full_like(param, expected))


def test_reduce_grad_allows_missing_grad(dist_env, mesh_2d):
    spec = es.ParamSpec("o c", reduce="dp")
    param = torch.nn.Parameter(torch.zeros(2, 3))

    assert es.reduce_grad_(param, spec, mesh_2d) is param
    assert param.grad is None

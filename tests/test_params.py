import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel

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


def test_param_shard_dims_reads_specs_from_params():
    param = torch.nn.Parameter(torch.zeros(2, 3))
    spec = es.ParamSpec("o/tp c/sp")
    es.set_param_spec(param, spec)

    assert es.param_shard_dims(spec) == ("tp", "sp")
    assert es.param_shard_dims(param) == ("tp", "sp")


def test_param_shard_dims_requires_attached_spec():
    param = torch.nn.Parameter(torch.zeros(2, 3))

    try:
        es.param_shard_dims(param)
    except ValueError as error:
        assert "ParamSpec" in str(error)
    else:
        raise AssertionError("Expected missing ParamSpec to fail")


def test_param_local_slices_uses_mesh_coordinates(dist_env, mesh_2d):
    spec = es.ParamSpec("o/dp c/sp")
    global_shape = (5, 7)
    coord = (mesh_2d.mesh == dist.get_rank()).nonzero()[0].tolist()
    dp_sections = es.helpers.compute_split_shapes_for_factors(
        global_shape[0], mesh_2d.mesh.shape[0], 1
    )
    sp_sections = es.helpers.compute_split_shapes_for_factors(
        global_shape[1], mesh_2d.mesh.shape[1], 1
    )
    expected = (
        slice(sum(dp_sections[:coord[0]]), sum(dp_sections[:coord[0] + 1])),
        slice(sum(sp_sections[:coord[1]]), sum(sp_sections[:coord[1] + 1])),
    )

    assert es.param_local_slices(spec, global_shape, mesh_2d) == expected
    assert es.param_local_shape(spec, global_shape, mesh_2d) == (
        dp_sections[coord[0]],
        sp_sections[coord[1]],
    )


def test_param_shard_metadata_supports_compound_groups(dist_env, mesh_2d):
    mesh = es.wrap_mesh(mesh_2d)
    spec = es.ParamSpec("o/dp-sp c")
    global_shape = (dist_env.world_size + 3, 2)
    group = mesh["dp-sp"].get_group()
    rank = dist.get_rank(group)
    sections = es.helpers.compute_split_shapes_for_factors(
        global_shape[0], dist.get_world_size(group), 1
    )

    metadata = es.param_shard_metadata(spec, global_shape, mesh)

    assert metadata.global_shape == global_shape
    assert metadata.local_slices == (
        slice(sum(sections[:rank]), sum(sections[:rank + 1])),
        slice(None),
    )
    assert metadata.local_shape == (sections[rank], 2)
    assert metadata.shard_dims == ("dp-sp",)


def test_param_local_slices_rejects_sharded_factored_axes(dist_env, mesh_2d):
    spec = es.ParamSpec("(a/dp b) c")

    try:
        es.param_local_slices(spec, (6, 2), mesh_2d)
    except NotImplementedError as error:
        assert "factored-axis" in str(error)
    else:
        raise AssertionError("Expected sharded factored-axis metadata to fail")


def test_sync_param_broadcasts_shared_values(dist_env, mesh_2d):
    mesh = es.wrap_mesh(mesh_2d)
    spec = es.ParamSpec("o c", shared="dp-sp")
    param = torch.nn.Parameter(torch.full((2, 3), float(dist.get_rank() + 1)))

    es.sync_param_(param, spec, mesh)

    assert_close(param, torch.ones_like(param))


def test_module_param_helpers_use_attached_specs(dist_env, mesh_2d):
    mesh = es.wrap_mesh(mesh_2d)
    module = nn.Linear(3, 2, bias=False)
    es.set_param_spec(module.weight, es.ParamSpec("o c", shared="dp-sp", reduce="dp-sp"))
    module.weight.data.fill_(float(dist.get_rank() + 1))

    assert es.get_param_spec(module.weight).shared == ("dp-sp",)
    assert es.sync_module_params_(module, mesh) is module
    assert_close(module.weight, torch.ones_like(module.weight))

    module.weight.grad = torch.full_like(module.weight, float(dist.get_rank() + 1))
    es.reduce_module_grads_(module, mesh)
    expected = float(dist.get_world_size() * (dist.get_world_size() + 1) // 2)
    assert_close(module.weight.grad, torch.full_like(module.weight, expected))


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


def test_ddp_grad_reduction_hook_uses_param_specs(dist_env, mesh_2d):
    model = nn.Linear(1, 1, bias=False)
    model.weight.data.fill_(1.0)
    es.set_param_spec(model.weight, es.ParamSpec("o c", reduce="sp"))
    ddp = DistributedDataParallel(model, process_group=mesh_2d["dp"].get_group())
    es.register_grad_reduction_hook_(ddp, mesh_2d, ddp_group="dp")

    x = torch.tensor([[float(dist.get_rank() + 1)]])
    ddp(x).sum().backward()

    dp_size = dist.get_world_size(mesh_2d["dp"].get_group())
    sp_size = dist.get_world_size(mesh_2d["sp"].get_group())
    sp_rank = dist.get_rank(mesh_2d["sp"].get_group())
    expected = 0.0
    for peer_sp_rank in range(sp_size):
        expected += 1.0 + peer_sp_rank + sp_size * (dp_size - 1) / 2

    assert sp_rank < sp_size
    assert_close(model.weight.grad, torch.full_like(model.weight.grad, expected))

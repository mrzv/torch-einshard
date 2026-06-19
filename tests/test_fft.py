import importlib
import warnings

import pytest
import torch
import torch.distributed as dist

import torch_einshard as es

from conftest import assert_close


def _assert_no_runtime_warning(fn):
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        result = fn()
    runtime_warnings = [warning for warning in record if issubclass(warning.category, RuntimeWarning)]
    assert runtime_warnings == []
    return result


def test_local_fft_renames_axis():
    x = torch.randn(2, 5, 3, dtype=torch.complex64)

    z = es.einfft("b x c -> b k c", x, axes={"x": "k"})

    assert_close(z, torch.fft.fftn(x, dim=(1,)))


def test_local_fftn_permute_output():
    x = torch.randn(2, 5, 7, dtype=torch.complex64)

    z = es.einfft("b x y -> ky b kx", x, axes={"x": "kx", "y": "ky"})

    expected = torch.fft.fftn(x, dim=(1, 2)).permute(2, 0, 1)
    assert_close(z, expected)


def test_local_fft_expands_iterable_axis_family():
    x = torch.randn(2, 5, 7, dtype=torch.complex64)

    z = es.einfft(
        "b *spatial -> b *spatial",
        x,
        axes=["spatial"],
        families={"spatial": ("x", "y")},
    )

    assert_close(z, torch.fft.fftn(x, dim=(1, 2)))


def test_inverse_fft():
    x = torch.randn(2, 5, dtype=torch.complex64)

    z = es.einfft("b k -> b x", x, axes={"k": "x"}, inverse=True, norm="ortho")

    assert_close(z, torch.fft.ifftn(x, dim=(1,), norm="ortho"))


def test_rejects_partial_specs():
    x = torch.randn(2, 5, dtype=torch.complex64)

    with pytest.raises(ValueError, match="partial"):
        es.einfft("b x // tp -> b k", x, axes={"x": "k"})


def test_rejects_string_axes():
    x = torch.randn(2, 5, dtype=torch.complex64)

    with pytest.raises(TypeError, match="not a string"):
        es.einfft("b time -> b freq", x, axes="time")


def test_sharded_transform_axis_uses_distributed_fft(dist_env, mesh_tp):
    group = mesh_tp["tp"].get_group()
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)
    size = world_size * world_size * 3
    shapes = es.helpers.compute_split_shapes(size, world_size)

    full = torch.randn(2, size, dtype=torch.complex64)
    x = torch.split(full, shapes, dim=1)[rank].contiguous()

    z = _assert_no_runtime_warning(
        lambda: es.einfft(
            "b x/tp -> b k/tp",
            x,
            axes={"x": "k"},
            mesh=mesh_tp,
            shapes={"tp": {"x": shapes, "k": shapes}},
        )
    )

    expected = torch.split(torch.fft.fftn(full, dim=(1,)), shapes, dim=1)[rank]
    assert_close(z, expected)


@pytest.mark.parametrize("inverse", [False, True])
@pytest.mark.parametrize("norm", [None, "forward", "backward", "ortho"])
def test_sharded_transform_axis_fast_path_norms(dist_env, mesh_tp, inverse, norm):
    group = mesh_tp["tp"].get_group()
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)
    size = world_size * world_size * 3
    shapes = es.helpers.compute_split_shapes(size, world_size)

    full = torch.randn(2, size, dtype=torch.complex64)
    x = torch.split(full, shapes, dim=1)[rank].contiguous()

    z = _assert_no_runtime_warning(
        lambda: es.einfft(
            "b x/tp -> b k/tp",
            x,
            axes={"x": "k"},
            inverse=inverse,
            norm=norm,
            mesh=mesh_tp,
            shapes={"tp": {"x": shapes, "k": shapes}},
        )
    )

    fft = torch.fft.ifftn if inverse else torch.fft.fftn
    expected = torch.split(fft(full, dim=(1,), norm=norm), shapes, dim=1)[rank]
    assert_close(z, expected)


def test_sharded_transform_axis_fast_path_permute_output(dist_env, mesh_tp):
    group = mesh_tp["tp"].get_group()
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)
    size = world_size * world_size * 3
    shapes = es.helpers.compute_split_shapes(size, world_size)

    full = torch.randn(2, size, dtype=torch.complex64)
    x = torch.split(full, shapes, dim=1)[rank].contiguous()

    z = _assert_no_runtime_warning(
        lambda: es.einfft(
            "b x/tp -> k/tp b",
            x,
            axes={"x": "k"},
            mesh=mesh_tp,
            shapes={"tp": {"x": shapes, "k": shapes}},
        )
    )

    expected = torch.split(torch.fft.fftn(full, dim=(1,)).mT, shapes, dim=0)[rank]
    assert_close(z, expected)


@pytest.mark.parametrize("inverse,norm", [(False, None), (True, "ortho")])
def test_sharded_and_local_transform_axes_fast_path(dist_env, mesh_tp, inverse, norm):
    group = mesh_tp["tp"].get_group()
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)
    x_size = world_size * world_size * 3
    y_size = 5
    shapes = es.helpers.compute_split_shapes(x_size, world_size)

    full = torch.randn(2, x_size, y_size, dtype=torch.complex64)
    x = torch.split(full, shapes, dim=1)[rank].contiguous()

    z = _assert_no_runtime_warning(
        lambda: es.einfft(
            "b x/tp y -> b kx/tp ky",
            x,
            axes={"x": "kx", "y": "ky"},
            inverse=inverse,
            norm=norm,
            mesh=mesh_tp,
            shapes={"tp": {"x": shapes, "kx": shapes}},
        )
    )

    fft = torch.fft.ifftn if inverse else torch.fft.fftn
    expected = torch.split(fft(full, dim=(1, 2), norm=norm), shapes, dim=1)[rank]
    assert_close(z, expected)


def test_sharded_and_local_transform_axes_fast_path_backward(dist_env, mesh_tp, monkeypatch):
    fft_impl = importlib.import_module("torch_einshard.fft")
    distributed_fft_calls = 0
    original_distributed_fft = fft_impl._distributed_fft_1d_no_autograd

    def counted_distributed_fft(*args, **kwargs):
        nonlocal distributed_fft_calls
        distributed_fft_calls += 1
        return original_distributed_fft(*args, **kwargs)

    monkeypatch.setattr(fft_impl, "_distributed_fft_1d_no_autograd", counted_distributed_fft)

    group = mesh_tp["tp"].get_group()
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)
    x_size = world_size * world_size * 3
    y_size = 5
    shapes = es.helpers.compute_split_shapes(x_size, world_size)

    full = torch.randn(2, x_size, y_size, dtype=torch.complex64)
    x = torch.split(full.detach().clone(), shapes, dim=1)[rank].contiguous().requires_grad_()
    z = es.einfft(
        "b x/tp y -> b kx/tp ky",
        x,
        axes={"x": "kx", "y": "ky"},
        mesh=mesh_tp,
        shapes={"tp": {"x": shapes, "kx": shapes}},
    )
    (z.abs() ** 2).sum().backward()

    expected_full = full.detach().clone().requires_grad_()
    expected = torch.fft.fftn(expected_full, dim=(1, 2))
    (expected.abs() ** 2).sum().backward()
    assert distributed_fft_calls == 2
    assert_close(x.grad, torch.split(expected_full.grad, shapes, dim=1)[rank])


@pytest.mark.parametrize("inverse,norm", [(False, None), (True, "ortho")])
def test_multi_sharded_transform_axes_fast_path(dist_env, mesh_2d, inverse, norm):
    sp_group = mesh_2d["sp"].get_group()
    dp_group = mesh_2d["dp"].get_group()
    sp_world_size = dist.get_world_size(sp_group)
    dp_world_size = dist.get_world_size(dp_group)
    if sp_world_size == 1 or dp_world_size == 1:
        pytest.skip("multi-sharded FFT fast path requires a non-degenerate 2D mesh")
    sp_rank = dist.get_rank(sp_group)
    dp_rank = dist.get_rank(dp_group)
    x_size = sp_world_size * sp_world_size * 3
    y_size = dp_world_size * dp_world_size * 2
    shapes = {
        "sp": es.helpers.compute_split_shapes(x_size, sp_world_size),
        "dp": es.helpers.compute_split_shapes(y_size, dp_world_size),
    }

    full = torch.randn(x_size, y_size, dtype=torch.complex64)
    x = torch.split(full, shapes["sp"], dim=0)[sp_rank]
    x = torch.split(x, shapes["dp"], dim=1)[dp_rank].contiguous()

    z = _assert_no_runtime_warning(
        lambda: es.einfft(
            "x/sp y/dp -> kx/sp ky/dp",
            x,
            axes={"x": "kx", "y": "ky"},
            inverse=inverse,
            norm=norm,
            mesh=mesh_2d,
            shapes={"sp": {"x": shapes["sp"], "kx": shapes["sp"]}, "dp": {"y": shapes["dp"], "ky": shapes["dp"]}},
        )
    )

    fft = torch.fft.ifftn if inverse else torch.fft.fftn
    expected = fft(full, dim=(0, 1), norm=norm)
    expected = torch.split(expected, shapes["sp"], dim=0)[sp_rank]
    expected = torch.split(expected, shapes["dp"], dim=1)[dp_rank]
    assert_close(z, expected)


def test_multi_sharded_transform_axes_fast_path_backward(dist_env, mesh_2d, monkeypatch):
    fft_impl = importlib.import_module("torch_einshard.fft")
    distributed_fft_calls = 0
    original_distributed_fft = fft_impl._distributed_fft_1d_no_autograd

    def counted_distributed_fft(*args, **kwargs):
        nonlocal distributed_fft_calls
        distributed_fft_calls += 1
        return original_distributed_fft(*args, **kwargs)

    monkeypatch.setattr(fft_impl, "_distributed_fft_1d_no_autograd", counted_distributed_fft)

    sp_group = mesh_2d["sp"].get_group()
    dp_group = mesh_2d["dp"].get_group()
    sp_world_size = dist.get_world_size(sp_group)
    dp_world_size = dist.get_world_size(dp_group)
    if sp_world_size == 1 or dp_world_size == 1:
        pytest.skip("multi-sharded FFT fast path requires a non-degenerate 2D mesh")
    sp_rank = dist.get_rank(sp_group)
    dp_rank = dist.get_rank(dp_group)
    x_size = sp_world_size * sp_world_size * 3
    y_size = dp_world_size * dp_world_size * 2
    shapes = {
        "sp": es.helpers.compute_split_shapes(x_size, sp_world_size),
        "dp": es.helpers.compute_split_shapes(y_size, dp_world_size),
    }

    full = torch.randn(x_size, y_size, dtype=torch.complex64)
    x = torch.split(full.detach().clone(), shapes["sp"], dim=0)[sp_rank]
    x = torch.split(x, shapes["dp"], dim=1)[dp_rank].contiguous().requires_grad_()
    z = es.einfft(
        "x/sp y/dp -> kx/sp ky/dp",
        x,
        axes={"x": "kx", "y": "ky"},
        mesh=mesh_2d,
        shapes={"sp": {"x": shapes["sp"], "kx": shapes["sp"]}, "dp": {"y": shapes["dp"], "ky": shapes["dp"]}},
    )
    (z.abs() ** 2).sum().backward()

    expected_full = full.detach().clone().requires_grad_()
    expected = torch.fft.fftn(expected_full, dim=(0, 1))
    (expected.abs() ** 2).sum().backward()
    expected_grad = torch.split(expected_full.grad, shapes["sp"], dim=0)[sp_rank]
    expected_grad = torch.split(expected_grad, shapes["dp"], dim=1)[dp_rank]
    assert distributed_fft_calls == 4
    assert_close(x.grad, expected_grad)


@pytest.mark.parametrize("inverse,norm", [(False, None), (False, "ortho"), (True, "forward")])
def test_sharded_transform_axis_fast_path_backward(dist_env, mesh_tp, monkeypatch, inverse, norm):
    fft_impl = importlib.import_module("torch_einshard.fft")
    distributed_fft_calls = 0
    original_distributed_fft = fft_impl._distributed_fft_1d_no_autograd

    def counted_distributed_fft(*args, **kwargs):
        nonlocal distributed_fft_calls
        distributed_fft_calls += 1
        return original_distributed_fft(*args, **kwargs)

    monkeypatch.setattr(fft_impl, "_distributed_fft_1d_no_autograd", counted_distributed_fft)

    group = mesh_tp["tp"].get_group()
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)
    size = world_size * world_size * 3
    shapes = es.helpers.compute_split_shapes(size, world_size)

    full = torch.randn(2, size, dtype=torch.complex64)
    x = torch.split(full.detach().clone(), shapes, dim=1)[rank].contiguous().requires_grad_()
    z = es.einfft(
        "b x/tp -> b k/tp",
        x,
        axes={"x": "k"},
        inverse=inverse,
        norm=norm,
        mesh=mesh_tp,
        shapes={"tp": {"x": shapes, "k": shapes}},
    )
    (z.abs() ** 2).sum().backward()

    expected_full = full.detach().clone().requires_grad_()
    fft = torch.fft.ifftn if inverse else torch.fft.fftn
    expected = fft(expected_full, dim=(1,), norm=norm)
    (expected.abs() ** 2).sum().backward()
    assert distributed_fft_calls == 2
    assert_close(x.grad, torch.split(expected_full.grad, shapes, dim=1)[rank])


def test_sharded_transform_axis_fast_path_permute_output_backward(dist_env, mesh_tp):
    group = mesh_tp["tp"].get_group()
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)
    size = world_size * world_size * 3
    shapes = es.helpers.compute_split_shapes(size, world_size)

    full = torch.randn(2, size, dtype=torch.complex64)
    x = torch.split(full.detach().clone(), shapes, dim=1)[rank].contiguous().requires_grad_()
    z = es.einfft(
        "b x/tp -> k/tp b",
        x,
        axes={"x": "k"},
        mesh=mesh_tp,
        shapes={"tp": {"x": shapes, "k": shapes}},
    )
    (z.abs() ** 2).sum().backward()

    expected_full = full.detach().clone().requires_grad_()
    expected = torch.fft.fftn(expected_full, dim=(1,)).mT
    (expected.abs() ** 2).sum().backward()
    assert_close(x.grad, torch.split(expected_full.grad, shapes, dim=1)[rank])


def test_sharded_transform_axis_slow_path_warns(dist_env, mesh_tp):
    group = mesh_tp["tp"].get_group()
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)
    if world_size == 1:
        pytest.skip("slow sharded FFT fallback warning requires more than one rank")
    size = world_size * 3 + 1
    shapes = es.helpers.compute_split_shapes(size, world_size)

    full = torch.randn(2, size, dtype=torch.complex64)
    x = torch.split(full, shapes, dim=1)[rank].contiguous()

    with pytest.warns(RuntimeWarning, match="gather/FFT/split fallback"):
        z = es.einfft(
            "b x/tp -> b k/tp",
            x,
            axes={"x": "k"},
            mesh=mesh_tp,
            shapes={"tp": {"x": shapes, "k": shapes}},
        )

    expected = torch.split(torch.fft.fftn(full, dim=(1,)), shapes, dim=1)[rank]
    assert_close(z, expected)


def test_sharded_transform_axis_permute_output(dist_env, mesh_tp):
    group = mesh_tp["tp"].get_group()
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)
    if world_size == 1:
        pytest.skip("slow sharded FFT fallback warning requires more than one rank")
    size = world_size * 3 + 1
    shapes = es.helpers.compute_split_shapes(size, world_size)

    full = torch.randn(2, size, dtype=torch.complex64)
    x = torch.split(full, shapes, dim=1)[rank].contiguous()

    with pytest.warns(RuntimeWarning, match="gather/FFT/split fallback"):
        z = es.einfft(
            "b x/tp -> k/tp b",
            x,
            axes={"x": "k"},
            mesh=mesh_tp,
            shapes={"tp": {"x": shapes, "k": shapes}},
        )

    expected = torch.split(torch.fft.fftn(full, dim=(1,)).mT, shapes, dim=0)[rank]
    assert_close(z, expected)


def test_sharded_output_axis(dist_env, mesh_tp):
    group = mesh_tp["tp"].get_group()
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)
    size = world_size * 3 + 1
    shapes = es.helpers.compute_split_shapes(size, world_size)

    x = torch.randn(2, size, dtype=torch.complex64)

    z = es.einfft(
        "b x -> b k/tp",
        x,
        axes={"x": "k"},
        mesh=mesh_tp,
        shapes=shapes,
    )

    expected = torch.split(torch.fft.fftn(x, dim=(1,)), shapes, dim=1)[rank]
    assert_close(z, expected)

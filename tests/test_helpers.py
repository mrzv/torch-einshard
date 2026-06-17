import torch_einshard as es


def test_init_process_group_cuda_without_local_rank_returns_full_tuple(monkeypatch):
    calls = []
    monkeypatch.delenv("SLURM_NTASKS", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setenv("RANK", "5")
    monkeypatch.setattr(es.helpers.dist, "init_process_group", lambda **kwargs: calls.append(kwargs))
    monkeypatch.setattr(es.helpers.torch.cuda, "device_count", lambda: 4)

    assert es.helpers.init_process_group("nccl", use_cuda=True) == (1, 5, 8)
    assert calls == [{"backend": "nccl", "rank": 5, "world_size": 8}]


def test_compute_split_shapes_single_chunk():
    assert es.helpers.compute_split_shapes(7, 1) == [7]


def test_compute_split_shapes_even_chunks():
    assert es.helpers.compute_split_shapes(8, 4) == [2, 2, 2, 2]


def test_compute_split_shapes_balanced_uneven_chunks():
    assert es.helpers.compute_split_shapes(10, 4) == [3, 3, 3, 1]


def test_compute_split_shapes_avoids_empty_last_chunk():
    assert es.helpers.compute_split_shapes(3, 4) == [0, 0, 0, 3]


def test_compute_split_shapes_for_factors_preserves_factor_boundaries():
    assert es.helpers.compute_split_shapes_for_factors(721, 4, 4) == [180, 180, 180, 181]


def test_compute_split_shapes_for_factors_matches_base_for_unit_factor():
    assert es.helpers.compute_split_shapes_for_factors(10, 4, 1) == es.helpers.compute_split_shapes(10, 4)


def test_resolve_split_shapes_none():
    assert es.helpers.resolve_split_shapes(None, "dp", "a") is None


def test_resolve_split_shapes_flat_list():
    assert es.helpers.resolve_split_shapes([2, 3], "dp", "a") == [2, 3]


def test_resolve_split_shapes_by_mesh_dimension():
    shapes = {"dp": [2, 3], "sp": [4, 5]}
    assert es.helpers.resolve_split_shapes(shapes, "sp", "a") == [4, 5]


def test_resolve_split_shapes_by_mesh_dimension_and_axis():
    shapes = {"dp": {"a": [2, 3], "b": [4, 5]}}
    assert es.helpers.resolve_split_shapes(shapes, "dp", "b") == [4, 5]


def test_resolve_split_shapes_missing_mesh_dimension():
    try:
        es.helpers.resolve_split_shapes({"dp": [2, 3]}, "sp", "a")
    except ValueError as error:
        assert "mesh dimension 'sp'" in str(error)
    else:
        raise AssertionError("Expected ValueError")


def test_resolve_split_shapes_missing_axis():
    try:
        es.helpers.resolve_split_shapes({"dp": {"a": [2, 3]}}, "dp", "b")
    except ValueError as error:
        assert "axis 'b'" in str(error)
        assert "mesh dimension 'dp'" in str(error)
    else:
        raise AssertionError("Expected ValueError")

import torch_einshard as es


def test_compute_split_shapes_single_chunk():
    assert es.helpers.compute_split_shapes(7, 1) == [7]


def test_compute_split_shapes_even_chunks():
    assert es.helpers.compute_split_shapes(8, 4) == [2, 2, 2, 2]


def test_compute_split_shapes_balanced_uneven_chunks():
    assert es.helpers.compute_split_shapes(10, 4) == [3, 3, 3, 1]


def test_compute_split_shapes_avoids_empty_last_chunk():
    assert es.helpers.compute_split_shapes(3, 4) == [0, 0, 0, 3]

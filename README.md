# torch-einshard

`torch-einshard` expresses local and distributed PyTorch tensor operations with
einsum-like axis and sharding notation.

## Installation

```sh
python -m pip install torch-einshard
```

The main entry point is `torch_einshard.einshard`:

```python
import torch_einshard as es

z = es.einshard("a b, b c -> a c", x, y)
```

Axis names work like `torch.einsum` subscripts. Sharded logical axes can also
name a PyTorch `DeviceMesh` dimension with `/`:

```text
a/sp b/dp, b/dp c -> a/sp c
```

This means axis `a` is sharded over mesh dimension `sp`, axis `b` is sharded
over `dp`, and the contraction over `b/dp` produces partial local results that
are all-reduced over `dp`.

Tensor-level partial values use `//`:

```text
b n h // tp -> b n h
```

A partial tensor has the full logical shape locally, but each rank holds only
one contribution to the value. Converting it back to a non-partial tensor
sum-reduces over the named mesh dimension.

## Quick Examples

Local contraction:

```python
z = es.einshard("a b k c, b c l d -> k l d a", x, y)
```

Distributed split:

```python
from torch.distributed.device_mesh import init_device_mesh

mesh = init_device_mesh("cpu", (2, 4), mesh_dim_names=("dp", "sp"))
z = es.einshard("a b -> a/sp b", x, mesh=mesh, shapes={"sp": a_shapes})
```

Tensor-parallel contraction:

```python
z = es.einshard("b n c, h/tp c -> b n h/tp", x, weight_shard, mesh=mesh)
```

Named-axis FFT:

```python
z = es.einfft("b x y c -> b kx ky c", x, axes={"x": "kx", "y": "ky"})
```

Halo exchange and local-window construction:

```python
patches = es.einwindow(
    "b h/sp_h w/sp_w c -> b h/sp_h w/sp_w kh kw c",
    x,
    {"h": "kh", "w": "kw"},
    {"h": 1, "w": 1},
    mesh=mesh,
    shapes={"sp_h": h_shapes, "sp_w": w_shapes},
)
```

## Documentation

The full documentation is available on
[Read the Docs](https://torch-einshard.readthedocs.io/en/latest/). Markdown
source is kept under
[`docs/`](https://github.com/lbnl-sciml/torch-einshard/tree/main/docs).

- [Getting started](https://torch-einshard.readthedocs.io/en/latest/getting-started.html)
  ([source](https://github.com/lbnl-sciml/torch-einshard/blob/main/docs/getting-started.md))
  covers installation, quickstart, and test commands.
- [Notation](https://torch-einshard.readthedocs.io/en/latest/notation.html)
  ([source](https://github.com/lbnl-sciml/torch-einshard/blob/main/docs/notation.md))
  covers axis, sharding, partial, factored-axis, family, and ellipsis notation.
- [Meshes and shapes](https://torch-einshard.readthedocs.io/en/latest/mesh-and-shapes.html)
  ([source](https://github.com/lbnl-sciml/torch-einshard/blob/main/docs/mesh-and-shapes.md))
  covers `DeviceMesh`, compound mesh groups, and uneven split metadata.

The rendered site's sidebar also provides the user guides, API reference, known
limitations, roadmap, performance notes, and design documentation.

Build the HTML docs with:

```sh
uv run sphinx-build -b html docs docs/_build/html
```

## Running Tests

Local and single-process tests:

```sh
uv run pytest
```

Full distributed test suite:

```sh
./run_tests.sh
```

`run_tests.sh` uses `torchrun --nproc-per-node 8`.

## Current Status

`torch-einshard` supports local einsum-style operations, selected distributed
split/gather/repartition patterns, tensor-parallel contractions, partial tensor
notation, named-axis FFTs, halo/window/convolution helpers, roll operations,
optimization-policy diagnostics, and parameter metadata helpers.

Known limitations are tracked in the
[limitations reference](https://torch-einshard.readthedocs.io/en/latest/reference/limitations.html)
([source](https://github.com/lbnl-sciml/torch-einshard/blob/main/docs/reference/limitations.md)).
Remaining feature and planning work is tracked in the
[roadmap](https://torch-einshard.readthedocs.io/en/latest/development/roadmap.html)
([source](https://github.com/lbnl-sciml/torch-einshard/blob/main/docs/development/roadmap.md)).

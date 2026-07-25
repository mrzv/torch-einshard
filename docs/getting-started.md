# Getting Started

Import the package as `torch_einshard`:

```python
import torch_einshard as es
```

## Installation

Install the package from PyPI:

```sh
python -m pip install torch-einshard
```

For development from a source checkout, install or sync the project with `uv`:

```sh
uv sync
```

For an editable source install into an existing environment, use:

```sh
uv pip install -e .
```

The main entry point is `einshard`:

```python
z = es.einshard("a b, b c -> a c", x, y)
```

The expression has the same broad shape as `torch.einsum`: input tensor specs on
the left, output tensor spec on the right, and named logical axes throughout.

## Local Operations

Local operations lower to `torch.einsum` with optional reshape operations for
factored axes:

```python
z = es.einshard("a b k c, b c l d -> k l d a", x, y)
z = es.einshard("b (h p) c -> b h p c", x, sizes={"p": 4})
```

## Distributed Operations

Distributed operations require a PyTorch `DeviceMesh`:

```python
from torch.distributed.device_mesh import init_device_mesh

mesh = init_device_mesh("cpu", (2, 4), mesh_dim_names=("dp", "sp"))
z = es.einshard("a/sp b/dp, b/dp c -> a/sp c", x, y, mesh=mesh)
```

Use `shapes` when a collective needs split metadata, especially for uneven
shards:

```python
z = es.einshard(
    "a b -> a/sp b/dp",
    x,
    mesh=mesh,
    shapes={"sp": sp_shapes, "dp": dp_shapes},
)
```

## Building Documentation

The documentation uses Sphinx with MyST Markdown:

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

# torch-einshard

`torch-einshard` expresses local and distributed PyTorch tensor operations with einsum-like axis and sharding notation.

The main entry point is `torch_einshard.einshard`:

```python
import torch_einshard as es

z = es.einshard("a b, b c -> a c", x, y)
```

Axis names work like `torch.einsum` subscripts, but sharded logical axes can additionally name a mesh dimension with `/`:

```text
a/sp b/dp, b/dp c -> a/sp c
```

This means:

- axis `a` is sharded over mesh dimension `sp`
- axis `b` is sharded over mesh dimension `dp`
- the contraction over `b/dp` produces a partial local result that is all-reduced over `dp`

Distributed operations expect a PyTorch `DeviceMesh`:

```python
from torch.distributed.device_mesh import init_device_mesh

mesh = init_device_mesh("cpu", (2, 4), mesh_dim_names=("dp", "sp"))
z = es.einshard("a/sp b/dp, b/dp c -> a/sp c", x, y, mesh=mesh)
```

## Supported Local Patterns

Local operations are translated to `torch.einsum`.

Contraction and permutation:

```python
z = es.einshard("a b k c, b c l d -> k l d a", x, y)
```

Outer product:

```python
z = es.einshard("i, j -> i j", x, y)
```

Diagonal extraction:

```python
z = es.einshard("i i -> i", x)
```

Local axis permutation:

```python
z = es.einshard("b t c h w -> b t h w c", x)
```

## Supported Distributed Unary Patterns

Unary distributed operations support split, gather, multi-axis split/gather, and gather-then-split repartition.

Single-axis split:

```python
z = es.einshard("a b -> a/dp b", x, mesh=mesh, shapes=shapes)
```

Single-axis gather:

```python
z = es.einshard("a/dp b -> a b", x, mesh=mesh, shapes=shapes)
```

Multi-axis split:

```python
z = es.einshard(
    "a b -> a/sp b/dp",
    x,
    mesh=mesh,
    shapes={"sp": sp_shapes, "dp": dp_shapes},
)
```

Multi-axis gather:

```python
z = es.einshard(
    "a/sp b/dp -> a b",
    x,
    mesh=mesh,
    shapes={"sp": sp_shapes, "dp": dp_shapes},
)
```

Split or gather followed by output permutation:

```python
z = es.einshard("b a -> a/dp b", x, mesh=mesh, shapes=shapes)
```

Repartition from one logical axis to another over the same mesh dimension:

```python
z = es.einshard(
    "a/dp b -> a b/dp",
    x,
    mesh=mesh,
    shapes={"dp": {"a": a_shapes, "b": b_shapes}},
)
```

Current repartition semantics are correctness-first: gather the source sharded axis, then split the destination axis. A future implementation may replace this with all-to-all where possible.

## Supported Distributed Binary Patterns

Binary distributed contractions support one sharded contracted axis. The local contraction is computed first and then all-reduced over the contracted shard dimension.

Generic form:

```text
... k/p, k/p ... -> ...
```

Example:

```python
z = es.einshard("a/sp b/dp, b/dp c -> a/sp c", x, y, mesh=mesh)
```

This contracts `b/dp` and all-reduces the output over `dp`.

## Tensor-Parallel Linear Patterns

Column-parallel linear projection is supported by sharding the output feature axis:

```python
z = es.einshard("b n c, h/tp c -> b n h/tp", x, weight_shard, mesh=mesh)
```

This produces local shards of output features and does not require a forward all-reduce.

Row-parallel linear projection is supported by sharding the contracted input feature axis:

```python
z = es.einshard("b n c/tp, h c/tp -> b n h", x_shard, weight_shard, mesh=mesh)
```

This contracts `c/tp` and all-reduces over `tp`.

A two-layer MLP can be expressed as:

```text
b n c, h/tp c -> b n h/tp
b n h/tp, c h/tp -> b n c
```

Attention projections follow the same pattern:

```text
b l c, q/tp c -> b l q/tp
b l c, k/tp c -> b l k/tp
b l c, v/tp c -> b l v/tp
b l v/tp, c v/tp -> b l c
```

## Distributed Roll

`einroll` rolls tensors described by sharded axis notation.

Single-axis roll:

```python
z = es.einroll("a/dp b", x, {"a": 2}, mesh=mesh, shapes=shapes)
```

Multi-axis roll:

```python
z = es.einroll(
    "a/sp b/dp",
    x,
    {"a": -2, "b": 3},
    mesh=mesh,
    shapes={"sp": sp_shapes, "dp": dp_shapes},
)
```

Current `einroll` semantics are correctness-first: gather each sharded rolled axis, apply `torch.roll`, then split back. A future implementation may optimize this with neighbor exchange or all-to-all.

## Low-Level Autograd Mappings

The package also exposes custom autograd mappings in `torch_einshard.mappings`.

All-reduce in forward, identity in backward:

```python
allreduce_forward_identity_backward(x, comm)
```

Identity in forward, all-reduce in backward:

```python
identity_forward_allreduce_backward(x, comm)
```

All-gather in forward, split in backward:

```python
allgather_forward_split_backward(x, comm, dim, shapes)
```

Split in forward, all-gather in backward:

```python
split_forward_allgather_backward(x, comm, dim, shapes)
```

Reduce-scatter in forward, all-gather in backward:

```python
reducescatter_forward_allgather_backward(x, comm, dim, shapes)
```

All-gather in forward, reduce-scatter in backward:

```python
allgather_forward_reducescatter_backward(x, comm, dim, shapes)
```

The current `reduce_scatter` helper is implemented as all-reduce followed by split for backend portability.

## Shape Metadata

`shapes` controls split sizes. If omitted, split sizes are computed with `compute_split_shapes`.

Single mesh dimension:

```python
shapes = [4, 4, 4, 4]
```

Multiple mesh dimensions:

```python
shapes = {
    "sp": [4, 4],
    "dp": [8, 8, 8, 8],
}
```

Same mesh dimension used for different logical axes:

```python
shapes = {
    "dp": {
        "a": a_shapes,
        "b": b_shapes,
    }
}
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

## Current Limitations

- The grammar supports at most two input tensors.
- General multi-dimensional distributed contractions are not implemented.
- Partial-value notation is not yet represented in `einshard` syntax.
- Autograd-only communication is available as low-level mappings, not as notation.
- Repartition and `einroll` use correctness-first gather/split implementations rather than optimized all-to-all or neighbor exchange.
- `src/torch_einshard/mesh.py` is incomplete; tests and examples use PyTorch `DeviceMesh`.

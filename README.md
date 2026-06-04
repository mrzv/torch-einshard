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

Mesh dimension names may include hyphens, for example `tp-sp`, if the supplied `DeviceMesh` uses that exact mesh-dimension name.

Tensor-level partial values use `//`. A partial tensor has the full logical shape locally, but each rank only holds one contribution to the value. The contributions are sum-reduced over the named mesh dimension when converting back to a non-partial tensor:

```text
b n h // tp -> b n h
```

Multiple partial dimensions are written with parentheses. They are currently reduced sequentially in the listed order:

```text
loss // (sp,dp) -> loss
```

Distributed operations expect a PyTorch `DeviceMesh`:

```python
from torch.distributed.device_mesh import init_device_mesh

mesh = init_device_mesh("cpu", (2, 4), mesh_dim_names=("dp", "sp"))
z = es.einshard("a/sp b/dp, b/dp c -> a/sp c", x, y, mesh=mesh)
```

Compound mesh groups can be enabled by wrapping a PyTorch `DeviceMesh`:

```python
mesh = es.wrap_mesh(mesh)
z = es.einshard("loss // dp-sp -> loss", loss, mesh=mesh)
```

Compound names such as `dp-sp` span the listed mesh dimensions while preserving any remaining mesh coordinates.
Compound process groups are created lazily on first lookup and cached on the wrapped mesh. Reuse the wrapped mesh instance instead of calling `wrap_mesh` repeatedly; equivalent names such as `dp-sp` and `sp-dp` share the same cached group.

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

Einsum ellipses are supported for unnamed local dimensions:

```python
z = es.einshard("... c, c o -> ... o", x, w)
z = es.einshard("... c -> c", x)
```

Distributed unary split/gather and tensor-parallel binary contractions can also use ellipses for unsharded leading dimensions:

```python
z = es.einshard("... a -> ... a/dp", x, mesh=mesh, shapes=shapes)
z = es.einshard("... c, h/tp c -> ... h/tp", x, w, mesh=mesh)
```

Factored axes use one-level parenthesized groups. Grouped input dimensions are expanded before the local `einsum`, and grouped output dimensions are packed after it:

```python
z = es.einshard("b (h p) c -> b h p c", x, sizes={"p": 4})
z = es.einshard("b h p c -> b (h p) c", z)
```

Factor sizes are inferred from tensor dimensions when exactly one factor in a group is omitted from `sizes`. Supplying no sizes for a group such as `(h p)` is ambiguous and raises `ValueError`.

Axis families remove repeated 2D/3D notation boilerplate. `*family` expands to a sequence of axes, and `[*spatial *window]` zips families into repeated factored groups:

```python
spatial = ("h", "w", "d")[:dim]
window = ("wh", "ww", "wd")[:dim]

windows = es.einshard(
    "b t [*spatial *window] c -> (b *spatial) t *window c",
    x,
    families={"spatial": spatial, "window": window},
    sizes={"window": window_size},
)
```

For `dim == 2`, this expands to `b t (h wh) (w ww) c -> (b h w) t wh ww c`. For `dim == 3`, it expands to `b t (h wh) (w ww) (d wd) c -> (b h w d) t wh ww wd c`.

## Supported Distributed Unary Patterns

Unary distributed operations support split, gather, multi-axis split/gather, optimized ownership swaps, and gather-then-split repartition.

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

Repartition one logical axis from one mesh dimension to another:

```python
z = es.einshard(
    "a/dp b -> a/sp b",
    x,
    mesh=mesh,
    shapes={"dp": {"a": dp_shapes}, "sp": {"a": sp_shapes}},
)
```

Current repartition semantics are correctness-first: gather the source sharded axis, then split the destination axis. Same-mesh axis-to-axis repartition uses a point-to-point all-to-all-style exchange when split metadata is available, and falls back to gather/split otherwise. Pure multi-axis ownership swaps across equal-sized mesh dimensions, such as `b h/sp1 w/sp2 c -> b h/sp2 w/sp1 c`, exchange local blocks directly when matching split metadata is available. Performance-sensitive repartition fallbacks emit `RuntimeWarning`.

Partial-to-full all-reduce:

```python
z = es.einshard("a b // tp -> a b", x, mesh=mesh)
```

Partial-to-shard reduce-scatter:

```python
z = es.einshard("a b // tp -> a/tp b", x, mesh=mesh, shapes=shapes)
```

Shard-to-partial all-gather with reduce-scatter backward:

```python
z = es.einshard("a/tp b -> a b // tp", x, mesh=mesh, shapes=shapes)
```

Full-to-partial output keeps the forward value unchanged and all-reduces gradients in backward:

```python
z = es.einshard("a b -> a b // tp", x, mesh=mesh)
```

Scalar reductions can also use partial notation:

```python
z = es.einshard("loss // (sp,dp) -> loss", loss, mesh=mesh)
```

Partial notation currently represents sum reductions only.

## Supported Distributed Binary Patterns

Binary distributed contractions support one or more sharded contracted axes. The local contraction is computed first and then all-reduced over each contracted shard dimension.

Generic form:

```text
... k/p, k/p ... -> ...
```

Example:

```python
z = es.einshard("a/sp b/dp, b/dp c -> a/sp c", x, y, mesh=mesh)
```

This contracts `b/dp` and all-reduces the output over `dp`.

Multiple sharded contracted axes are reduced sequentially:

```python
z = es.einshard("a/dp b/sp c, a/dp b/sp d -> c d", x, y, mesh=mesh)
```

Shared sharded axes can also appear in both inputs and the output, which covers distributed batched matmul-style layouts:

```python
z = es.einshard("b/dp a c, b/dp c d -> b/dp a d", x, y, mesh=mesh)
```

The partial output can also be requested explicitly:

```python
z = es.einshard("a/sp b/dp, b/dp c -> a/sp c // dp", x, y, mesh=mesh)
```

In that case, `z` is the local partial contraction result and no forward all-reduce is applied.

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

To keep the local partial output instead of all-reducing immediately:

```python
z = es.einshard("b n c/tp, h c/tp -> b n h // tp", x_shard, weight_shard, mesh=mesh)
```

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

Current `einroll` semantics are correctness-first. Sharded axes with explicit `shapes` use direct point-to-point slice exchange; axes without shape metadata fall back to gather, `torch.roll`, and split with a `RuntimeWarning`.

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

The current `reduce_scatter` helper is implemented as all-reduce followed by split for backend portability. Uneven `all_gather` uses a padded equal-size gather internally for backends that reject variable-size gathers.

## Roadmap

Remaining work is tracked in `PLAN.md`. The main open areas are:

- Optional notation for autograd-only communication.
- More general multi-axis repartition where changed mesh dimensions are not a pure ownership swap.
- Nested or nonlocal distributed factored-axis transforms, if concrete use cases need them.
- Checkpoint-style shard metadata derived from parameter specs.

## Parameter Metadata

`ParamSpec` describes persistent parameter layout plus synchronization and gradient-reduction metadata. It does not replace `einshard`; tensor computations still use explicit `einshard(...)` expressions.

```python
mesh = es.wrap_mesh(mesh)
weight_spec = es.ParamSpec(
    "out/tp in",
    shared=("sp1-sp2",),
    reduce=("sp1-sp2",),
)

weight = torch.nn.Parameter(local_weight)
es.sync_param_(weight, weight_spec, mesh)

y = es.einshard(
    "batch in, out/tp in -> batch out/tp",
    x,
    weight,
    mesh=mesh,
    shapes={"tp": {"out": out_shapes}},
)
loss = y.sum()
loss.backward()
es.reduce_grad_(weight, weight_spec, mesh)
```

`shared` uses broadcast from group rank 0 to make replicated parameter values identical. `reduce` uses sum all-reduce on `param.grad`. Compound names such as `sp1-sp2` work with `wrap_mesh` and reuse the wrapped mesh's cached compound process groups.

`shared` dimensions cannot overlap with axis shard dimensions in the layout. For example, `ParamSpec("out/tp in", shared="tp")` is rejected because `out` is already sharded over `tp`.

For modules, attach specs to parameters and use the module-level helpers:

```python
es.set_param_spec(weight, weight_spec)
es.sync_module_params_(module, mesh)
es.reduce_module_grads_(module, mesh)
```

When using PyTorch DDP, SciGPT-style extra gradient reductions can be registered as a DDP communication hook:

```python
ddp = torch.nn.parallel.DistributedDataParallel(module, process_group=mesh["dp"].get_group())
es.register_grad_reduction_hook_(ddp, mesh, ddp_group="dp")
```

The hook performs DDP-style averaging over `ddp_group`, then applies sum all-reduces for attached `ParamSpec.reduce` groups.

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

When a dict form is supplied, missing mesh dimensions or missing axis-specific entries are reported as `ValueError`s before the collective runs.

Same mesh dimension used for different logical axes:

```python
shapes = {
    "dp": {
        "a": a_shapes,
        "b": b_shapes,
    }
}
```

Factor-aware split sizes preserve divisibility for patch/factor operations before assigning any remainder to the final shard:

```python
shapes = es.helpers.compute_split_shapes_for_factors(size=721, num_chunks=4, factor=4)
# [180, 180, 180, 181]
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
- Partial notation represents sum reductions only.
- Factored axes support one-level parenthesized groups; grouped transformations are local reshape operations around `torch.einsum`.
- Axis families are a pre-parse notation expansion; expanded expressions must still be valid `einshard` notation.
- Repartition and `einroll` still use correctness-first gather/split fallbacks for unsupported cases or missing shape metadata; performance-sensitive fallbacks emit `RuntimeWarning`.
- Multi-axis repartition swaps are currently limited to pure ownership swaps across equal-sized mesh dimensions with matching split metadata.
- Parenthesized partial reductions over multiple mesh dimensions are applied sequentially; use `wrap_mesh` with a compound name for a single compound group reduction.

Invalid public `einshard` and `einroll` expressions raise `ValueError` with the original expression included in the message.

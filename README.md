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

## FFT

`einfft` applies `torch.fft.fftn` or `torch.fft.ifftn` over named axes while using sharding notation for input and output layout.

```python
z = es.einfft("b x c -> b k c", x, axes={"x": "k"})
```

The `axes` mapping names each transformed input axis and the corresponding output frequency axis. Multiple axes use a single multidimensional FFT:

```python
z = es.einfft(
    "b x y c -> b kx ky c",
    x,
    axes={"x": "kx", "y": "ky"},
    norm="ortho",
)
```

Use `inverse=True` for `torch.fft.ifftn`:

```python
x = es.einfft("b k c -> b x c", z, axes={"k": "x"}, inverse=True)
```

Use `real=True` for real FFT variants. Forward real mode calls `torch.fft.rfftn`; inverse real mode calls `torch.fft.irfftn`. The last axis in the `axes` mapping is the half-spectrum axis, matching PyTorch's `rfftn`/`irfftn` convention:

```python
z = es.einfft("b x y -> b kx ky", x, axes={"x": "kx", "y": "ky"}, real=True)
x = es.einfft(
    "b kx ky -> b x y",
    z,
    axes={"kx": "x", "ky": "y"},
    inverse=True,
    real=True,
    signal_sizes={"y": y_size},
)
```

`signal_sizes` is only needed for inverse real FFTs when the original length cannot be inferred from the half-spectrum size, such as odd-length signals.

Sharded transform axes are supported. The optimized path handles complex-to-complex FFTs when each sharded transform axis stays on the same mesh dimension with equal shard sizes and local shard size divisible by the mesh size. Multiple sharded transform axes are supported when they use distinct mesh dimensions. The optimized path uses a distributed Cooley-Tukey decomposition with all-to-all transposes and local factor FFTs.

```python
z = es.einfft(
    "b x/tp -> b k/tp",
    x_shard,
    axes={"x": "k"},
    mesh=mesh,
    shapes={"tp": {"x": x_shapes, "k": k_shapes}},
)
```

Unsupported sharded transform layouts, including sharded real FFT variants, fall back to gather the full transform axis, run the local FFT, then split the output frequency axis if requested. That path emits a `RuntimeWarning` because it materializes the full transform axis on every rank. Non-FFT axes must preserve their sharding. `einfft` currently supports explicit named axes; it does not support ellipsis axes, factored axes, or partial tensor specs.

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

Binary distributed operations support sharded contractions and sharded elementwise products. For contractions, the local contraction is computed first and then all-reduced over each contracted shard dimension, unless the output explicitly keeps the partial value or the contracted mesh dimension is reused to shard an output axis.

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

If one operand has a replicated shared axis and the output requests the sharded layout, that operand is split before the local operation:

```python
z = es.einshard("b/dp a c, b c d -> b/dp a d", x_shard, y, mesh=mesh, shapes=shapes)
```

Sharded elementwise products use the same normalization path:

```python
z = es.einshard("l/tp e, l/tp e -> l/tp e", x_shard, y_shard, mesh=mesh, shapes=shapes)
z = es.einshard("l/tp e, l e -> l/tp e", x_shard, y, mesh=mesh, shapes=shapes)
```

The partial output can also be requested explicitly:

```python
z = es.einshard("a/sp b/dp, b/dp c -> a/sp c // dp", x, y, mesh=mesh)
```

In that case, `z` is the local partial contraction result and no forward all-reduce is applied.

Binary outputs can request different sharding for free axes than the inputs use. `einshard` normalizes those input axes before the local contraction with the necessary split or gather collectives:

```python
z = es.einshard("l/tp e, e f/tp -> l f/tp", x_shard, y_shard, mesh=mesh, shapes=shapes)
z = es.einshard("l/tp e, e f -> l f", x_shard, y, mesh=mesh, shapes=shapes)
z = es.einshard("l e, e f -> l/tp f", x, y, mesh=mesh, shapes=shapes)
```

Contracted axes can also be normalized when one operand has the full contracted dimension and the other is sharded:

```python
z = es.einshard("l f, f/tp e -> l e", x, y_shard, mesh=mesh, shapes=shapes)
z = es.einshard("l/tp f, f/tp e -> l/tp e", x_shard, y_shard, mesh=mesh, shapes=shapes)
```

For uneven shards, pass split metadata for each affected logical axis and mesh dimension:

```python
shapes = {"tp": {"l": l_shapes, "f": f_shapes}}
```

If a sharded contraction dimension is reused to shard a free output axis, `einshard` reduce-scatters the partial contraction result instead of all-reducing it:

```python
z = es.einshard("l f/tp, f/tp e -> l/tp e", x_shard, y_shard, mesh=mesh, shapes=shapes)
```

This computes the local partial `l e // tp` and then reduce-scatters it to `l/tp e`.

The same contraction can keep the partial value while still sharding a free output axis over that mesh dimension:

```python
z = es.einshard("l f/tp, f/tp e -> l/tp e // tp", x_shard, y_shard, mesh=mesh, shapes=shapes)
```

In this form no forward all-reduce is applied. The local partial `l e // tp` is split to `l/tp e // tp`, so each rank keeps its partial contribution for its output shard.

The contracted axis can be sharded over different mesh dimensions in each input while the reduction mesh dimension is reused for a different output axis:

```python
z = es.einshard(
    "l/sp e/dp, e/sp f/dp -> l/sp f/dp",
    x_shard,
    y_shard,
    mesh=mesh,
    shapes=shapes,
)
```

Here `e` is contracted, `f/dp` is gathered before `e` is normalized, and the local partial result is reduce-scattered back to `f/dp`.

A free axis can also be local in both inputs and sharded only in the output when a contracted mesh dimension supplies the reduction:

```python
z = es.einshard(
    "b l e/dp, b e/sp f -> b/dp l f",
    x_shard,
    y_shard,
    mesh=mesh,
    shapes=shapes,
)
```

Here the local contraction produces `b l f // dp`, then the result is reduce-scattered along the output batch axis to `b/dp l f`.

Multiple crossed contracted axes are supported when their shard-dimension changes form a pure owner-swap-compatible permutation:

```python
z = es.einshard(
    "a k/dp m/sp, k/sp m/dp b -> a/sp b/dp",
    x_shard,
    y_shard,
    mesh=mesh,
    shapes=shapes,
)
```

This owner-swap path currently requires equal-sized mesh dimensions and matching split metadata for the swapped axes. Other multi-crossed contracted-axis layouts raise `NotImplementedError` instead of falling back to unsafe sequential repartitioning.

When one free axis moves from sharded to local and another moves from local to sharded over the same mesh dimension, `einshard` can keep the local contraction in the source-sharded layout and repartition the result with an all-to-all-style exchange:

```python
z = es.einshard("l/tp e, e f -> l f/tp", x_shard, y, mesh=mesh, shapes=shapes)
```

This avoids materializing the full `l f` result on every rank when split metadata is available for both `l` and `f`.

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

With sequence-parallel activations, the MLP keeps the sequence axis sharded at the block boundary:

```text
l/tp e, e f/tp -> l f/tp
l f/tp, f/tp e -> l/tp e
```

The first projection gathers `l/tp` to full `l` while producing the sharded hidden axis `f/tp`. The second projection contracts over `f/tp` and reduce-scatters the result back to `l/tp`. For uneven shards, provide split metadata for both logical axes:

```python
shapes = {"tp": {"l": l_shapes, "f": f_shapes}}
```

Attention projections follow the same pattern:

```text
b l c, q/tp c -> b l q/tp
b l c, k/tp c -> b l k/tp
b l c, v/tp c -> b l v/tp
b l v/tp, c v/tp -> b l c
```

Sequence-parallel attention score and value contractions are also expressible:

```text
b h q/tp d, b h k/tp d -> b h q/tp k
b h q/tp k, b h k/tp d -> b h q/tp d
```

The score pattern gathers the sharded key axis so each query shard can attend over full `k`. The value pattern splits the contracted `k` axis to match `v`, computes a local partial over full `q`, and reduce-scatters back to `q/tp`. For uneven sequence shards, provide both query and key split metadata:

```python
shapes = {"tp": {"q": q_shapes, "k": k_shapes}}
```

## Ghost Cells And Windows

`einhalo` extends one or more axes with ghost cells. Local axes are padded directly. Sharded axes exchange only the needed boundary intervals with owning ranks, then use autograd to route halo gradients back to those owners.

```python
xg = es.einhalo(
    "b h/sp_h w/sp_w c",
    x,
    {"h": 1, "w": 1},
    mesh=mesh,
    shapes={"sp_h": h_shapes, "sp_w": w_shapes},
)
```

Halo widths can be symmetric integers or `(left, right)` pairs:

```python
xg = es.einhalo("h/sp c", x, {"h": (2, 1)}, mesh=mesh, shapes=shapes)
```

Supported boundaries are `constant`, `periodic`, and `replicate`. Constant boundaries use `fill=0` by default:

```python
xg = es.einhalo("h/sp c", x, {"h": 1}, boundary="constant", fill=-1, mesh=mesh, shapes=shapes)
```

`einwindow` is a sharding-aware `unfold`/`im2col`. It first applies the needed halo, then preserves the owned center axis and adds an explicit local window axis:

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

For a 3x3 convolution, the window axes can then be contracted normally:

```python
y = es.einshard(
    "b h/sp_h w/sp_w kh kw c, o kh kw c -> b h/sp_h w/sp_w o",
    patches,
    weight,
    mesh=mesh,
)
```

`einconv` provides the lower-memory convolution path directly. It applies the
needed `einhalo` padding, calls PyTorch `conv1d`/`conv2d`/`conv3d` without
materializing im2col windows, and checkpoints the full halo-plus-convolution
forward by default so backward recomputes activations instead of saving them:

```python
y = es.einconv(
    "b h/sp_h w/sp_w c, o c kh kw -> b h/sp_h w/sp_w o",
    x,
    weight,
    {"h": "kh", "w": "kw"},
    bias=bias,
    mesh=mesh,
    shapes={"sp_h": h_shapes, "sp_w": w_shapes},
)
```

The `checkpoint` option defaults to `"full"`. Use `checkpoint="conv"` to
checkpoint only the local convolution after halo exchange, or `checkpoint=False`
to disable checkpointing. The initial implementation supports 1D/2D/3D
convolutions with stride 1, one input-channel axis, one output-channel axis, and
local kernel axes. Padding must preserve each spatial length; omitted padding
defaults to same-padding for odd effective kernel sizes. Grouped convolutions are
not supported yet.

For neighborhood attention, window the key and value tensors, then contract the query only against local neighborhood axes instead of a full sequence/spatial axis.

Axis families work for both halo/window widths and window-axis names:

```python
patches = es.einwindow(
    "b *spatial c -> b *spatial *window c",
    x,
    {"spatial": ("kh", "kw")},
    {"spatial": (1, 1)},
    families={"spatial": ("h/sp_h", "w/sp_w"), "window": ("kh", "kw")},
    mesh=mesh,
    shapes={"sp_h": h_shapes, "sp_w": w_shapes},
)
```

Sharded halo exchange handles uneven shards, halos larger than a local shard, and `periodic` wraparound by slicing the requested ghost interval into rank-owned pieces.

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

Axis families can make dimension-polymorphic roll notation concise. Family-valued shifts expand to per-axis shifts, and family entries may include sharding suffixes:

```python
z = es.einroll(
    "*spatial",
    x,
    {"spatial": (-2, 3)},
    mesh=mesh,
    shapes={"sp": sp_shapes, "dp": dp_shapes},
    families={"spatial": ("h/sp", "w/dp")},
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
for name, param, spec in es.iter_param_specs(module):
    print(name, spec.layout)
es.sync_module_params_(module, mesh)
es.reduce_module_grads_(module, mesh)
```

When using PyTorch DDP, SciGPT-style extra gradient reductions can be registered as a DDP communication hook:

```python
ddp = torch.nn.parallel.DistributedDataParallel(module, process_group=mesh["dp"].get_group())
es.register_grad_reduction_hook_(ddp, mesh, ddp_group="dp")
```

The hook performs DDP-style averaging over `ddp_group`, then applies sum all-reduces for attached `ParamSpec.reduce` groups.

For buckets where every parameter has the same extra reduction metadata, the hook can combine DDP averaging and the extra reduction into one all-reduce over a compound group:

```python
es.register_grad_reduction_hook_(
    ddp,
    mesh,
    ddp_group="dp",
    combined_reduce_group="dp-sp1-sp2",
    combined_reduce="sp1-sp2",
)
```

This is equivalent to summing the bucket over `dp-sp1-sp2` and then dividing by the `dp` group size. It avoids a separate `dp` all-reduce followed by an `sp1-sp2` all-reduce. The fast path is used only when all parameters in a bucket have exactly `reduce=("sp1-sp2",)`. Mixed buckets fall back to the per-parameter-spec reductions.

Parameter specs can also derive checkpoint/test-copy shard metadata:

```python
metadata = es.param_shard_metadata(weight_spec, global_shape=(1024, 2048), mesh=mesh)
local_slices = metadata.local_slices
local_shape = metadata.local_shape
```

Use `es.param_local_slices(...)`, `es.param_local_shape(...)`, or `es.param_shard_dims(...)` when only one piece of metadata is needed. These helpers support single mesh dimensions and compound names such as `dp-sp` via `wrap_mesh`. Factor-aware splits can be requested for patch-like cases:

```python
local_slices = es.param_local_slices(
    es.ParamSpec("height/sp1 width/sp2"),
    global_shape=(721, 1440),
    mesh=mesh,
    factors={"height": 4, "width": 4},
)
```

Shard metadata helpers require an initialized process group and report missing mesh dimensions or raw `DeviceMesh` compound-group use as explicit errors. Sharded factored-axis groups are intentionally rejected until factored parameter metadata is represented directly.

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

Performance benchmark structure and CI policy are planned in `PERFORMANCE.md`. Benchmarks should remain opt-in and separate from the correctness test suites.

Quick local benchmark smoke test:

```sh
uv run python benchmarks/bench_local.py --warmup 1 --iters 2 --size small --device cpu
```

Full benchmark runner:

```sh
NPROC=8 ./run_benchmarks.sh --device cpu --size small --output results.jsonl
```

## Current Limitations

- The grammar supports at most two input tensors.
- Partial notation represents sum reductions only.
- Factored axes support one-level parenthesized groups; grouped transformations are local reshape operations around `torch.einsum`.
- Axis families are a pre-parse notation expansion; expanded expressions must still be valid `einshard` notation.
- Repartition and `einroll` still use correctness-first gather/split fallbacks for unsupported cases or missing shape metadata; performance-sensitive fallbacks emit `RuntimeWarning`.
- Multi-axis repartition swaps are currently limited to pure ownership swaps across equal-sized mesh dimensions with matching split metadata.
- Parenthesized partial reductions over multiple mesh dimensions are applied sequentially; use `wrap_mesh` with a compound name for a single compound group reduction.

Invalid public `einshard` and `einroll` expressions raise `ValueError` with the original expression included in the message.

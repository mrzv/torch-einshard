# Distributed Pattern Plan

This document tracks remaining distributed-pattern work inspired by `../MachineLearning/SciGPT/scaling-transformers-physical-sciences`.

Implemented coverage includes local einsum-style operations, local one-level factored axes, multi-axis unary split/gather, tensor-parallel linear contraction patterns, distributed contractions over one or more sharded contracted axes, low-level identity/all-reduce and reduce-scatter autograd mappings, `//` partial-value notation, gather-then-split repartition semantics, optimized same-mesh repartition, pure multi-axis ownership swaps, compound mesh groups, and `einroll` with correctness-first gather/roll/split behavior.

The remaining work is mostly about broadening notation expressiveness, reducing communication overhead, and deciding whether higher-level model/parameter metadata belongs in this package.

## Factored Axes

Patch expansion and unpatching use one-level parenthesized groups to represent factored logical axes.

Implemented notation:

```text
b t (h p) (w q) c -> b t c h p w q
b t h p w q c -> b t (h p) (w q) c
```

Implemented behavior:

- Factor groups are first-class parser nodes but lower to local reshape operations around `torch.einsum`.
- Factor sizes are supplied with the public `sizes` argument.
- Exactly one omitted factor size per group can be inferred from the concrete tensor dimension.
- Sharded annotations inside a group are allowed when the operation remains local, for example `(h/sp p) -> h/sp p`.

Remaining work:

- Decide whether nested groups are needed.
- Decide whether nonlocal distributed operations should allow grouped axes beyond local reshape-only transformations.

## Further Repartition Optimization

Unary repartition semantics are covered today by gather-then-split, including a single logical axis moving from one mesh dimension to another. Same-mesh axis-to-axis repartition uses a point-to-point all-to-all-style exchange when metadata is available, with gather/split retained as fallback. Pure multi-axis ownership swaps across equal-sized mesh dimensions directly exchange local blocks when matching split metadata is available.

Example:

```text
b h/sp1 w c -> b h w/sp1 c
b h/sp1 w c -> b h/sp2 w c
```

Implemented ownership swap:

- Multi-axis ownership swaps across spatial mesh dimensions:

```text
b h/sp1 w/sp2 c -> b h/sp2 w/sp1 c
```

Remaining optimization:

- Support more general multi-axis repartition where changed mesh dimensions are not a pure permutation or split metadata differs between source and destination ownership.
- Preserve the current gather-then-split behavior as a correctness fallback for uneven or unsupported cases, with warnings when the fallback may be expensive.

## Compound Groups

SciGPT uses groups such as `sp1-sp2`, `tp-sp1-sp2`, and `dp-sp1-sp2`. Hyphenated names are accepted in sharding and `//` notation, and `wrap_mesh` resolves them as lazily cached compound groups over PyTorch `DeviceMesh` dimensions.

Axis-wise notation can already describe many operations over separate dimensions:

```text
b h/sp1 w/sp2 c -> b h w c
```

Implemented behavior:

- `// (sp1,sp2)` reduces sequentially in the listed order.
- `// sp1-sp2` reduces over one compound group when the mesh is wrapped with `wrap_mesh`.
- Equivalent compound names such as `sp1-sp2` and `sp2-sp1` share a cached process group on the wrapped mesh.

Remaining work:

- Add checkpoint-style shard metadata tests if a concrete metadata API is added.

## Parameter Metadata

Parameter sharding is expressible with axis notation. `ParamSpec` now represents persistent parameter layout plus shared-value and gradient-reduction metadata.

Implemented Python API:

```python
es.ParamSpec("o c", shared=("tp-sp1-sp2",), reduce=("sp1-sp2",))
es.ParamSpec("o/tp c", shared=("sp1-sp2",), reduce=("sp1-sp2",))
es.ParamSpec("o c/tp", shared=("sp1-sp2",), reduce=("sp1-sp2",))
```

Implemented behavior:

- `sync_param_` broadcasts parameter values from group rank 0 over `shared` mesh groups.
- `reduce_grad_` sum-all-reduces `param.grad` over `reduce` mesh groups.
- Compound names work through `wrap_mesh`.
- `shared` metadata is rejected when it overlaps with axis shard dimensions.

Deferred string notation:

```text
[o c] shared(tp-sp1-sp2) reduce(sp1-sp2)
[o/tp c] shared(sp1-sp2) reduce(sp1-sp2)
[o c/tp] shared(sp1-sp2) reduce(sp1-sp2)
```

Remaining work:

- Decide whether bracketed string notation is needed in addition to the Python API.
- Add checkpoint-style shard metadata helpers if checkpoint integration needs them.
- Add module-wide registration helpers if model integration needs them.

## Deferred Cleanup

These are known cleanup items rather than new features.

- Consider whether the grammar should support more than two input tensors.
- Keep README examples aligned with tests as more distributed cases are added.

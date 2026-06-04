# Distributed Pattern Plan

This document tracks remaining distributed-pattern work inspired by `../MachineLearning/SciGPT/scaling-transformers-physical-sciences`.

Implemented coverage includes local einsum-style operations, multi-axis unary split/gather, tensor-parallel linear contraction patterns, distributed contractions over one or more sharded contracted axes, low-level identity/all-reduce and reduce-scatter autograd mappings, `//` partial-value notation, gather-then-split repartition semantics, and `einroll` with correctness-first gather/roll/split behavior.

The remaining work is mostly about broadening notation expressiveness, reducing communication overhead, and deciding whether higher-level model/parameter metadata belongs in this package.

## Factored Axes

Patch expansion and unpatching would be clearer if notation could represent factored logical axes.

Possible notation:

```text
b t h w p q c -> b t c h p w q
```

Open questions:

- How should factored axes map to concrete tensor dimensions?
- Should factors be first-class grammar nodes or only local reshape annotations?
- How should shape metadata be supplied for factored axes?
- How should factored axes compose with sharding, for example patching an axis that is already sharded?

## Further Repartition Optimization

Unary repartition semantics are covered today by gather-then-split, including a single logical axis moving from one mesh dimension to another. Same-mesh axis-to-axis repartition uses a point-to-point all-to-all-style exchange when metadata is available, with gather/split retained as fallback.

Example:

```text
b h/sp1 w c -> b h w/sp1 c
b h/sp1 w c -> b h/sp2 w c
```

Remaining optimization:

- Support multi-axis ownership swaps across spatial mesh dimensions:

```text
b h/sp1 w/sp2 c -> b h/sp2 w/sp1 c
```

- Preserve the current gather-then-split behavior as a correctness fallback for uneven or unsupported cases, with warnings when the fallback may be expensive.

## Compound Groups

SciGPT uses groups such as `sp1-sp2`, `tp-sp1-sp2`, and `dp-sp1-sp2`. Hyphenated names are accepted in sharding and `//` notation, and `wrap_mesh` resolves them as compound groups over PyTorch `DeviceMesh` dimensions.

Axis-wise notation can already describe many operations over separate dimensions:

```text
b h/sp1 w/sp2 c -> b h w c
```

Remaining work:

- Optimize scalar reductions over compound groups instead of reducing listed partial dimensions sequentially.
- Add tests for compound-group reductions and checkpoint-style shard metadata if needed.
- Decide whether `// (sp1,sp2)` remains the user-facing compound syntax or whether named compound groups should also be accepted.

## Parameter Metadata

Parameter sharding is expressible with axis notation, but shared/reduced metadata is not yet represented.

Possible module-level annotations:

```text
[o c] shared(tp-sp1-sp2) reduce(sp1-sp2)
[o/tp c] shared(sp1-sp2) reduce(sp1-sp2)
[o c/tp] shared(sp1-sp2) reduce(sp1-sp2)
```

Needed work:

- Decide whether parameter metadata belongs in this package or a higher-level module layer.
- Define synchronization and gradient-reduction semantics.
- Add tests only after a concrete API is chosen.

## Deferred Cleanup

These are known cleanup items rather than new features.

- Consider whether the grammar should support more than two input tensors.
- Keep README examples aligned with tests as more distributed cases are added.

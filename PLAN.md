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

## Autograd Annotations

Some communication patterns change only backward behavior while the forward tensor layout is unchanged.

Possible notation:

```text
b n c -> b n c :: backward_reduce(tp)
b n c -> b n c / grad:tp
```

Needed work:

- Decide whether autograd-only behavior belongs in `einshard` notation or in named helper functions.
- Add parser support if this becomes notation.
- Connect notation to the existing `identity_forward_allreduce_backward` primitive.
- Define how autograd-only annotations interact with `//` partial outputs.

## Optimized Repartition

Unary repartition semantics are covered today by gather-then-split, including a single logical axis moving from one mesh dimension to another.

Example:

```text
b h/sp1 w c -> b h w/sp1 c
b h/sp1 w c -> b h/sp2 w c
```

Remaining optimization:

- Replace gather-then-split with all-to-all where the source and destination shard dimensions are compatible.
- Support multi-axis ownership swaps across spatial mesh dimensions:

```text
b h/sp1 w/sp2 c -> b h/sp2 w/sp1 c
```

- Preserve the current gather-then-split behavior as a correctness fallback for uneven or unsupported cases.
- Add tests that compare optimized paths against the existing fallback.

## Optimized Distributed Roll

`einroll` currently implements correct semantics by gathering sharded axes, applying `torch.roll`, and splitting back.

Remaining optimization:

- Implement neighbor exchange or all-to-all for sharded roll without materializing the full axis on each rank.
- Preserve the existing `einroll` API and test behavior.
- Add uneven-shard tests once backend support is explicit.
- Decide whether multi-axis sharded rolls should optimize one axis at a time or use a combined exchange.

## Compound Groups

SciGPT uses groups such as `sp1-sp2`, `tp-sp1-sp2`, and `dp-sp1-sp2`.

Axis-wise notation can already describe many operations over separate dimensions:

```text
b h/sp1 w/sp2 c -> b h w c
```

Remaining work:

- Decide how compound group names should be represented in notation.
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

- Decide whether `src/torch_einshard/mesh.py` should be completed or removed in favor of PyTorch `DeviceMesh`.
- Improve parser errors; Parsley failures are currently low-level.
- Consider whether the grammar should support more than two input tensors.
- Keep README examples aligned with tests as more distributed cases are added.

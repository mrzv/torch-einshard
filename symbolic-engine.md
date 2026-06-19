# Symbolic Engine Plan

This document describes a behavior-preserving migration from the current distributed `einshard` dispatcher to a symbolic execution engine. The goal is to derive distributed execution rules from axis placement semantics instead of adding formula-specific branches for each tensor-parallel, data-parallel, or spatial-parallel pattern.

The symbolic engine should not remove explicit primitive logic. The package still needs concrete semantics for split, all-gather, all-reduce, reduce-scatter, same-mesh all-to-all repartition, owner-swap, and rank-local `torch.einsum`. The engine should remove formula-specific hardcoding by composing those primitives from symbolic tensor states.

## Current Baseline

Local formulas already lower generically through `einsum.py`. Distributed formulas are handled in `distributed.py` by two dispatchers:

- `distributed_1d_1` handles unary layout changes, partial-value changes, optimized repartition, owner-swap, fallback gather/split, and final permutation.
- `distributed_1d_2` handles binary normalization, contracted-axis placement, sharded contraction partials, shared output axes, reduce-scatter outputs, post-contraction repartition, owner-swap-compatible crossed contractions, and fallback collectives.

The migration should treat the existing tests as the acceptance harness. The important covered behaviors include:

- Unary split/gather round trips, including ellipsis.
- Multi-axis split/gather.
- Same-mesh axis-to-axis repartition with uneven shards.
- Gather/split fallback when repartition crosses mesh dimensions.
- Multi-axis owner-swap when mesh sizes and split metadata match.
- Partial-to-full, full-to-partial, partial-to-shard, shard-to-partial, scalar partials, and compound mesh partial names.
- Tensor-parallel column and row linear patterns.
- Sharded contractions with explicit partial outputs.
- Cross-sharded contractions that reduce-scatter into output axes.
- Shared output axes, including splitting a replicated operand to match a sharded operand.
- Attention score/value patterns where one free axis is gathered or a contracted axis is split.

## Migration Principles

- Preserve public behavior before improving it.
- Extract and name the current semantics instead of designing a greenfield optimizer first.
- Keep the first planner deterministic and rule-based.
- Model forward behavior and backward behavior together; the primitive choice is an autograd mapping, not only a forward collective.
- Keep `compile_einshard`, cache policy, cost search, and public plan inspection private or deferred until unary and binary parity are established.
- Keep `shapes` optional except where the existing optimized path already needs split metadata.
- Keep distributed factored axes unsupported until they are explicitly designed and tested.

## Desired Model

Each parsed tensor spec should become an internal logical state:

```text
TensorState:
  axes: ordered logical axes
  placements: axis name -> local | shard(mesh_dim)
  partials: ordered mesh dimensions
```

Example:

```text
a b -> a/dp b
```

Input state:

```text
axes = [a, b]
placements = {a: local, b: local}
partials = ()
```

Output state:

```text
axes = [a, b]
placements = {a: shard(dp), b: local}
partials = ()
```

The planner sees that `a` changes from `local` to `shard(dp)` and emits a split-forward/all-gather-backward step over axis `a` and mesh dimension `dp`.

## Primitive Plan Steps

The first implementation should model the actual autograd-paired mappings in `mappings.py`.

### Rank-Local Operations

- `rank_local_einsum`: runs `einsum.py` on each rank after input states have been normalized enough for local slices to be valid.
- `permute`: reorders axes when only axis order differs.
- `reshape_expand_groups`: expands local factored axes before rank-local einsum.
- `reshape_pack_groups`: packs local factored axes after rank-local einsum.

`rank_local_einsum` is intentionally not named `local_einsum`: its inputs can still represent sharded logical tensors, but each rank computes using its current local shards.

### Placement And Repartition Operations

- `split_forward_allgather_backward(axis, mesh_dim)`: forward `local -> shard(mesh_dim)`, backward all-gather.
- `allgather_forward_split_backward(axis, mesh_dim)`: forward `shard(mesh_dim) -> local`, backward split.
- `allgather_forward_reducescatter_backward(axis, mesh_dim)`: forward gather when backward should reduce-scatter because the mesh dimension remains active in output placement or partials.
- `alltoall_repartition(source_axis, dest_axis, mesh_dim)`: repartitions ownership from a source logical axis sharded on `mesh_dim` to a different destination logical axis sharded on the same mesh dimension when split metadata is available.
- `owner_swap(source_mesh_dims, dest_mesh_dims)`: swaps ownership across equal-sized mesh dimensions without materializing full tensors when split metadata matches.

### Partial Operations

- `allreduce_forward_identity_backward(mesh_dim)`: removes a forward partial and leaves backward gradients unchanged.
- `identity_forward_allreduce_backward(mesh_dim)`: introduces or preserves a forward partial and all-reduces in backward.
- `reducescatter_forward_allgather_backward(axis, mesh_dim)`: removes a partial and produces an output shard on the same mesh dimension.
- `allgather_forward_reducescatter_backward(axis, mesh_dim)`: gathers in forward while preserving reduce-scatter behavior in backward.

### Correctness Fallbacks

- `all_gather + split`: fallback for legal single-axis repartitioning when optimized same-mesh all-to-all preconditions are not met. Multi-axis owner-swap failures should continue to reject unless a safe fallback is explicitly designed.
- `all_gather + rank_local_einsum + split`: future fallback for symbolically valid distributed operations that lack an optimized implementation.

Fallbacks should warn when they may materialize larger tensors. The current gather/split repartition warning should be preserved.

## Planning Pipeline

The first implementation should be deterministic and rule-based.

1. Expand axis families using `cached_expand_axis_families` before parsing.
2. Parse with `parse_sharding`.
3. Reject distributed factored-axis operations before symbolic distributed planning, matching current public behavior.
4. Normalize supported ellipses to synthetic local axes where the current dispatcher already does so.
5. Build `TensorState` objects for each input and output.
6. Classify axes:
   - Free axes appear in one input and the output.
   - Shared output axes appear in both inputs and the output.
   - Contracted axes appear in inputs but not output.
   - Output-only axes are invalid unless an explicitly supported operation introduces them.
7. For unary formulas, plan state transitions directly from input state to output state.
8. For binary formulas, normalize input states so rank-local `torch.einsum` can run:
   - Match shared output-axis placement.
   - Align contracted axes to a common placement.
   - Gather or split free axes when needed for output placement.
   - Preserve special ordering for reduce-scatter output axes before changing contracted axes.
9. Run `rank_local_einsum` on normalized states.
10. Attach partials produced by sharded contractions.
11. Transform the rank-local result into the requested output state:
    - Remove or preserve partials according to `//` notation.
    - Prefer reduce-scatter when an output shard can consume a partial.
    - Apply post-contraction all-to-all or gather/split repartition.
    - Permute final axes if necessary.
12. Add cache and cost ranking only after parity is established.

## Core Rules

### Unary Layout Changes

```text
a b -> a/dp b
```

Rule:

```text
local(a) -> shard(a, dp) = split_forward_allgather_backward(a, dp)
```

```text
a/dp b -> a b
```

Rule:

```text
shard(a, dp) -> local(a) = allgather_forward_split_backward(a, dp)
```

```text
a/dp b -> a b // dp
```

Rule:

```text
shard(a, dp) -> local(a) while preserving partial(dp)
= allgather_forward_reducescatter_backward(a, dp)
```

```text
a/dp b -> a b/dp
```

Preferred rule when shape metadata exists:

```text
alltoall_repartition(a, b, dp)
```

Fallback rule:

```text
allgather_forward_split_backward(a, dp)
split_forward_allgather_backward(b, dp)
```

### Partial Values

```text
a b // tp -> a b
```

Rule:

```text
partial(tp) -> full = allreduce_forward_identity_backward(tp)
```

```text
a b -> a b // tp
```

Rule:

```text
full -> partial(tp) = identity_forward_allreduce_backward(tp)
```

```text
a b // tp -> a/tp b
```

Rule:

```text
partial(tp) + local(a) -> shard(a, tp)
= reducescatter_forward_allgather_backward(a, tp)
```

### Binary Contractions

```text
b n c/tp, h c/tp -> b n h
```

Rule:

```text
contracted c is sharded on tp
rank_local_einsum creates partial(tp)
output does not preserve partial(tp)
remove partial(tp) with allreduce_forward_identity_backward(tp)
```

```text
b n c/tp, h c/tp -> b n h // tp
```

Rule:

```text
contracted c is sharded on tp
rank_local_einsum creates partial(tp)
output preserves partial(tp)
no all-reduce in forward
```

```text
l f/tp, f/tp e -> l/tp e
```

Rule:

```text
contracted f creates partial(tp)
output wants l sharded over tp
use reducescatter_forward_allgather_backward(l, tp)
instead of all-reduce then split
```

### Shared Output Axes

```text
b/dp a c, b/dp c d -> b/dp a d
```

Rule:

```text
shared output axis b has matching shard(dp) in both inputs and output
rank_local_einsum is valid on each rank
```

```text
b/dp a c, b c d -> b/dp a d
```

Rule:

```text
shared output axis b is shard(dp) in one input and local in the other
split replicated operand on b over dp before rank_local_einsum
```

### Repartition Optimizations

```text
b h/sp1 w/sp2 c -> b h/sp2 w/sp1 c
```

Rule:

```text
if sp1 and sp2 have equal mesh size and split metadata matches:
  owner_swap((sp1, sp2), (sp2, sp1))
else:
  reject when multiple shard-dimension changes cannot be expressed safely
```

## Cost Model

The symbolic formula determines legal plans, not necessarily the best plan. Cost-based search should be deferred until after behavior parity.

Initial deterministic priorities:

- Prefer rank-local operations over communication.
- Prefer `reducescatter_forward_allgather_backward` over all-reduce plus split when the final output is sharded on the same mesh dimension.
- Prefer same-mesh `alltoall_repartition` over gather/split when split metadata is available.
- Prefer `owner_swap` over gather/split for pure ownership permutation across equal-sized mesh dimensions with matching split metadata.
- Penalize fallbacks that materialize full tensors.

Later cost inputs can include tensor shapes, mesh shape, process group sizes, split metadata, dtype, device, memory budget, and whether forward-only or forward-plus-backward cost should be optimized.

## Plan Cache

Plan caching should be added only after the executable plan representation is stable. A cache key should include enough information to avoid reusing an invalid plan:

- Expanded formula string.
- Mesh dimension names and mesh shape.
- Input ranks and axis order after ellipsis expansion.
- Presence, structure, and resolved values of `shapes` metadata used by optimized repartition, owner-swap, split, gather, and reduce-scatter steps.
- Axis family expansion inputs.
- Possibly dtype/device if the cost model becomes backend-sensitive.

The cached object should be an executable plan, not the final tensor result.

## Error Model

Initial migration should preserve current behavior and broad exception types where practical. After parity, errors should be made more symbolic and explain which rule failed.

Examples:

- Output axis is not present in any input.
- Shared output axis has incompatible input placements.
- Contracted axis has incompatible sharding and no legal normalization path.
- Optimized repartition requires split metadata but none was provided.
- Factored axes are used in a distributed operation before distributed group support exists.

Warnings should be used when the operation is valid but falls back to a more expensive communication path.

## Implementation Strategy

### Phase 0: Behavior Matrix

- Inventory current supported cases from the distributed unary and binary tests.
- Record expected primitive choices for split, gather, fallback gather/split, all-to-all, owner-swap, all-reduce, reduce-scatter, and explicit partial preservation.
- Treat the existing distributed test suite as the migration acceptance harness.

### Phase 1: Pure Symbolic State Helpers

- Add private helpers that convert parsed specs into `TensorState` objects.
- Add axis classification helpers for unary and binary formulas.
- Add placement-delta and partial-delta helpers.
- Keep existing public behavior unchanged.
- Reuse `Axis`, `Axes`, `TensorSpec`, and `partials` structures where practical.
- Add tests for state construction and axis classification without invoking distributed collectives.

### Phase 2: Executable Primitive Plan Vocabulary

- Add private plan-step objects or functions matching the existing autograd mappings.
- Each step should expose symbolic effects and an `execute` path using the current mapping functions.
- Preserve `resolve_split_shapes` behavior and current shape validation.

### Phase 3: Replace Unary Dispatcher Internals

- Reimplement `distributed_1d_1` as a symbolic plan over unary state transitions.
- Preserve current operation ordering and mapping choices:
  1. Remove input partials unless preserved, using reduce-scatter immediately when an output shard can consume the partial.
  2. Prefer owner-swap for compatible multi-axis mesh-dimension changes.
  3. Prefer same-mesh all-to-all for single gather/split repartition.
  4. Warn on gather/split fallback.
  5. Apply remaining split/gather changes, choosing `allgather_forward_reducescatter_backward` during shard-to-partial gathers when the output partial requires it.
  6. Introduce any requested output partials that were not created by a gather mapping.
  7. Permute if necessary.
- Preserve existing split, gather, partial, all-to-all, owner-swap behavior and warnings.

### Phase 4: Replace Binary Dispatcher Internals

- Reimplement `distributed_1d_2` around explicit symbolic stages:
  1. Determine contracted-axis targets.
  2. Determine reduction partials produced by sharded contractions.
  3. Determine reduce-scatter output axes.
  4. Normalize inputs.
  5. Run rank-local einsum.
  6. Resolve contraction partials.
  7. Apply post-repartition.
- Preserve current ordering constraints, especially gathering reduce-scatter output axes before changing contracted axes.
- Preserve tensor-parallel MLP, attention, shared-axis, multi-axis contraction, and owner-swap-compatible crossed-contraction tests.

### Phase 5: Internal Plan Inspection

- Add an internal debugging hook only after real executable plans exist.
- Use it in tests to assert that important patterns choose `alltoall_repartition`, `reducescatter_forward_allgather_backward`, or `owner_swap` instead of a fallback where applicable.
- Keep public `compile_einshard` deferred until the API is stable.

### Phase 6: Cache And Cost-Based Search

- Represent plans as a small graph of tensor-state transitions.
- Generate multiple valid plans for ambiguous cases.
- Rank them with the cost model.
- Keep deterministic tie-breaking for reproducibility.

## Non-Goals For The First Version

- No public `compile_einshard` API until internals are stable.
- No automatic FSDP policy inference.
- No cross-formula fusion.
- No global model-level scheduling.
- No topology-aware collective modeling beyond mesh dimension sizes.
- No distributed support for factored axes unless explicitly designed and tested.
- No attempt to support arbitrary output-only axes.
- No cost-based search before unary and binary parity.

## Open Questions

- Should users eventually be able to forbid expensive fallbacks instead of receiving warnings?
- Should optimized same-mesh repartition require `shapes`, or should equal-shard metadata inferred from local tensors be enough?
- Should cost ranking optimize forward only, backward only, or combined training-step cost?
- How should symbolic planning interact with future operation families such as convolution, halo exchange, FFT, or fused multi-formula execution?

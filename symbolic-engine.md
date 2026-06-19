# Symbolic Engine Plan

This document sketches a symbolic planner for `einshard` formulas. The goal is to derive distributed execution rules from axis placement semantics instead of adding one hand-written branch for each tensor-parallel or spatial-parallel pattern.

The planner should not remove all hardcoded logic. The package still needs explicit primitive semantics for collectives such as split, all-gather, all-reduce, reduce-scatter, all-to-all, owner-swap, and local `torch.einsum`. The planner should remove formula-specific hardcoding by composing those primitives from symbolic tensor states.

## Motivation

Today, local formulas already lower generically through `einsum.py`. Most distributed complexity lives in `distributed.py`, where unary and binary formulas are analyzed by hand and mapped to specific collective sequences.

That approach works for covered cases, but it does not scale well as new patterns are added:

- Tensor-parallel MLP and attention patterns introduce many similar contractions.
- Repartitioning needs several variants: split, gather, gather/split fallback, same-mesh all-to-all, and owner-swap.
- Partial-value notation with `//` interacts with both unary layout changes and binary contractions.
- Axis families, ellipses, and factored axes add notation-level variation without changing the core distributed semantics.

A symbolic engine should make these cases fall out of a small set of general rules.

## Desired Model

Each parsed formula should become an intermediate representation with explicit logical state:

```text
TensorState:
  axes: ordered logical axes
  placements: axis name -> local | shard(mesh_dim)
  partials: set(mesh_dim)
```

Examples:

```text
a b -> a/dp b
```

Input state:

```text
axes = [a, b]
placements = {a: local, b: local}
partials = {}
```

Output state:

```text
axes = [a, b]
placements = {a: shard(dp), b: local}
partials = {}
```

A planner can see that `a` changes from local to `shard(dp)`, so it emits a split over axis `a` and mesh dimension `dp`.

## Primitive Operations

The engine should know a small catalog of primitive transitions. Each primitive needs preconditions, effects, shape requirements, backward mapping, and an estimated cost.

### Local Operations

- `local_einsum`: computes a local einsum from normalized local tensor states.
- `permute`: reorders axes when input and output axis order differ.
- `reshape_expand_groups`: expands factored axes before local einsum.
- `reshape_pack_groups`: packs factored axes after local einsum.

### Placement Operations

- `split(axis, mesh_dim)`: `local -> shard(mesh_dim)`.
- `all_gather(axis, mesh_dim)`: `shard(mesh_dim) -> local`.
- `all_to_all_repartition(source_axis, dest_axis, mesh_dim)`: moves ownership from one axis sharded on a mesh dimension to another axis sharded on the same mesh dimension.
- `owner_swap(source_mesh_dims, dest_mesh_dims)`: swaps ownership across equal-sized mesh dimensions without materializing full tensors.

### Partial Operations

- `all_reduce(mesh_dim)`: removes `partial(mesh_dim)` while preserving layout.
- `identity_with_all_reduce_backward(mesh_dim)`: introduces an output partial in forward and all-reduces in backward.
- `reduce_scatter(axis, mesh_dim)`: removes `partial(mesh_dim)` and produces `shard(mesh_dim)` on an output axis.
- `all_gather_with_reduce_scatter_backward(axis, mesh_dim)`: gathers in forward when backward should reduce-scatter.

### Correctness Fallbacks

- `all_gather + split`: correctness fallback for repartitioning when optimized all-to-all or owner-swap preconditions are not met.
- `all_gather + local_op + split`: correctness fallback for operations that are symbolically valid but lack an optimized distributed implementation.

Fallbacks should warn when they may materialize larger tensors.

## Planning Pipeline

The first implementation should be deterministic and rule-based, not a full optimizer.

1. Expand axis families using `cached_expand_axis_families`.
2. Parse the formula with `parse_sharding`.
3. Normalize notation-level features:
   - Expand supported ellipses to synthetic local axes.
   - Expand factored axes into flat logical axes when the operation is local or when distributed support is intentionally added.
   - Keep `//` partial mesh dimensions separate from axis placements.
4. Build `TensorState` objects for each input and the output.
5. Classify axes:
   - Free axes appear in inputs and output.
   - Contracted axes appear in inputs but not output.
   - Shared output axes appear in both inputs and output.
   - Output-only axes are invalid unless introduced by an explicit supported operation.
6. Normalize input states so local `torch.einsum` can run:
   - Match sharding for shared output axes.
   - Align contracted axes to a common placement.
   - Gather or split operands as needed.
7. Run `local_einsum` on the normalized states.
8. Attach partials produced by sharded contractions.
9. Transform the local result state into the requested output state:
   - Remove or preserve partials according to `//` notation.
   - Split, gather, reduce-scatter, or repartition output axes.
   - Permute final axes if necessary.
10. Cache the compiled plan.

## Core Rules

### Unary Layout Changes

```text
a b -> a/dp b
```

Rule:

```text
local(a) -> shard(a, dp) = split(a, dp)
```

```text
a/dp b -> a b
```

Rule:

```text
shard(a, dp) -> local(a) = all_gather(a, dp)
```

```text
a/dp b -> a b/dp
```

Preferred rule when shape metadata exists:

```text
all_to_all_repartition(a, b, dp)
```

Fallback rule:

```text
all_gather(a, dp)
split(b, dp)
```

### Partial Values

```text
a b // tp -> a b
```

Rule:

```text
partial(tp) -> full = all_reduce(tp)
```

```text
a b -> a b // tp
```

Rule:

```text
full -> partial(tp) = identity forward, all_reduce backward
```

```text
a b // tp -> a/tp b
```

Rule:

```text
partial(tp) + local(a) -> shard(a, tp) = reduce_scatter(a, tp)
```

### Binary Contractions

```text
b n c/tp, h c/tp -> b n h
```

Rule:

```text
contracted c is sharded on tp
local_einsum creates partial(tp)
output does not preserve partial(tp)
remove partial(tp) with all_reduce(tp)
```

```text
b n c/tp, h c/tp -> b n h // tp
```

Rule:

```text
contracted c is sharded on tp
local_einsum creates partial(tp)
output preserves partial(tp)
no all_reduce in forward
```

```text
l f/tp, f/tp e -> l/tp e
```

Rule:

```text
contracted f creates partial(tp)
output wants l sharded over tp
use reduce_scatter(l, tp) instead of all_reduce(tp) then split(l, tp)
```

### Shared Output Axes

```text
b/dp a c, b/dp c d -> b/dp a d
```

Rule:

```text
shared output axis b has matching shard(dp) in both inputs and output
local_einsum is valid on each rank
```

```text
b/dp a c, b c d -> b/dp a d
```

Rule:

```text
shared output axis b is shard(dp) in one input and local in the other
split replicated operand on b over dp before local_einsum
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
  gather/split fallback or reject if fallback is too expensive by policy
```

## Cost Model

The symbolic formula determines legal plans, not necessarily the best plan. A cost model should rank valid plans using runtime metadata.

Inputs:

- Tensor shapes and local shard shapes.
- Mesh shape and mesh dimension names.
- Process group sizes.
- Split metadata from `shapes`.
- Dtype and approximate bytes per element.
- Whether forward-only or forward-plus-backward cost should be optimized.
- Optional memory budget.

Initial costs can be rough:

- Prefer local operations over communication.
- Prefer `reduce_scatter` over `all_reduce + split` when the final output is sharded on the same mesh dimension.
- Prefer same-mesh `all_to_all_repartition` over `all_gather + split` when metadata is available.
- Prefer `owner_swap` over gather/split for pure ownership permutation across equal-sized mesh dimensions.
- Penalize fallbacks that materialize full tensors.

The first version can use deterministic priorities. A later version can search a small plan graph and choose the lowest estimated cost.

## Plan Cache

Compiled plans should be cached. A cache key should include enough information to avoid reusing an invalid plan:

- Expanded formula string.
- Mesh dimension names and mesh shape.
- Input ranks and axis order after ellipsis expansion.
- Presence and structure of `shapes` metadata.
- Axis family expansion inputs.
- Possibly dtype/device if the cost model becomes backend-sensitive.

The cached object should be an executable plan, not just the final tensor result.

## Error Model

Errors should explain which symbolic rule failed.

Examples:

- Output axis is not present in any input.
- Shared output axis has incompatible input placements.
- Contracted axis has incompatible sharding and no legal normalization path.
- Repartition requires split metadata but none was provided.
- Factored axes are used in a distributed operation before distributed group support exists.

Warnings should be used when the operation is valid but falls back to a more expensive communication path.

## Implementation Strategy

### Phase 1: Extract State Helpers

- Add internal helpers that convert parsed specs into `TensorState` objects.
- Keep existing public behavior unchanged.
- Reuse existing `Axis`, `Axes`, `TensorSpec`, and `partials` structures where practical.
- Add tests for state construction and axis classification without invoking distributed collectives.

### Phase 2: Replace Unary Dispatcher Internals

- Reimplement `distributed_1d_1` as a plan over unary state transitions.
- Preserve existing split, gather, partial, all-to-all, and owner-swap behavior.
- Keep current warnings for gather/split fallbacks.

### Phase 3: Replace Binary Dispatcher Internals

- Reimplement `distributed_1d_2` around contraction analysis and output-state transformation.
- Preserve current tensor-parallel MLP, attention, shared-axis, and multi-axis contraction tests.
- Prefer reduce-scatter when a sharded output axis can consume a contraction partial.

### Phase 4: Add Plan Inspection

- Add an internal or public debugging hook to return the symbolic plan for a formula.
- Use this in tests to assert that important patterns choose `all_to_all`, `reduce_scatter`, or `owner_swap` instead of gather/split fallback.

### Phase 5: Cost-Based Search

- Represent plans as a small graph of tensor-state transitions.
- Generate multiple valid plans for ambiguous cases.
- Rank them with the cost model.
- Keep deterministic tie-breaking for reproducibility.

## Non-Goals For The First Version

- No automatic FSDP policy inference.
- No cross-formula fusion.
- No global model-level scheduling.
- No topology-aware collective modeling beyond mesh dimension sizes.
- No distributed support for factored axes unless explicitly designed and tested.
- No attempt to support arbitrary output-only axes.

## Open Questions

- Should the planner be public as `compile_einshard` or remain internal until stable?
- Should users be able to forbid expensive fallbacks instead of receiving warnings?
- Should `shapes` become required for all optimized repartitions?
- Should cost ranking optimize forward only, backward only, or combined training step cost?
- How should symbolic planning interact with future operation families such as convolution, halo exchange, FFT, or fused multi-formula execution?

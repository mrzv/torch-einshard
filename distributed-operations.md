# Distributed Operations For More Flexible Plans

This document lists distributed operations that are missing or only partially implemented today, and explains what additional planning choices they would unlock.

The current implementation has useful primitives for split/allgather/allreduce/reduce-scatter-style transitions, one-axis all-to-all repartition, owner-swap, and roll helpers. The planner would become more flexible with the operations below.

## 1. Native Reduce-Scatter

Current `reduce_scatter` is implemented as `all_reduce` followed by local `split` in `src/torch_einshard/helpers.py`.

That means formulas like this can be modeled as reduce-scatter:

```python
z = es.einshard(
    "l f/tp, f/tp e -> l/tp e",
    x_f_shard,
    w_f_shard,
    mesh=mesh_tp,
    shapes={"tp": {"l": l_shapes, "f": f_shapes}},
)
```

But runtime does not yet get the real communication savings compared to allreduce-then-split.

A native `dist.reduce_scatter_tensor` path, plus an uneven-shape equivalent, would make reduce-scatter plans meaningfully distinct from allreduce-then-split plans.

This would improve policies that prefer:

- lower communication volume
- lower peak output materialization
- cheaper training backward paths for sharded outputs

## 2. General All-To-All / All-To-Allv Repartition

Current `alltoall_repartition` handles one source axis to one destination axis along the same mesh dimension. This supports patterns such as:

```python
z = es.einshard(
    "l/tp e, e f -> l f/tp",
    x_l_shard,
    w,
    mesh=mesh_tp,
    shapes={"tp": {"l": l_shapes, "f": f_shapes}},
)
```

More general all-to-all support would allow direct reshards such as:

```python
z = es.einshard("a/dp b, c d -> a c/sp", x, y, mesh=mesh)
z = es.einshard("a/tp b -> a/sp b", x, mesh=mesh)
```

Useful extensions include:

- repartition between different mesh dimensions
- multiple source and destination axes
- more general uneven all-to-allv shapes beyond the current one-source-axis/one-destination-axis case
- different output ownership choices without materializing full intermediates

This would give `memory`, `communication`, and `inference` policies more direct paths than gather/compute/split fallbacks.

## 3. Broadcast, Scatter, And Explicit Replication

`distributed.py` currently has `augment_parallelism(...): # TODO: add replication`.

Right now, local axes often imply that every rank already has the full tensor. Explicit replication and ownership operations would let the planner choose between broadcasting, scattering, gathering, or reusing replicated operands.

Example:

```python
z = es.einshard(
    "b n c, h/tp c -> b n h/tp",
    x,
    w_shard,
    mesh=mesh_tp,
)
```

With explicit replication support, the planner could decide whether to:

- broadcast or replicate weights
- gather activations
- keep an operand replicated because later formulas reuse it
- scatter replicated inputs into a sharded layout before compute

This is important for cross-formula policy decisions because communication may prefer paying once and reusing a replicated value, while memory may prefer shorter-lived local shards.

## 4. SUMMA-Style Distributed Matmul

Current binary planning mostly normalizes operands, performs a local einsum, then applies output collectives.

A SUMMA-like matmul planner would communicate panels across a 2D mesh and accumulate partial outputs. This is useful for formulas such as:

```python
z = es.einshard(
    "a/dp k, k/sp b -> a/dp b/sp",
    x,
    y,
    mesh=mesh_2d,
)
```

Instead of forcing one operand layout to match the other, the planner could stream panels through the mesh.

This would unlock policy choices around:

- panel size and memory footprint
- communication/computation overlap
- output sharding ownership
- whether to accumulate partials locally or reduce as panels progress

This is one of the biggest missing pieces for flexible distributed matrix multiplication plans.

## 5. Fused Or Coalesced Collectives

For related formula sets, multiple collectives could often be combined into fewer launches.

Example pattern with repeated partial reductions:

```python
y0_partial = es.einshard("b n h/tp, o h/tp -> b n o // tp", h0, w0, mesh=mesh_tp)
y1_partial = es.einshard("b n h/tp, o h/tp -> b n o // tp", h1, w1, mesh=mesh_tp)
y2_partial = es.einshard("b n h/tp, o h/tp -> b n o // tp", h2, w2, mesh=mesh_tp)

y0 = es.einshard("b n o // tp -> b n o", y0_partial, mesh=mesh_tp)
y1 = es.einshard("b n o // tp -> b n o", y1_partial, mesh=mesh_tp)
y2 = es.einshard("b n o // tp -> b n o", y2_partial, mesh=mesh_tp)
```

A latency policy would benefit from coalescing repeated allreduces or reduce-scatters when their communication groups and tensor lifetimes align.

This is less a new tensor semantic and more a scheduling primitive. It would matter most for:

- `latency` policy, by reducing collective launch count
- `communication` policy, when coalescing enables better bandwidth use
- cross-formula fusion, where several formulas share communication structure

## 6. Async Collectives And Overlap

`helpers.py` currently has `# TODO: add async_op option`.

Async versions of allgather, reduce-scatter, allreduce, and all-to-all would let the planner overlap communication with local einsum or other computation.

This would create planning choices where a policy might prefer a higher-byte plan because the communication can be hidden behind compute.

Useful examples include:

- starting an allgather for the next matmul panel while computing the current panel
- overlapping gradient reduce-scatter with backward local einsum
- issuing all-to-all repartition before downstream consumers need the full result

This is especially relevant to `latency` and `training` policies.

## 7. Reduce-Scatter To Different Ownership

Current reduce-scatter plans work naturally when the reduction mesh dimension is also the output shard dimension.

More flexible plans would combine reduction with repartitioning, for example reducing over `tp` but placing the output over `dp` or `sp` without materializing a full intermediate.

Conceptually:

```python
z = es.einshard(
    "a/tp k, k/tp b -> a/dp b",
    x,
    y,
    mesh=mesh,
)
```

This could be implemented as a combination of reduce-scatter and all-to-allv, or as a custom collective schedule.

It would unlock more direct plans for cases where the best reduction dimension and the best output ownership dimension differ.

## Highest-Impact Additions

The highest-impact missing operations are likely:

1. Native efficient `reduce_scatter`.
2. Generalized all-to-all/all-to-allv reshards.
3. Broadcast and explicit replication support.
4. SUMMA-style panel communication for distributed matmul.
5. Coalesced and async collectives for cross-formula policy wins.

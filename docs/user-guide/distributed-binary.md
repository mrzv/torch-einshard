# Distributed Binary Operations

Binary distributed operations support sharded contractions and sharded
elementwise products. For contractions, the local contraction is computed first
and then all-reduced over each contracted shard dimension, unless the output
explicitly keeps the partial value or the contracted mesh dimension is reused to
shard an output axis.

Generic form:

```text
... k/p, k/p ... -> ...
```

Example:

```python
z = es.einshard("a/sp b/dp, b/dp c -> a/sp c", x, y, mesh=mesh)
```

This contracts `b/dp` and all-reduces the output over `dp`.

## Sharded Contractions

Multiple sharded contracted axes are reduced sequentially:

```python
z = es.einshard("a/dp b/sp c, a/dp b/sp d -> c d", x, y, mesh=mesh)
```

Shared sharded axes can appear in both inputs and the output, which covers
distributed batched matmul-style layouts:

```python
z = es.einshard("b/dp a c, b/dp c d -> b/dp a d", x, y, mesh=mesh)
```

If one operand has a replicated shared axis and the output requests the sharded
layout, that operand is split before the local operation:

```python
z = es.einshard("b/dp a c, b c d -> b/dp a d", x_shard, y, mesh=mesh, shapes=shapes)
```

The partial output can be requested explicitly:

```python
z = es.einshard("a/sp b/dp, b/dp c -> a/sp c // dp", x, y, mesh=mesh)
```

In that case, `z` is the local partial contraction result and no forward
all-reduce is applied.

## Elementwise Products

Sharded elementwise products use the same normalization path:

```python
z = es.einshard("l/tp e, l/tp e -> l/tp e", x_shard, y_shard, mesh=mesh, shapes=shapes)
z = es.einshard("l/tp e, l e -> l/tp e", x_shard, y, mesh=mesh, shapes=shapes)
```

## Output Sharding Changes

Binary outputs can request different sharding for free axes than the inputs use.
`einshard` normalizes those input axes before the local contraction with the
necessary split or gather collectives:

```python
z = es.einshard("l/tp e, e f/tp -> l f/tp", x_shard, y_shard, mesh=mesh, shapes=shapes)
z = es.einshard("l/tp e, e f -> l f", x_shard, y, mesh=mesh, shapes=shapes)
z = es.einshard("l e, e f -> l/tp f", x, y, mesh=mesh, shapes=shapes)
```

Contracted axes can also be normalized when one operand has the full contracted
dimension and the other is sharded:

```python
z = es.einshard("l f, f/tp e -> l e", x, y_shard, mesh=mesh, shapes=shapes)
z = es.einshard("l/tp f, f/tp e -> l/tp e", x_shard, y_shard, mesh=mesh, shapes=shapes)
```

For uneven shards, pass split metadata for each affected logical axis and mesh
dimension:

```python
shapes = {"tp": {"l": l_shapes, "f": f_shapes}}
```

## Reduce-Scatter Outputs

If a sharded contraction dimension is reused to shard a free output axis,
`einshard` reduce-scatters the partial contraction result instead of all-reducing
it:

```python
z = es.einshard("l f/tp, f/tp e -> l/tp e", x_shard, y_shard, mesh=mesh, shapes=shapes)
```

This computes the local partial `l e // tp` and then reduce-scatters it to
`l/tp e`.

The same contraction can keep the partial value while still sharding a free
output axis over that mesh dimension:

```python
z = es.einshard("l f/tp, f/tp e -> l/tp e // tp", x_shard, y_shard, mesh=mesh, shapes=shapes)
```

In this form no forward all-reduce is applied. The local partial `l e // tp` is
split to `l/tp e // tp`, so each rank keeps its partial contribution for its
output shard.

## Crossed Layouts

The contracted axis can be sharded over different mesh dimensions in each input
while the reduction mesh dimension is reused for a different output axis:

```python
z = es.einshard(
    "l/sp e/dp, e/sp f/dp -> l/sp f/dp",
    x_shard,
    y_shard,
    mesh=mesh,
    shapes=shapes,
)
```

Here `e` is contracted, `f/dp` is gathered before `e` is normalized, and the
local partial result is reduce-scattered back to `f/dp`.

A free axis can also be local in both inputs and sharded only in the output when
a contracted mesh dimension supplies the reduction:

```python
z = es.einshard(
    "b l e/dp, b e/sp f -> b/dp l f",
    x_shard,
    y_shard,
    mesh=mesh,
    shapes=shapes,
)
```

Multiple crossed contracted axes are supported when their shard-dimension
changes form a pure owner-swap-compatible permutation:

```python
z = es.einshard(
    "a k/dp m/sp, k/sp m/dp b -> a/sp b/dp",
    x_shard,
    y_shard,
    mesh=mesh,
    shapes=shapes,
)
```

This owner-swap path currently requires equal-sized mesh dimensions and matching
split metadata for the swapped axes. Other multi-crossed contracted-axis layouts
raise `NotImplementedError` instead of falling back to unsafe sequential
repartitioning.

When one free axis moves from sharded to local and another moves from local to
sharded over the same mesh dimension, `einshard` can keep the local contraction
in the source-sharded layout and repartition the result with an
all-to-all-style exchange:

```python
z = es.einshard("l/tp e, e f -> l f/tp", x_shard, y, mesh=mesh, shapes=shapes)
```

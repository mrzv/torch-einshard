# Distributed Unary Operations

Unary distributed operations support split, gather, multi-axis split/gather,
optimized ownership swaps, gather-then-split repartition, and partial-value
conversions.

## Split And Gather

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

## Repartition

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

Current repartition semantics are correctness-first: gather the source sharded
axis, then split the destination axis. Same-mesh axis-to-axis repartition uses a
point-to-point all-to-all-style exchange when split metadata is available, and
falls back to gather/split otherwise. Pure multi-axis ownership swaps across
equal-sized mesh dimensions exchange local blocks directly when matching split
metadata is available. Performance-sensitive repartition fallbacks emit
`RuntimeWarning`.

## Partial Values

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

Full-to-partial output keeps the forward value unchanged and all-reduces
gradients in backward:

```python
z = es.einshard("a b -> a b // tp", x, mesh=mesh)
```

Scalar reductions can also use partial notation:

```python
z = es.einshard("loss // (sp,dp) -> loss", loss, mesh=mesh)
```

Partial notation currently represents sum reductions only.

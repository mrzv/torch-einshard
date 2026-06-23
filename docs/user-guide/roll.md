# Distributed Roll

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

Axis families can make dimension-polymorphic roll notation concise.
Family-valued shifts expand to per-axis shifts, and family entries may include
sharding suffixes:

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

Current `einroll` semantics are correctness-first. Sharded axes with explicit
`shapes` use direct point-to-point slice exchange; axes without shape metadata
fall back to gather, `torch.roll`, and split with a `RuntimeWarning`.

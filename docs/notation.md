# Notation

`torch-einshard` expressions describe logical tensor axes and, when needed,
where those axes are sharded.

## Local Axes

Local axes use einsum-style names:

```text
a b, b c -> a c
```

Axis names identify dimensions. Repeated names across inputs are contracted when
they do not appear in the output, and preserved when they do appear in the
output.

## Sharded Axes

Append `/mesh_dim` to shard a logical axis over a mesh dimension:

```text
a/sp b/dp, b/dp c -> a/sp c
```

This means:

- axis `a` is sharded over mesh dimension `sp`.
- axis `b` is sharded over mesh dimension `dp`.
- contraction over `b/dp` produces a partial result that is reduced over `dp`.

Mesh dimension names may include hyphens, for example `tp-sp`, if the supplied
mesh wrapper can resolve that exact compound group.

## Partial Tensors

Tensor-level partial values use `//`:

```text
b n h // tp -> b n h
```

A partial tensor has the full logical shape locally, but each rank holds one
contribution to the value. Converting a partial tensor to a non-partial tensor
sum-reduces over the named mesh dimension.

Multiple partial dimensions are written with parentheses and currently reduced
sequentially in the listed order:

```text
loss // (sp,dp) -> loss
```

Use a compound mesh name, such as `sp-dp`, when a single compound group reduction
is required.

## Factored Axes

Factored axes use one-level parenthesized groups. Grouped input dimensions are
expanded before local `einsum`, and grouped output dimensions are packed after it:

```python
z = es.einshard("b (h p) c -> b h p c", x, sizes={"p": 4})
z = es.einshard("b h p c -> b (h p) c", z)
```

Factor sizes are inferred from tensor dimensions when exactly one factor in a
group is omitted from `sizes`. Supplying no sizes for a group such as `(h p)` is
ambiguous and raises `ValueError`.

## Axis Families

Axis families remove repeated 2D/3D notation boilerplate. `*family` expands to a
sequence of axes, and `[*spatial *window]` zips families into repeated factored
groups:

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

For `dim == 2`, this expands to:

```text
b t (h wh) (w ww) c -> (b h w) t wh ww c
```

For `dim == 3`, this expands to:

```text
b t (h wh) (w ww) (d wd) c -> (b h w d) t wh ww wd c
```

## Ellipses

Einsum ellipses are supported for unnamed local dimensions:

```python
z = es.einshard("... c, c o -> ... o", x, w)
z = es.einshard("... c -> c", x)
```

Distributed unary split/gather and tensor-parallel binary contractions can also
use ellipses for unsharded leading dimensions:

```python
z = es.einshard("... a -> ... a/dp", x, mesh=mesh, shapes=shapes)
z = es.einshard("... c, h/tp c -> ... h/tp", x, w, mesh=mesh)
```

# Local Operations

Local operations are translated to `torch.einsum`.

## Contraction And Permutation

```python
z = es.einshard("a b k c, b c l d -> k l d a", x, y)
```

## Outer Product

```python
z = es.einshard("i, j -> i j", x, y)
```

## Diagonal Extraction

```python
z = es.einshard("i i -> i", x)
```

## Local Axis Permutation

```python
z = es.einshard("b t c h w -> b t h w c", x)
```

## Ellipses

Einsum ellipses are supported for unnamed local dimensions:

```python
z = es.einshard("... c, c o -> ... o", x, w)
z = es.einshard("... c -> c", x)
```

## Factored Axes

Factored axes use one-level parenthesized groups:

```python
z = es.einshard("b (h p) c -> b h p c", x, sizes={"p": 4})
z = es.einshard("b h p c -> b (h p) c", z)
```

Factor sizes are inferred from tensor dimensions when exactly one factor in a
group is omitted from `sizes`.

## Axis Families

Axis families make dimension-polymorphic local expressions concise:

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

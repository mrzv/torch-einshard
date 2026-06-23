# Tensor-Parallel Patterns

## Linear Projections

Column-parallel linear projection is supported by sharding the output feature
axis:

```python
z = es.einshard("b n c, h/tp c -> b n h/tp", x, weight_shard, mesh=mesh)
```

This produces local shards of output features and does not require a forward
all-reduce.

Row-parallel linear projection is supported by sharding the contracted input
feature axis:

```python
z = es.einshard("b n c/tp, h c/tp -> b n h", x_shard, weight_shard, mesh=mesh)
```

This contracts `c/tp` and all-reduces over `tp`.

To keep the local partial output instead of all-reducing immediately:

```python
z = es.einshard("b n c/tp, h c/tp -> b n h // tp", x_shard, weight_shard, mesh=mesh)
```

## MLP Patterns

A two-layer MLP can be expressed as:

```text
b n c, h/tp c -> b n h/tp
b n h/tp, c h/tp -> b n c
```

With sequence-parallel activations, the MLP keeps the sequence axis sharded at
the block boundary:

```text
l/tp e, e f/tp -> l f/tp
l f/tp, f/tp e -> l/tp e
```

The first projection gathers `l/tp` to full `l` while producing the sharded
hidden axis `f/tp`. The second projection contracts over `f/tp` and
reduce-scatters the result back to `l/tp`. For uneven shards, provide split
metadata for both logical axes:

```python
shapes = {"tp": {"l": l_shapes, "f": f_shapes}}
```

## Attention Patterns

Attention projections follow the same pattern:

```text
b l c, q/tp c -> b l q/tp
b l c, k/tp c -> b l k/tp
b l c, v/tp c -> b l v/tp
b l v/tp, c v/tp -> b l c
```

Sequence-parallel attention score and value contractions are also expressible:

```text
b h q/tp d, b h k/tp d -> b h q/tp k
b h q/tp k, b h k/tp d -> b h q/tp d
```

The score pattern gathers the sharded key axis so each query shard can attend
over full `k`. The value pattern splits the contracted `k` axis to match `v`,
computes a local partial over full `q`, and reduce-scatters back to `q/tp`. For
uneven sequence shards, provide both query and key split metadata:

```python
shapes = {"tp": {"q": q_shapes, "k": k_shapes}}
```

# Low-Level Autograd Mappings

The package exposes custom autograd mappings in `torch_einshard.mappings`.

All-reduce in forward, identity in backward:

```python
allreduce_forward_identity_backward(x, comm)
```

Identity in forward, all-reduce in backward:

```python
identity_forward_allreduce_backward(x, comm)
```

All-gather in forward, split in backward:

```python
allgather_forward_split_backward(x, comm, dim, shapes)
```

Split in forward, all-gather in backward:

```python
split_forward_allgather_backward(x, comm, dim, shapes)
```

Reduce-scatter in forward, all-gather in backward:

```python
reducescatter_forward_allgather_backward(x, comm, dim, shapes)
```

All-gather in forward, reduce-scatter in backward:

```python
allgather_forward_reducescatter_backward(x, comm, dim, shapes)
```

The current `reduce_scatter` helper uses native `dist.reduce_scatter_tensor` for
equal split sizes and falls back to all-reduce followed by split for uneven split
sizes. Uneven `all_gather` uses a padded equal-size gather internally for
backends that reject variable-size gathers.

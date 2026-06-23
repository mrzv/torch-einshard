# Mesh And Shape Metadata

Distributed operations use PyTorch `DeviceMesh` objects.

```python
from torch.distributed.device_mesh import init_device_mesh

mesh = init_device_mesh("cpu", (2, 4), mesh_dim_names=("dp", "sp"))
z = es.einshard("a/sp b/dp, b/dp c -> a/sp c", x, y, mesh=mesh)
```

## Compound Mesh Groups

Compound mesh groups can be enabled by wrapping a PyTorch `DeviceMesh`:

```python
mesh = es.wrap_mesh(mesh)
z = es.einshard("loss // dp-sp -> loss", loss, mesh=mesh)
```

Compound names such as `dp-sp` span the listed mesh dimensions while preserving
any remaining mesh coordinates. Compound process groups are created lazily on
first lookup and cached on the wrapped mesh. Reuse the wrapped mesh instance
instead of calling `wrap_mesh` repeatedly; equivalent names such as `dp-sp` and
`sp-dp` share the same cached group.

## Shape Metadata

`shapes` controls split sizes. If omitted, split sizes are computed with
`compute_split_shapes` where possible.

Single mesh dimension:

```python
shapes = [4, 4, 4, 4]
```

Multiple mesh dimensions:

```python
shapes = {
    "sp": [4, 4],
    "dp": [8, 8, 8, 8],
}
```

When a dict form is supplied, missing mesh dimensions or missing axis-specific
entries are reported as `ValueError`s before the collective runs.

Same mesh dimension used for different logical axes:

```python
shapes = {
    "dp": {
        "a": a_shapes,
        "b": b_shapes,
    }
}
```

Factor-aware split sizes preserve divisibility for patch/factor operations
before assigning any remainder to the final shard:

```python
shapes = es.helpers.compute_split_shapes_for_factors(size=721, num_chunks=4, factor=4)
# [180, 180, 180, 181]
```

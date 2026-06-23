# Parameter Metadata

`ParamSpec` describes persistent parameter layout plus synchronization and
gradient-reduction metadata. It does not replace `einshard`; tensor computations
still use explicit `einshard(...)` expressions.

```python
mesh = es.wrap_mesh(mesh)
weight_spec = es.ParamSpec(
    "out/tp in",
    shared=("sp1-sp2",),
    reduce=("sp1-sp2",),
)

weight = torch.nn.Parameter(local_weight)
es.sync_param_(weight, weight_spec, mesh)

y = es.einshard(
    "batch in, out/tp in -> batch out/tp",
    x,
    weight,
    mesh=mesh,
    shapes={"tp": {"out": out_shapes}},
)
loss = y.sum()
loss.backward()
es.reduce_grad_(weight, weight_spec, mesh)
```

`shared` uses broadcast from group rank 0 to make replicated parameter values
identical. `reduce` uses sum all-reduce on `param.grad`. Compound names such as
`sp1-sp2` work with `wrap_mesh` and reuse the wrapped mesh's cached compound
process groups.

`shared` dimensions cannot overlap with axis shard dimensions in the layout. For
example, `ParamSpec("out/tp in", shared="tp")` is rejected because `out` is
already sharded over `tp`.

## Module Helpers

For modules, attach specs to parameters and use the module-level helpers:

```python
es.set_param_spec(weight, weight_spec)
for name, param, spec in es.iter_param_specs(module):
    print(name, spec.layout)
es.sync_module_params_(module, mesh)
es.reduce_module_grads_(module, mesh)
```

## DDP Communication Hook

When using PyTorch DDP, SciGPT-style extra gradient reductions can be registered
as a DDP communication hook:

```python
ddp = torch.nn.parallel.DistributedDataParallel(module, process_group=mesh["dp"].get_group())
es.register_grad_reduction_hook_(ddp, mesh, ddp_group="dp")
```

The hook performs DDP-style averaging over `ddp_group`, then applies sum
all-reduces for attached `ParamSpec.reduce` groups.

For buckets where every parameter has the same extra reduction metadata, the
hook can combine DDP averaging and the extra reduction into one all-reduce over a
compound group:

```python
es.register_grad_reduction_hook_(
    ddp,
    mesh,
    ddp_group="dp",
    combined_reduce_group="dp-sp1-sp2",
    combined_reduce="sp1-sp2",
)
```

This is equivalent to summing the bucket over `dp-sp1-sp2` and then dividing by
the `dp` group size. It avoids a separate `dp` all-reduce followed by an
`sp1-sp2` all-reduce. The fast path is used only when all parameters in a bucket
have exactly `reduce=("sp1-sp2",)`. Mixed buckets fall back to the
per-parameter-spec reductions.

## Shard Metadata

Parameter specs can derive checkpoint/test-copy shard metadata:

```python
metadata = es.param_shard_metadata(weight_spec, global_shape=(1024, 2048), mesh=mesh)
local_slices = metadata.local_slices
local_shape = metadata.local_shape
```

Use `es.param_local_slices(...)`, `es.param_local_shape(...)`, or
`es.param_shard_dims(...)` when only one piece of metadata is needed. These
helpers support single mesh dimensions and compound names such as `dp-sp` via
`wrap_mesh`. Factor-aware splits can be requested for patch-like cases:

```python
local_slices = es.param_local_slices(
    es.ParamSpec("height/sp1 width/sp2"),
    global_shape=(721, 1440),
    mesh=mesh,
    factors={"height": 4, "width": 4},
)
```

Shard metadata helpers require an initialized process group and report missing
mesh dimensions or raw `DeviceMesh` compound-group use as explicit errors.
Sharded factored-axis groups are intentionally rejected until factored parameter
metadata is represented directly.

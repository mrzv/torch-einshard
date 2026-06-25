# Parameter Metadata

Parameter metadata describes persistent parameter layout plus synchronization and
gradient-reduction obligations. `ParamSpec` is the compatibility API for writing
that metadata explicitly. Annotated `einshard(...)` operands can also register
`ParameterState` metadata from the formula itself.

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

Concrete native or DDP-backed gradient reductions also cannot overlap parameter
layout shard dimensions. For example, `ParamSpec("out/tp in", reduce="tp")` is
rejected because the native execution paths all-reduce full local gradient
tensors and would mix different `out` shards. External gradient backends are not
validated by this rule because they own their execution semantics.

## Formula Annotations

Input operands can be annotated as persistent parameters:

```python
weight = torch.nn.Parameter(torch.ones(3))

y = es.einshard(
    "batch/dp channel, channel [param, grad=async] -> batch/dp channel",
    x,
    weight,
    mesh=mesh,
)
state = es.get_parameter_state(weight)
```

The annotated tensor argument must be a `torch.nn.Parameter`. `einshard` validates
parameter metadata before dispatch, executes the tensor operation, and attaches
the resulting `ParameterState` only after the operation succeeds. Failed shape,
mesh, or metadata checks do not leave partial formula state attached.

Supported annotation forms include:

```text
[param]
[param, grad=async]
[param, grad=sp1-sp2]
[param, grad=sp1-sp2:async]
[param, grad=ddp]
[param, grad=external]
[param, grad=dp:ddp]
[param, grad=dp:external]
[param, grad=none]
[param, init_sync=none]
[param, init_sync=external]
[param, init_sync=tp]
```

For local formulas, inferred native gradient obligations are recorded in
`state.grad_comm`. In the example above, the parameter gradient is reduced over
`batch`, and `batch` is sharded over `dp`, so the state records a native async
gradient obligation over `dp`.

Distributed formulas are more conservative. If the operation plan may already
perform backward communication for the annotated operand, `grad=async` remains a
pending inferred obligation instead of guessing an extra reduction. Calling
`reduce_grad_`, `reduce_module_grads_`, or the DDP hook with a pending native/DDP
obligation raises an error until a later planner-aware inference step or an
explicit override resolves it.

Explicit mesh-dim annotations such as `grad=sp1-sp2`, `grad=dp:ddp`, and
`grad=dp:external` are concrete. Legacy `ParamSpec.reduce` metadata is also
concrete native metadata. Bare forms such as `grad=async`, `grad=ddp`, and
`grad=external` request inference; distributed formulas can leave those
obligations pending until planner-aware inference resolves them. `grad=external`
records that another system owns the obligation, and native reduction helpers
skip it. `grad=none` is an explicit opt-out and conflicts with later non-none
formula metadata for the same parameter.

Initialization sync is inferred from managed mesh dimension names when a mesh is
available: dimensions used by the parameter layout are excluded, and remaining
managed dimensions become `state.shared`. Use `init_sync=none`,
`init_sync=external`, or an explicit mesh group to override this inference.

## Module Helpers

For modules, attach specs or states to parameters and use the module-level
helpers:

```python
es.set_param_spec(weight, weight_spec)

state = es.ParameterState.from_param_spec(weight_spec)
es.set_parameter_state(weight, state)

for name, param, spec in es.iter_param_specs(module):
    print(name, spec.layout)
for name, param, state in es.iter_parameter_states(module):
    print(name, state.layout_shard_dims)
es.sync_module_params_(module, mesh)
es.reduce_module_grads_(module, mesh)
```

`set_param_spec` attaches both the legacy spec and the equivalent
`ParameterState`. Formula annotations merge with layout-only or semantically
compatible `ParamSpec` metadata and reject incompatible layouts, conflicting
explicit opt-outs, or conflicting gradient/init-sync obligations.

## Native Gradient Hooks

For non-DDP training loops, concrete native gradient obligations can be executed
with per-parameter autograd hooks:

```python
handle = es.register_native_grad_reduction_hooks_(module, mesh)

loss.backward()
handle.wait()
optimizer.step()
handle.remove()
```

The hook path executes concrete native obligations such as `ParamSpec.reduce` or
`grad=sp1-sp2`. It skips external and DDP-backed obligations, and it rejects
pending native obligations because the correct mesh dimensions have not been
finalized yet.

Today these hooks reduce each incoming gradient contribution synchronously, which
keeps repeated `backward()` gradient accumulation correct. The returned handle is
used to remove hooks and is reserved for future asynchronous work; `wait()` is a
safe no-op when no async work has been launched.

All registered parameters must participate in backward in the same order on every
rank. Use PyTorch DDP, the DDP communication hook below, or `grad=external` for
models with rank-dependent control flow, unused parameters, or more complex
bucket scheduling requirements.

## DDP Communication Hook

When using PyTorch DDP, SciGPT-style extra gradient reductions can be registered
as a DDP communication hook:

```python
ddp = torch.nn.parallel.DistributedDataParallel(module, process_group=mesh["dp"].get_group())
es.register_grad_reduction_hook_(ddp, mesh, ddp_group="dp")
```

The hook performs DDP-style averaging over `ddp_group`, then applies sum
all-reduces for concrete native attached gradient obligations. Legacy
`ParamSpec.reduce` groups are represented as native `ParameterState.grad_comm`
metadata. Pending inferred obligations, DDP-backed obligations, and external
obligations are not executed by the native extra-reduction path.

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
have compatible native reductions such as `reduce=("sp1-sp2",)` or an equivalent
`ParameterState.grad_comm`. Mixed buckets fall back to per-parameter native
reductions.

## Shard Metadata

Parameter specs and states can derive checkpoint/test-copy shard metadata:

```python
metadata = es.param_shard_metadata(weight_spec, global_shape=(1024, 2048), mesh=mesh)
metadata = es.param_shard_metadata(state, global_shape=(1024, 2048), mesh=mesh)
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

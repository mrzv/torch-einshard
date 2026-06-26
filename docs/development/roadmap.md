# Roadmap

This document tracks remaining distributed-pattern work inspired by `../MachineLearning/SciGPT/scaling-transformers-physical-sciences`.

Implemented coverage includes local einsum-style operations, einsum ellipses for local and supported distributed patterns, local one-level factored axes, axis-family notation expansion, multi-axis unary split/gather, tensor-parallel linear contraction patterns, distributed contractions over one or more sharded contracted axes, low-level identity/all-reduce and reduce-scatter autograd mappings, `//` partial-value notation, gather-then-split repartition semantics, optimized same-mesh repartition, pure multi-axis ownership swaps, compound mesh groups, and `einroll` with correctness-first gather/roll/split behavior.

The remaining work is mostly about broadening notation expressiveness, reducing communication overhead, and deciding whether higher-level model/parameter metadata belongs in this package.

## Factored Axes

Patch expansion and unpatching use one-level parenthesized groups to represent factored logical axes.

Implemented notation:

```text
b t (h p) (w q) c -> b t c h p w q
b t h p w q c -> b t (h p) (w q) c
```

Implemented behavior:

- Factor groups are first-class parser nodes but lower to local reshape operations around `torch.einsum`.
- Factor sizes are supplied with the public `sizes` argument.
- Exactly one omitted factor size per group can be inferred from the concrete tensor dimension.
- Sharded annotations inside a group are allowed when the operation remains local, for example `(h/sp p) -> h/sp p`.

Remaining work:

- Decide whether nested groups are needed.
- Decide whether nonlocal distributed operations should allow grouped axes beyond local reshape-only transformations.

## Ellipses

Einsum-style `...` notation represents unnamed local dimensions.

Implemented behavior:

- Local einsum operations pass ellipses through to `torch.einsum`.
- Distributed unary split/gather/repartition can use ellipses when input and output both include them.
- Distributed binary tensor-parallel contraction patterns ignore ellipses in sharding metadata and let local `torch.einsum` handle the unnamed dimensions.

Remaining work:

- Decide whether distributed unary operations should support reducing or introducing ellipsis dimensions, for example `... c -> c` with sharded named axes.

## Axis Families

Axis-family expansion removes repeated 2D/3D notation boilerplate before parsing.

Implemented notation:

```python
es.einshard(
    "b t [*spatial *window] c -> (b *spatial) t *window c",
    x,
    families={"spatial": ("h", "w", "d"), "window": ("wh", "ww", "wd")},
    sizes={"window": window_size},
)
```

Implemented behavior:

- `*family` expands to a sequence of axis names.
- `[*a *b]` zips equal-length families into repeated factored groups, for example `(h wh) (w ww)`.
- `sizes={"family": values}` expands to per-axis sizes for the family members.
- `einroll` also accepts `families` and expands family-valued `shifts` entries to per-axis shifts.

Remaining work:

- Decide whether sharded axis-family syntax is needed, rather than expanding to explicit sharded axes before calling `einshard`.

## Further Repartition Optimization

Unary repartition semantics are covered today by gather-then-split, including a single logical axis moving from one mesh dimension to another. Same-mesh axis-to-axis repartition uses a point-to-point all-to-all-style exchange when metadata is available, with gather/split retained as fallback. Pure multi-axis ownership swaps across equal-sized mesh dimensions directly exchange local blocks when matching split metadata is available.

Example:

```text
b h/sp1 w c -> b h w/sp1 c
b h/sp1 w c -> b h/sp2 w c
```

Implemented ownership swap:

- Multi-axis ownership swaps across spatial mesh dimensions:

```text
b h/sp1 w/sp2 c -> b h/sp2 w/sp1 c
```

Remaining optimization:

- Support more general multi-axis repartition where changed mesh dimensions are not a pure permutation or split metadata differs between source and destination ownership.
- Preserve the current gather-then-split behavior as a correctness fallback for uneven or unsupported cases, with warnings when the fallback may be expensive.

## Compound Groups

SciGPT uses groups such as `sp1-sp2`, `tp-sp1-sp2`, and `dp-sp1-sp2`. Hyphenated names are accepted in sharding and `//` notation, and `wrap_mesh` resolves them as lazily cached compound groups over PyTorch `DeviceMesh` dimensions.

Axis-wise notation can already describe many operations over separate dimensions:

```text
b h/sp1 w/sp2 c -> b h w c
```

Implemented behavior:

- `// (sp1,sp2)` reduces sequentially in the listed order.
- `// sp1-sp2` reduces over one compound group when the mesh is wrapped with `wrap_mesh`.
- Equivalent compound names such as `sp1-sp2` and `sp2-sp1` share a cached process group on the wrapped mesh.

Implemented behavior:

- Parameter shard metadata helpers support compound groups through `wrap_mesh`.

## Parameter Metadata

Parameter sharding is expressible with axis notation. `ParameterState` is the
metadata API for persistent parameter layout plus initialization-sync and
gradient-communication obligations. Annotated formula operands can register
`ParameterState` metadata directly from `einshard` formulas, and explicit
registration helpers cover parameters hidden inside modules.

The migration path is described in [Parameter Inference Plan](parameter-inference.md).
The former explicit spec wrapper has been removed; new code should use formula
annotations, `ParameterState.from_layout`, or the registration helpers below.

Implemented Python API examples:

```python
es.ParameterState.from_layout("o c", init_sync="tp-sp1-sp2", grad="sp1-sp2")
es.register_parameter_layout(weight, "o/tp c", mesh=mesh, grad="sp1-sp2")
es.register_linear_parameters_(linear, weight_layout="o c/tp", mesh=mesh, weight_grad="sp1-sp2")
```

Implemented behavior:

- `sync_param_` broadcasts parameter values from group rank 0 over `shared` mesh groups.
- `reduce_grad_` sum-all-reduces `param.grad` over concrete native gradient obligations.
- `sync_module_params_` and `reduce_module_grads_` apply attached states over a whole module.
- `iter_parameter_states` yields attached `(name, param, state)` triples for state-aware helpers.
- `validate_module_parameter_states_` preflights attached parameter metadata, including rank consistency, concrete mesh-group existence, and pending native/DDP gradient obligations.
- `finalize_parameter_grad_comm_` and `finalize_module_parameter_grad_comm_` resolve pending parameter-gradient obligations with explicit concrete mesh-group policies.
- `register_native_grad_reduction_hooks_` executes concrete native gradient obligations with per-parameter autograd hooks for non-DDP training loops.
- `register_grad_reduction_hook_` adds DDP-style averaging plus extra concrete native or DDP-backed reductions as a DDP communication hook.
- `register_grad_reduction_hook_` can optionally combine DDP averaging and a uniform extra reduction into one compound-group all-reduce for matching buckets.
- Compound names work through `wrap_mesh`.
- `shared` metadata is rejected when it overlaps with axis shard dimensions.
- Concrete native/DDP gradient reductions are rejected when they overlap with parameter layout shard dimensions.
- `param_local_slices`, `param_local_shape`, and `param_shard_metadata` derive local shard metadata from `ParameterState` layouts.
- Input operand annotations such as `[param]`, `[param, grad=async]`, `[param, grad=none]`, and `[param, init_sync=none]` are parsed and stored in `TensorSpec`.
- `einshard` registers annotated `torch.nn.Parameter` operands after successful execution, merges compatible formula metadata with layout-only or semantically compatible `ParameterState` metadata, and rejects conflicting layouts or conflicting explicit opt-outs.
- Local formula uses infer visible native gradient obligations; distributed inferred obligations remain pending until planner-aware inference can prove the correct execution behavior.
- `ParameterState.from_layout`, `register_parameter_layout`, `register_module_parameter_layouts_`, `register_linear_parameters_`, `register_conv_parameters_`, and `register_norm_parameters_` provide explicit state registration for hidden parameters, fused/custom modules, and common linear/conv/norm-style modules that cannot expose parameters as formula operands.
- SciGPT-style tensor-parallel MLP and attention projection patterns are covered by tests using explicit `einshard` calls, and MLP/norm/spatial-position parameter metadata patterns are covered by registration-helper tests.

Deferred string notation:

```text
[o c] shared(tp-sp1-sp2) reduce(sp1-sp2)
[o/tp c] shared(sp1-sp2) reduce(sp1-sp2)
[o c/tp] shared(sp1-sp2) reduce(sp1-sp2)
```

Remaining work:

- Add planner-aware distributed backward inference so pending annotated parameter obligations can be finalized automatically when the operation plan proves the required communication.
- Add execution backends for native async parameter-gradient reductions and automatic DDP-backed obligation finalization.
- Add bucketed/native scheduling for unused-parameter or rank-dependent-control-flow cases; the current native hook path requires identical backward participation and hook order across ranks.
- Extend higher-level module/layer wrappers beyond the current generic and linear/conv/norm registration foundation where specialized validation is useful.
- Continue removing downstream uses of the former explicit spec API in favor of `ParameterState` registration.

## Deferred Cleanup

These are known cleanup items rather than new features.

- Consider whether the grammar should support more than two input tensors.
- Keep README examples aligned with tests as more distributed cases are added.

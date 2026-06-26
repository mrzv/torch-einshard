# Parameter Inference Plan

This document describes a planned extension of the symbolic engine that makes
`ParamSpec` unnecessary as a user-facing API. The goal is not to remove
parameter metadata. The goal is to infer and store that metadata from symbolic
parameter uses instead of asking users to spell it out separately.

The intended end state is:

```text
ParamSpec(layout, shared=..., reduce=...)
```

is replaced by an internal state derived from annotated formulas, symbolic
backward analysis, and module wrappers:

```text
ParameterState(layout, init_sync, grad_comm, shard_metadata)
```

`ParamSpec` can then become a temporary compatibility wrapper and eventually be
removed once all supported parameter cases have migrated.

## Summary

This design can eliminate `ParamSpec` from the primary API if the symbolic
engine becomes parameter-aware.

The symbolic engine must learn three things that are outside the current
operation-local `TensorState` model:

- Which operands are persistent model parameters.
- Which parameter identity a symbolic operand refers to across calls.
- Which communication obligations are required for initialization, gradients,
  and tooling.

The core user-facing addition is an operand annotation:

```text
... c, out/tp c [param, grad=async] -> ... out/tp
```

The annotation says that the second operand is a persistent parameter and that
the engine should infer the gradient communication obligation for that operand,
scheduling the inferred communication asynchronously when possible.

Bare scheduling annotations such as `[async]` are intentionally not part of the
design. `async` modifies a concrete semantic obligation such as `grad`; it does
not stand alone.

## Current Implementation Status

The initial foundation is implemented.

- Input operand annotations are parsed and stored on `TensorSpec`; output
  annotations and standalone scheduling annotations such as `[async]` are
  rejected.
- `ParameterState`, `ParameterInitSync`, and `ParameterGradComm` exist, and
  legacy `ParamSpec` metadata is mirrored into compatible `ParameterState`
  objects.
- Parameter helpers, module sync/reduce helpers, shard metadata helpers, and the
  DDP communication hook consume `ParameterState` while preserving `ParamSpec`
  compatibility.
- `einshard` registers annotated `torch.nn.Parameter` operands after successful
  operation execution and validates metadata conflicts before dispatch.
- Local formula uses can infer visible native gradient obligations from sharded
  axes that are reduced while forming the parameter gradient.
- Distributed formula uses remain conservative: inferred native/DDP gradient
  obligations stay pending unless the user provides an explicit concrete
  override. Pending native/DDP obligations intentionally fail in reduction
  helpers until planner-aware distributed backward inference resolves them.
- Concrete native obligations can be executed with per-parameter native autograd
  hooks for non-DDP training loops. This path reduces each incoming gradient
  contribution synchronously and requires identical backward participation and
  hook order across ranks.
- Hidden linear-, conv-, and norm-style parameters can be registered explicitly with
  `ParameterState.from_layout`, `register_parameter_layout`,
  `register_linear_parameters_`, `register_conv_parameters_`, or
  `register_norm_parameters_`; this covers state attachment without inferring
  arbitrary module internals.

The remaining major gap is execution-layer and planner-aware distributed
inference work. The current implementation records obligations and preserves
unsafe cases as pending metadata; it does not yet launch native async gradient
communication or execute DDP-backed obligations from formula annotations.

## Current `ParamSpec` Responsibilities

`ParamSpec` currently combines four responsibilities.

First, it stores persistent parameter layout:

```python
es.ParamSpec("out/tp in")
es.ParamSpec("out in/tp")
es.ParamSpec("h/sp1 w/sp2 c")
```

This responsibility is already close to `TensorSpec` and `TensorState`.

Second, it stores initialization or value-synchronization metadata through
`shared`. This tells `sync_param_` which mesh groups should receive identical
initial values.

Third, it stores gradient communication metadata through `reduce`. This tells
`reduce_grad_` and the DDP communication hook which extra parameter-gradient
reductions are needed beyond the ordinary data-parallel reduction.

Fourth, it acts as the attached metadata source for non-forward tooling in this
package and in downstream integrations:

- DDP communication hooks and combined-reduction fast paths.
- Local shard slices, local shapes, and checkpoint shard metadata.
- Downstream group validation for mesh dimensions named by parameter metadata.
- Downstream global parameter and gradient norm accounting.
- Tests that materialize full reference weights or reconstruct distributed
  gradients.

A replacement for `ParamSpec` must cover all four responsibilities.

## Design Goals

- Make parameter layout use the same canonical representation as symbolic tensor
  planning.
- Infer initialization sync dimensions from parameter layout whenever possible.
- Infer parameter-gradient communication from symbolic backward analysis instead
  of parameter layout alone.
- Preserve explicit overrides for ambiguous, external, or intentionally skipped
  communication.
- Support current SciGPT-style tensor-parallel, spatial-parallel, and combined
  DDP cases.
- Keep the first implementation rule-based and diagnostic-heavy rather than
  trying to infer through arbitrary Python or PyTorch module internals.

## Non-Goals

- Do not infer hidden parameters from arbitrary `nn.Module` internals without a
  wrapper or explicit annotation API.
- Do not make layout alone determine gradient reductions. Identical layouts can
  require different gradient reductions in different formulas.
- Do not use `[async]` as a standalone annotation.
- Do not remove the need for internal parameter metadata. Only remove the need
  for users to write `ParamSpec` in supported cases.

## Formula Annotations

Operand annotations are attached to input operands in an `einshard` formula.

Common parameter form:

```text
operand [param, grad=async]
```

Meaning:

- The operand is persistent model state.
- The operand's tensor spec is the parameter layout.
- The engine should infer the parameter-gradient communication obligation.
- The inferred gradient communication should be scheduled asynchronously when
  possible.

Supported explicit forms:

```text
[param]
[param, grad=async]
[param, grad=sp1-sp2]
[param, grad=sp1-sp2:async]
[param, grad=tp-sp1-sp2:async]
[param, grad=dp:ddp]
[param, grad=dp:external]
[param, grad=ddp]
[param, grad=none]
[param, grad=external]
[param, init_sync=none]
[param, init_sync=tp]
[param, init_sync=sp1-sp2]
```

`grad` is operand-scoped. For a `[param]` operand, it means the gradient of that
parameter. This avoids the narrower term `wgrad`, which does not fit biases,
norm weights, positional embeddings, or future non-parameter gradient
annotations.

`grad` intentionally names the obligation rather than the collective. The engine
may satisfy that obligation with an all-reduce, reduce-scatter, DDP hook,
bucketed reduction, fused reduction, optimizer integration, or a future strategy.

The suffix after a mesh group names either a scheduling policy or an execution
backend. `:async` means `torch-einshard` owns the obligation and should schedule
it asynchronously when possible. `:ddp` means `torch-einshard` should configure a
PyTorch DDP-backed execution path for that obligation. `:external` means another
system owns the obligation and `torch-einshard` should not execute it.

### Annotation Grammar Sketch

The exact parser implementation can vary, but the semantic grammar should be
close to:

```text
annotation     := "[" annotation_item ("," annotation_item)* "]"
annotation_item := "param"
                 | "grad=" grad_policy
                 | "init_sync=" init_sync_policy
grad_policy    := "async"
                 | mesh_group
                 | mesh_group ":async"
                 | mesh_group ":ddp"
                 | mesh_group ":external"
                 | "ddp"
                 | "none"
                 | "external"
init_sync_policy := mesh_group | "none" | "external"
```

The parser should reject annotations that ask for scheduling without naming an
obligation, for example `[async]`.

## `ParameterState`

The symbolic engine should maintain a parameter registry keyed by
`torch.nn.Parameter` identity. The registry can use weak references so metadata
does not keep parameters alive after modules are destroyed.

A `ParameterState` should contain at least:

- The parameter identity and optional debug name.
- The canonical parameter `TensorSpec`.
- The canonical parameter `TensorState` for a managed mesh.
- The parameter's global shape and local shape metadata when known.
- The mesh dimensions used to shard parameter axes.
- The managed mesh dimensions over which the parameter is replicated.
- The inferred or overridden initialization sync groups.
- The inferred or overridden gradient communication obligation.
- The gradient execution backend, such as native, PyTorch DDP, external, or none.
- The requested gradient scheduling policy, such as synchronous or async.
- The source formula locations or call sites that contributed to the state.
- A flag for each field indicating whether it was inferred or explicitly
  overridden.

The registry should reject conflicting uses of the same parameter unless the
conflict is explicitly resolved by an override. For example, the same parameter
cannot be observed with two incompatible layouts.

## Initialization Sync Inference

For a managed model-parallel mesh, initialization sync is usually inferable from
layout:

```text
init_sync_dims = managed_model_mesh_dims - parameter_layout_shard_dims
```

Examples:

```text
out/tp in          -> init sync over sp1-sp2
out in/tp          -> init sync over sp1-sp2
out                -> init sync over tp-sp1-sp2
c                  -> init sync over tp-sp1-sp2
h/sp1 w/sp2 c      -> init sync over tp
out in kh kw       -> init sync over tp-sp1-sp2
```

The set of `managed_model_mesh_dims` is an integration policy, not necessarily
every dimension in the process mesh. In a DDP integration, `dp` is normally owned
by DDP and should not be treated as a model-parameter init sync dimension unless
explicitly configured.

Explicit overrides remain necessary for unusual cases:

```text
[param, init_sync=none]
[param, init_sync=tp]
[param, init_sync=external]
```

`shared` should not remain a normal user-facing concept in the new API. It is an
implementation detail of initialization sync.

## Gradient Communication Inference

Gradient communication is context-dependent and cannot be inferred from
parameter layout alone.

The engine should infer gradient communication by building the symbolic backward
view for each `[param]` operand. The parameter gradient must communicate over
mesh dimensions that become partial while forming that operand's gradient.

Sources of parameter-gradient partials include:

- Mesh dimensions that shard axes summed out while computing the parameter
  gradient.
- Mesh dimensions carried as partials by the incoming output gradient.
- Mesh dimensions introduced by forward collectives whose autograd mapping
  leaves the parameter gradient partial.

Operationally, the engine should prefer to express this as a communication
obligation:

```text
ParameterGradComm:
  mode: inferred | explicit | none
  mesh_dims: ordered mesh groups
  backend: native | ddp | external
  schedule: synchronous | async | backend_default
```

The execution layer can later choose the concrete collective or hook mechanism.
For `backend=external`, `schedule` is not interpreted by `torch-einshard`; the
external owner decides whether communication is async, synchronous, fused, or
omitted.

Examples:

```text
b nh/tp l hd, hd [param, grad=async] -> b nh/tp l hd
```

The parameter is `hd`. Its gradient sums over `b`, `nh`, and `l`. Because `nh` is
sharded over `tp`, the parameter gradient is partial over `tp`; infer
`grad=tp:async`.

```text
b h/sp1 w/sp2 c, c [param, grad=async] -> b h/sp1 w/sp2 c
```

The parameter is `c`. Its gradient sums over `b`, `h`, and `w`. Because `h` and
`w` are sharded over `sp1` and `sp2`, infer `grad=sp1-sp2:async`.

```text
... out, out [param, grad=async] -> ... out
```

The engine can infer reductions only for axes that are symbolically known. If
`...` hides spatial axes sharded over `sp1` and `sp2`, the formula must be
expanded or the caller must provide enough runtime symbolic state for the engine
to know those axes exist.

### Why Layout Alone Is Insufficient

The same layout can require different gradient reductions in different contexts.

For example, a normalization parameter with layout `c` can be used in a spatial
formula where the gradient reduces over `sp1-sp2`, or in a head-sharded formula
where the gradient also reduces over `tp`:

```text
b h/sp1 w/sp2 c, c [param, grad=async] -> b h/sp1 w/sp2 c
b nh/tp l hd, hd [param, grad=async] -> b nh/tp l hd
```

Both parameters may be one-dimensional and replicated over the same model mesh,
but their required parameter-gradient reductions differ because the reduced axes
in the backward formula differ.

## Shard Metadata Inference

Local shard metadata should be derived from `ParameterState.spec` and the mesh,
not from `ParamSpec.axes`.

The replacement helpers should preserve the current behavior of:

- `param_local_slices`
- `param_local_shape`
- `param_shard_metadata`
- test and checkpoint helpers that materialize or reconstruct full parameters

They should also expose the shard-dimension metadata needed by downstream global
weight or gradient norm accounting.

The implementation should continue to support compound mesh groups through the
same `wrap_mesh` resolution rules used by tensor operations.

## Examples

### Column-Parallel Linear

```text
... c, out/tp c [param, grad=async] -> ... out/tp
```

The parameter layout is `out/tp c`. Initialization sync is inferred over the
managed mesh dimensions not used by `out/tp`. Gradient communication is inferred
from the symbolic backward formula and scheduled asynchronously.

### Row-Parallel Linear

```text
... in/tp, out in/tp [param, grad=async] -> ... out // tp
```

The parameter layout is `out in/tp`. The forward result is partial over `tp`.
The parameter-gradient obligation is inferred from the backward state rather
than from the layout alone.

### Output Bias

```text
... out, out [param, grad=async] -> ... out
```

The bias layout is `out`. Initialization sync is inferred over model mesh
dimensions where the bias is replicated. Gradient communication is inferred only
from symbolically known reduced axes. The engine should not add `tp` just because
the parameter is replicated over `tp`; it should add `tp` only if the backward
formula produces a `tp` partial.

### Spatial Positional Embedding

```text
b t h/sp1 w/sp2 c, b t h/sp1 w/sp2 c [param, grad=none]
  -> b t h/sp1 w/sp2 c
```

The parameter layout is spatially sharded. Initialization sync is inferred over
`tp`. A no-reduction override is allowed when the formula or training semantics
make extra parameter-gradient communication unnecessary.

### Patch Embedding Or Convolution Wrapper

```text
parameter: out in kh kw [param, grad=async]
activation: b in h/sp1 w/sp2
output: b out h/sp1 w/sp2
windows: h -> kh, w -> kw
```

This is wrapper metadata, not a valid `einshard` einsum formula. Convolution
needs the window-to-kernel relationship that `einconv` represents with window
metadata. Real convolution implementations may also hide the parameter inside
`nn.Conv2d` or a custom module. Those modules need wrappers or an explicit
annotation API that registers equivalent `ParameterState` information.

## Hidden Parameters And Wrappers

Formula annotations only work when the parameter is an actual operand in the
formula. Many important cases hide parameters inside modules or functions:

- `nn.Linear`
- `torch.nn.functional.linear`
- `nn.Conv2d`
- normalization modules
- Transformer Engine modules
- custom fused kernels

Those cases need one of two integration paths.

First, provide module wrappers that internally call `einshard` with annotated
parameter operands or directly register equivalent symbolic parameter uses.

Second, provide an explicit low-level registration API for advanced users and
integration packages. This API should register the same `ParameterState` fields
that annotated formulas would have inferred, including any external or none
overrides.

The design should not try to inspect arbitrary module internals and guess the
parameter semantics.

## DDP And External Integration

DDP, FSDP, optimizer hooks, and external training frameworks may own part of the
gradient communication pipeline.

The design should distinguish a truly external owner from a backend that
`torch-einshard` configures.

```text
[param, grad=external]
[param, grad=dp:external]
[param, grad=ddp]
[param, grad=dp:ddp]
[param, grad=async]
```

`grad=external` means the parameter has a gradient communication obligation, but
`torch-einshard` should not execute it. The obligation remains visible for
diagnostics, but the external backend owns mesh-group validation and execution
semantics. A scheduling suffix such as `async` does not apply to this case
because the external owner controls scheduling.

`grad=dp:external` is the explicit form for delegating only the `dp` obligation
to an outside system.

`grad=ddp` means the engine should infer the gradient obligation and satisfy it
through a PyTorch DDP-backed execution path configured by `torch-einshard`.
`grad=dp:ddp` is the explicit form for routing the `dp` obligation through that
backend. This is not fully external: the symbolic engine still records and
validates the obligation, and the execution layer installs the appropriate DDP
hook or integration.

`grad=async` means the engine should infer the obligation and use a native
`torch-einshard` execution path that launches asynchronous communication when
possible.

`grad=none` means no communication obligation exists for that parameter use.

The existing DDP communication hook can be migrated from `ParamSpec.reduce` to
`ParameterState.grad_comm`. The combined-reduction fast path can remain an
execution optimization when all parameters in a bucket have compatible inferred
or explicit gradient obligations.

### Replacing DDP Under The Hood

Replacing PyTorch DDP entirely is possible, but it is a separate execution-layer
project. The symbolic model should first make DDP one backend for
`ParameterState.grad_comm`; a later native backend can satisfy the same
obligations without PyTorch DDP.

A native DDP replacement would need to provide:

- Per-parameter autograd hooks that detect when each gradient is ready.
- Stable gradient buckets and parameter-to-bucket mappings.
- Async all-reduce or reduce-scatter launch when a bucket becomes ready.
- Correct data-parallel averaging semantics.
- Extra tensor- or spatial-parallel reductions from inferred `grad_comm`.
- Fused or compound-group reductions such as `dp-sp1-sp2` when legal.
- Synchronization before optimizer steps.
- Handling for unused parameters, repeated parameters, gradient accumulation,
  `no_sync`-style accumulation, mixed precision, gradient views, and distributed
  error propagation.

The potential gains are a unified symbolic gradient planner, better fusion across
`dp`, `tp`, `sp1`, and `sp2`, earlier async launches for parameter-specific
obligations, future reduce-scatter or sharded-optimizer support, and better
diagnostics that explain why each parameter needs each reduction.

The cost is rebuilding mature DDP behavior. The migration should therefore treat
native DDP replacement as a backend added after the symbolic obligations are
correct, not as a prerequisite for removing `ParamSpec`.

## Diagnostics

Parameter inference should fail loudly when it cannot prove safe metadata.

Important diagnostics include:

- The same parameter is observed with incompatible layouts.
- A `[param]` operand uses `...` and the hidden axes are needed for gradient
  inference.
- A parameter is used by DDP hooks or shard metadata helpers before it has a
  `ParameterState`.
- A requested mesh group does not exist.
- A gradient obligation is inferred over a mesh dimension whose process group is
  unmanaged by the current integration policy.
- An annotation asks for scheduling without an obligation, such as `[async]`.

The error messages should point to the parameter name when known and include the
formula or registration source that created the conflicting state.

## Migration Plan

The migration should happen in stages.

### Stage 1: Parser And Data Model

Status: implemented foundation.

- Extend the grammar to parse operand annotations.
- Add `ParameterState` and gradient/init-sync obligation data structures.
- Attach metadata to `torch.nn.Parameter` objects and use parameter identity for
  conflict checks during each `einshard` registration pass.
- Convert annotated parameter operands into attached parameter metadata during
  `einshard`.
- Mirror `ParamSpec` into compatible `ParameterState` metadata during this stage.

### Stage 2: Inference

Status: partially implemented.

- Implemented: infer parameter layout from annotated operand specs.
- Implemented: infer init sync dims from managed mesh dims minus layout shard dims.
- Implemented: local formula inference for visible sharded axes that become
  parameter-gradient reductions.
- Implemented: `grad=none`, `grad=external`, `grad=ddp`, explicit mesh-dim
  overrides, and explicit scheduling/backend suffixes at the metadata layer.
- Remaining: planner-aware symbolic backward analysis for distributed `[param]`
  operands, including subtracting communication already handled by autograd
  mappings.
- Remaining: finalization of pending inferred obligations before gradient
  execution helpers run.

### Stage 3: Helper Migration

Status: implemented foundation.

- Reimplement parameter shard metadata helpers on `ParameterState`.
- Reimplement module init sync on inferred init-sync obligations.
- Reimplement gradient reduction helpers on inferred grad obligations.
- Migrate the DDP communication hook to `ParameterState.grad_comm`.
- Preserve the combined DDP reduction optimization for compatible buckets.
- Add a concrete-native per-parameter autograd hook path for simple non-DDP
  training loops.
- Expose `ParameterState` metadata needed by downstream group validation and norm
  accounting.

### Stage 4: Wrapper Coverage

Status: partially implemented.

- Implemented: explicit layout registration through `ParameterState.from_layout`
  and `register_parameter_layout`.
- Implemented: `nn.Linear`-style weight/bias registration through
  `register_linear_parameters_`, including derived bias layout and atomic
  conflict validation.
- Implemented: Conv1d/2d/3d-style weight/bias registration through
  `register_conv_parameters_`, including rank-derived default kernel layouts,
  derived bias layout, grouped-convolution rejection, and atomic conflict
  validation.
- Implemented: normalization-style weight/bias registration through
  `register_norm_parameters_`, including shared layout defaults, same-shape bias
  validation, and atomic conflict validation.

- Add extension points for Transformer Engine or other fused modules.
- Cover SciGPT-style MLP, attention, Swin, positional embedding, and head cases.

### Stage 5: Compatibility And Removal

- Make `ParamSpec` a compatibility wrapper that populates `ParameterState`.
- Update docs and examples to prefer annotations and wrappers.
- Emit deprecation warnings after feature parity is established.
- Remove `ParamSpec` only after tests and downstream users no longer require it.

## Removal Criteria For `ParamSpec`

`ParamSpec` can be removed from the primary API when all of these are true:

- Parameter layout can be inferred or registered for supported parameter uses.
- Init sync can be inferred or overridden.
- Gradient communication can be inferred or overridden.
- DDP hook behavior has parity with current `ParamSpec.reduce` behavior.
- Local shard metadata helpers work from `ParameterState`.
- Downstream group validation and global norm accounting can be implemented from
  `ParameterState`.
- Hidden-parameter cases have wrappers or explicit registration APIs.
- Documentation no longer teaches `ParamSpec` as the normal path.
- A compatibility period has allowed downstream users to migrate.

The answer is therefore yes: this design allows `ParamSpec` to go away as a
user-facing concept. It does not allow the underlying metadata to disappear.
That metadata becomes inferred symbolic parameter state.

## Open Questions

- What is the exact public API for low-level explicit parameter registration?
- Which mesh dimensions are managed by `torch-einshard` versus DDP/FSDP by
  default?
- Should parameter inference run eagerly at each annotated `einshard` call or be
  finalized by an explicit module preparation step?
- How should async gradient obligations be scheduled when parameters participate
  in multiple formulas?
- Should checkpoint APIs require finalized `ParameterState`, or should they be
  able to accept an explicit one-off tensor spec?

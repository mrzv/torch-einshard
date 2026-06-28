from .einsum import einsum
from .conv import einconv
from .fft import einfft
from .grammar import parse_sharding, Axes
from .distributed import distributed_1d
from .families import cached_expand_axis_families
from .halo import einhalo, einwindow
from .mesh import CompoundDeviceMesh, wrap_mesh
from . import params as _params
from .params import (
    ParamShardMetadata,
    NativeGradReductionHandle,
    ParameterGradComm,
    ParameterInitSync,
    ParameterState,
    finalize_module_parameter_grad_comm_,
    finalize_parameter_grad_comm_,
    get_parameter_state,
    init_param_,
    init_params_,
    iter_parameter_states,
    param_local_shape,
    param_local_slices,
    param_shard_dims,
    param_shard_metadata,
    reduce_grad_,
    reduce_module_grads_,
    register_conv_parameters_,
    register_grad_reduction_hook_,
    register_linear_parameters_,
    register_module_parameter_layouts_,
    register_native_grad_reduction_hooks_,
    register_norm_parameters_,
    register_parameter_layout,
    register_parameter_operand,
    register_parameter_state,
    parameter_operand_state,
    set_parameter_state,
    sync_module_params_,
    sync_param_,
    validate_module_parameter_states_,
)
from .roll import einroll
from .sharding import AxisGroup, EllipsisAxis, TensorSpec
from .symbolic import (
    ExecutionPlan,
    PlanPolicy,
    TensorState,
    build_binary_transition_plan,
    build_unary_transition_plan,
    get_default_policy,
    get_optimization_policy,
    optimize,
    resolve_plan_policy,
    set_default_policy,
    set_last_plan,
)

try:
    from torch.compiler import disable as _torch_compile_disable
except (AttributeError, ImportError):
    from torch._dynamo import disable as _torch_compile_disable


def _axes(spec):
    return spec.axes if hasattr(spec, "axes") else spec


def _partials(spec):
    return getattr(spec, "partials", ())


def _without_ellipsis(axes):
    return Axes(axis for axis in axes if not isinstance(axis, EllipsisAxis))


def _flat_axes(spec):
    axes = _axes(spec)
    return axes.flat() if hasattr(axes, "flat") else axes

def _has_groups(spec):
    return any(isinstance(axis, AxisGroup) for axis in _axes(spec))

def all_local(shard):
    for s in shard:
        if s is None: continue
        if not s.local(): return False
    return True

def local_operation(shard):
    if any(getattr(s, "partials", ()) for s in shard if s is not None):
        return False

    if shard[1] is None:
        # X -> Z
        return Axes(set(_flat_axes(shard[0])) ^ set(_flat_axes(shard[2]))).local()
    else:
        # X,Y -> Z
        input0_axes = _flat_axes(shard[0])
        input1_axes = _flat_axes(shard[1])
        output_axes = _flat_axes(shard[2])
        input_axes = list(input0_axes) + list(input1_axes)

        for output_axis in output_axes:
            if output_axis in input_axes:
                continue
            if any(axis.name == output_axis.name for axis in input_axes):
                return False

        input0_by_name = {axis.name: axis for axis in input0_axes}
        input1_by_name = {axis.name: axis for axis in input1_axes}
        for name in set(input0_by_name) & set(input1_by_name):
            input0_axis = input0_by_name[name]
            input1_axis = input1_by_name[name]
            if not input0_axis.local() or not input1_axis.local():
                return False
            if input0_axis != input1_axis:
                return False

        return True


def _mesh_dim_names(mesh):
    return _params._mesh_dim_names_from(mesh)


def _input_specs(shard):
    if shard[1] is None:
        return (shard[0],)
    return (shard[0], shard[1])


def _parameter_operand_registrations(shard, xs, mesh):
    input_specs = _input_specs(shard)
    mesh_dim_names = None
    infer_grad = local_operation(shard)
    registrations = []
    for index, spec in enumerate(input_specs):
        annotation = getattr(spec, "annotation", None)
        if not getattr(annotation, "is_param", False):
            continue
        if mesh_dim_names is None:
            mesh_dim_names = _mesh_dim_names(mesh)
        if index >= len(xs):
            raise ValueError("Annotated parameter operand is missing a tensor argument")
        state = parameter_operand_state(
            xs[index],
            input_specs,
            shard[2],
            index,
            mesh_dim_names=mesh_dim_names,
            infer_grad=infer_grad,
        )
        registrations.append((xs[index], state))
    return registrations


def _prepare_parameter_operand_registrations(registrations):
    merged_by_id = {}
    params_by_id = {}
    for param, state in registrations:
        key = id(param)
        if key in merged_by_id:
            existing = merged_by_id[key]
        else:
            existing = _params._parameter_state_from_attached_metadata(param)
        merged_by_id[key] = _params._merge_parameter_state(existing, state)
        _params._validate_parameter_state_rank(param, merged_by_id[key])
        params_by_id[key] = param
    return [(params_by_id[key], state) for key, state in merged_by_id.items()]


def _commit_parameter_operand_registrations(registrations):
    for param, state in registrations:
        set_parameter_state(param, state)


def _distributed_einshard(shard, xs, mesh, shapes, policy):
    if any(_has_groups(s) for s in shard if s is not None):
        raise NotImplementedError("Factored axes are currently supported only for local reshape operations")

    if mesh is None:
        raise ValueError("Distributed einshard operations require mesh")

    return distributed_1d(shard, *xs, mesh = mesh, shapes = shapes, policy = policy)


def _execute_einshard(shard, xs, mesh, shapes, sizes, policy):
    if local_operation(shard):
        output = einsum(shard, *xs, sizes=sizes)
        plan = ExecutionPlan()
        plan.add("rank_local_einsum")
        set_last_plan(plan)
        return output

    return _distributed_einshard(shard, xs, mesh, shapes, policy)


def _validate_output_axes_present_once(shard, *, exact):
    def key(axis):
        return axis if exact else axis.name

    input_axes = {key(axis) for axis in _flat_axes(shard[0]) if not isinstance(axis, EllipsisAxis)}
    if shard[1] is not None:
        input_axes.update(
            key(axis) for axis in _flat_axes(shard[1]) if not isinstance(axis, EllipsisAxis)
        )
    output_axes = set()
    for axis in _flat_axes(shard[2]):
        if isinstance(axis, EllipsisAxis):
            continue
        axis_key = key(axis)
        if axis_key in output_axes:
            raise ValueError(f"Output dimension {axis} appears more than once")
        output_axes.add(axis_key)
        if axis_key not in input_axes:
            raise ValueError(f"Output dimension {axis} must be present in input")


def _validate_local_einshard_like(shard):
    if getattr(shard[0], "partials", ()):
        raise ValueError("Local einsum does not support partial inputs")
    if shard[1] is not None and getattr(shard[1], "partials", ()):
        raise ValueError("Local einsum does not support partial inputs")
    _validate_output_axes_present_once(shard, exact=True)


def _validate_unary_distributed_einshard_like(shard):
    input_axes = tuple(_flat_axes(shard[0]))
    output_axes = tuple(_flat_axes(shard[2]))
    input_has_ellipsis = any(isinstance(axis, EllipsisAxis) for axis in input_axes)
    output_has_ellipsis = any(isinstance(axis, EllipsisAxis) for axis in output_axes)
    if input_has_ellipsis or output_has_ellipsis:
        if input_has_ellipsis != output_has_ellipsis:
            raise NotImplementedError("Distributed unary ellipsis requires ellipsis in both input and output")
        input_fixed = [axis for axis in input_axes if not isinstance(axis, EllipsisAxis)]
        output_fixed = [axis for axis in output_axes if not isinstance(axis, EllipsisAxis)]
        if len(input_fixed) != len(output_fixed):
            raise ValueError("Input tensor rank is incompatible with ellipsis notation")
        return
    if len(input_axes) != len(output_axes):
        raise ValueError("Input and output dimensions must match in the equation")
    input_names = {axis.name for axis in input_axes}
    output_names = {axis.name for axis in output_axes}
    if input_names != output_names:
        raise ValueError("Input and output axes must match")


def _validate_binary_input_only_sharded_contracting_axes(shard):
    input0_axes = [axis for axis in _flat_axes(shard[0]) if not isinstance(axis, EllipsisAxis)]
    input1_axes = [axis for axis in _flat_axes(shard[1]) if not isinstance(axis, EllipsisAxis)]
    input0_names = {axis.name for axis in input0_axes}
    input1_names = {axis.name for axis in input1_axes}
    output_names = {
        axis.name for axis in _flat_axes(shard[2]) if not isinstance(axis, EllipsisAxis)
    }
    output_partials = set(_partials(shard[2]))

    def validate_input_only_contracting_axes(axes, other_names):
        for axis in axes:
            if axis.name in other_names or axis.name in output_names:
                continue
            if axis.shard_dim and axis.shard_dim not in output_partials:
                raise ValueError(
                    f"Omitted sharded input-only dimension {axis} must be preserved as an output partial"
                )

    validate_input_only_contracting_axes(input0_axes, input1_names)
    validate_input_only_contracting_axes(input1_axes, input0_names)


def _validate_binary_distributed_einshard_like(shard):
    if getattr(shard[0], "partials", ()) or getattr(shard[1], "partials", ()):
        raise ValueError("Binary distributed einshard_like does not support partial inputs")


def _validate_distributed_tensor_specs(shard, mesh):
    mesh_dim_names = _mesh_dim_names(mesh)
    for spec in shard:
        if spec is None:
            continue
        axes = tuple(_flat_axes(spec))
        if any(isinstance(axis, EllipsisAxis) for axis in axes):
            continue
        TensorState.from_spec(spec, mesh_dim_names=mesh_dim_names)


def _validate_distributed_mesh_dims(shard, mesh):
    mesh_dim_names = _mesh_dim_names(mesh)
    groups = []
    for spec in shard:
        if spec is None:
            continue
        groups.extend(
            axis.shard_dim
            for axis in _flat_axes(spec)
            if not isinstance(axis, EllipsisAxis) and axis.shard_dim
        )
        groups.extend(_partials(spec))
    _params._validate_known_mesh_groups(mesh_dim_names, groups, "operation")


def _has_ellipsis(shard):
    return any(
        isinstance(axis, EllipsisAxis)
        for spec in shard
        if spec is not None
        for axis in _flat_axes(spec)
    )


def _validate_distributed_transition_plan(shard):
    if shard[1] is None:
        if _has_ellipsis(shard):
            return
        build_unary_transition_plan(shard[0], shard[2])
    else:
        build_binary_transition_plan(
            TensorSpec(_without_ellipsis(_axes(shard[0])), _partials(shard[0])),
            TensorSpec(_without_ellipsis(_axes(shard[1])), _partials(shard[1])),
            TensorSpec(_without_ellipsis(_axes(shard[2])), _partials(shard[2])),
        )


def _validate_einshard_like(shard, mesh):
    if mesh is not None:
        _validate_distributed_mesh_dims(shard, mesh)
    if shard[1] is not None:
        _validate_binary_input_only_sharded_contracting_axes(shard)
    if local_operation(shard):
        _validate_local_einshard_like(shard)
        return
    _validate_output_axes_present_once(shard, exact=False)
    if any(_has_groups(s) for s in shard if s is not None):
        raise NotImplementedError("Factored axes are currently supported only for local reshape operations")
    if mesh is None:
        raise ValueError("Distributed einshard operations require mesh")
    _validate_distributed_tensor_specs(shard, mesh)
    if shard[1] is None:
        _validate_unary_distributed_einshard_like(shard)
    else:
        _validate_binary_distributed_einshard_like(shard)
    _validate_distributed_transition_plan(shard)


def einshard(shard, *xs, mesh = None, shapes = None, sizes = None, families = None, optimize = None, policy = None):
    policy = resolve_plan_policy(optimize=optimize, policy=policy)
    shard, sizes = cached_expand_axis_families(shard, sizes, families)
    shard = parse_sharding(shard)
    registrations = _parameter_operand_registrations(shard, xs, mesh)
    registrations = _prepare_parameter_operand_registrations(registrations)
    output = _execute_einshard(shard, xs, mesh, shapes, sizes, policy)
    _commit_parameter_operand_registrations(registrations)
    return output


@_torch_compile_disable
def einshard_like(shard, *xs, mesh=None, sizes=None, families=None):
    shard, _ = cached_expand_axis_families(shard, sizes, families)
    shard = parse_sharding(shard)
    registrations = _parameter_operand_registrations(shard, xs, mesh)
    registrations = _prepare_parameter_operand_registrations(registrations)
    _validate_einshard_like(shard, mesh)
    _commit_parameter_operand_registrations(registrations)
    return None

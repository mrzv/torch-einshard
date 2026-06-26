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
from .sharding import AxisGroup
from .symbolic import (
    ExecutionPlan,
    PlanPolicy,
    get_default_policy,
    get_optimization_policy,
    optimize,
    resolve_plan_policy,
    set_default_policy,
    set_last_plan,
)


def _axes(spec):
    return spec.axes if hasattr(spec, "axes") else spec

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


def einshard(shard, *xs, mesh = None, shapes = None, sizes = None, families = None, optimize = None, policy = None):
    policy = resolve_plan_policy(optimize=optimize, policy=policy)
    shard, sizes = cached_expand_axis_families(shard, sizes, families)
    shard = parse_sharding(shard)
    registrations = _parameter_operand_registrations(shard, xs, mesh)
    registrations = _prepare_parameter_operand_registrations(registrations)
    output = _execute_einshard(shard, xs, mesh, shapes, sizes, policy)
    _commit_parameter_operand_registrations(registrations)
    return output

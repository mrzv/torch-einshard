# API Reference

This page is generated from the public Python API. Many public objects are still
documented primarily in the user guide pages; docstrings can be expanded over
time as the API stabilizes.

## Parameter Metadata APIs

Core metadata objects:

- {py:class}`torch_einshard.ParameterState`
- {py:class}`torch_einshard.ParameterInitSync`
- {py:class}`torch_einshard.ParameterGradComm`

Attach, inspect, and validate metadata:

- {py:func}`torch_einshard.set_parameter_state`
- {py:func}`torch_einshard.get_parameter_state`
- {py:func}`torch_einshard.iter_parameter_states`
- {py:func}`torch_einshard.validate_module_parameter_states_`

Register hidden or explicit parameter layouts:

- {py:func}`torch_einshard.register_parameter_layout`
- {py:func}`torch_einshard.register_module_parameter_layouts_`
- {py:func}`torch_einshard.register_linear_parameters_`
- {py:func}`torch_einshard.register_conv_parameters_`
- {py:func}`torch_einshard.register_norm_parameters_`
- {py:func}`torch_einshard.finalize_parameter_grad_comm_`
- {py:func}`torch_einshard.finalize_module_parameter_grad_comm_`

Execute parameter synchronization or gradient communication:

- {py:func}`torch_einshard.sync_param_`
- {py:func}`torch_einshard.sync_module_params_`
- {py:func}`torch_einshard.reduce_grad_`
- {py:func}`torch_einshard.reduce_module_grads_`
- {py:func}`torch_einshard.register_native_grad_reduction_hooks_`
- {py:func}`torch_einshard.register_grad_reduction_hook_`

Shard metadata helpers:

- {py:func}`torch_einshard.param_shard_dims`
- {py:func}`torch_einshard.param_local_slices`
- {py:func}`torch_einshard.param_local_shape`
- {py:func}`torch_einshard.param_shard_metadata`

## Full API

```{eval-rst}
.. automodule:: torch_einshard
   :members:
   :undoc-members:
```

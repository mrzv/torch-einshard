import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from .families import cached_expand_axis_families, expand_family_mapping
from .grammar import parse_sharding
from .halo import _halo_tensor, _normalize_halo
from .sharding import EllipsisAxis


def _axes(spec):
    return spec.axes if hasattr(spec, "axes") else spec


def _axis_names(axes, *, label):
    names = []
    for axis in axes:
        if isinstance(axis, EllipsisAxis):
            raise NotImplementedError(f"einconv does not support ellipsis axes in {label}")
        if not hasattr(axis, "name"):
            raise NotImplementedError(f"einconv does not support factored axes in {label}")
        if axis.name in names:
            raise ValueError(f"Axis {axis.name!r} appears more than once in {label}")
        names.append(axis.name)
    return names


def _normalize_sequence(value, ndim, *, label):
    if isinstance(value, int):
        return (value,) * ndim
    values = tuple(value)
    if len(values) != ndim:
        raise ValueError(f"{label} must have {ndim} values")
    return values


def _default_padding(weight, kernel_dims, dilation):
    padding = []
    for dim, dil in zip(kernel_dims, dilation):
        effective = dil * (weight.shape[dim] - 1)
        if effective % 2:
            raise ValueError("Default einconv padding requires odd effective kernel sizes")
        padding.append(effective // 2)
    return tuple(padding)


def _conv_function(ndim):
    if ndim == 1:
        return F.conv1d
    if ndim == 2:
        return F.conv2d
    if ndim == 3:
        return F.conv3d
    raise ValueError("einconv supports 1D, 2D, and 3D convolutions")


def _checkpoint(function, tensors, mode, use_reentrant):
    if mode in (False, None, "none"):
        return function(*tensors)
    if mode in (True, "full", "conv"):
        return torch_checkpoint(function, *tensors, use_reentrant=use_reentrant)
    raise ValueError("checkpoint must be one of 'full', 'conv', 'none', True, or False")


def einconv(
    shard,
    x,
    weight,
    windows,
    *,
    bias=None,
    padding=None,
    stride=1,
    dilation=1,
    groups=1,
    boundary="constant",
    fill=0,
    mesh=None,
    shapes=None,
    families=None,
    checkpoint="full",
    use_reentrant=False,
):
    """Apply a sharding-aware halo exchange followed by a PyTorch convolution."""
    shard, _ = cached_expand_axis_families(shard, families=families)
    windows = expand_family_mapping(windows, families, label="Window")
    input_spec, weight_spec, output_spec = parse_sharding(shard)
    if weight_spec is None:
        raise ValueError("einconv expects a binary sharding expression")
    if input_spec.partials or weight_spec.partials or output_spec.partials:
        raise ValueError("einconv does not support partial tensor specs")
    if groups != 1:
        raise NotImplementedError("einconv currently supports only groups=1")

    input_axes = _axes(input_spec)
    weight_axes = _axes(weight_spec)
    output_axes = _axes(output_spec)
    input_names = _axis_names(input_axes, label="input")
    weight_names = _axis_names(weight_axes, label="weight")
    output_names = _axis_names(output_axes, label="output")

    spatial_names = list(windows.keys())
    kernel_names = list(windows.values())
    if len(spatial_names) not in (1, 2, 3):
        raise ValueError("einconv supports 1D, 2D, and 3D convolutions")
    if len(set(kernel_names)) != len(kernel_names):
        raise ValueError("Window axes must be distinct")

    missing = [name for name in spatial_names if name not in input_names]
    if missing:
        raise ValueError(f"Window source axis {missing[0]!r} is not present in the input")
    missing = [name for name in kernel_names if name not in weight_names]
    if missing:
        raise ValueError(f"Window axis {missing[0]!r} is not present in the weight")
    if any(name in input_names or name in output_names for name in kernel_names):
        raise ValueError("Window axes must be weight-only axes")

    for axis in weight_axes:
        if not axis.local():
            raise ValueError("einconv weight axes must be local")

    output_by_name = {axis.name: axis for axis in output_axes}
    for axis in input_axes:
        if axis.name in output_by_name and output_by_name[axis.name] != axis:
            raise ValueError("einconv output must preserve input axis sharding")

    shared = [name for name in input_names if name in weight_names and name not in spatial_names]
    channel_names = [name for name in shared if name not in output_names]
    if len(channel_names) != 1:
        raise ValueError("einconv expects exactly one input channel axis")
    channel_name = channel_names[0]
    if not input_axes[input_names.index(channel_name)].local():
        raise ValueError("einconv input channel axis must be local")

    out_channel_names = [name for name in weight_names if name in output_names and name not in input_names]
    if len(out_channel_names) != 1:
        raise ValueError("einconv expects exactly one output channel axis")
    out_channel_name = out_channel_names[0]
    if not output_by_name[out_channel_name].local():
        raise ValueError("einconv output channel axis must be local")

    allowed_weight = {out_channel_name, channel_name, *kernel_names}
    extra_weight = [name for name in weight_names if name not in allowed_weight]
    if extra_weight:
        raise ValueError(f"Unexpected weight axis {extra_weight[0]!r}")

    expected_output = set(input_names) - {channel_name}
    expected_output.add(out_channel_name)
    if set(output_names) != expected_output:
        raise ValueError("einconv output axes must be input axes with the channel axis replaced by the output channel axis")

    ndim = len(spatial_names)
    stride = _normalize_sequence(stride, ndim, label="stride")
    dilation = _normalize_sequence(dilation, ndim, label="dilation")
    if any(value != 1 for value in stride):
        raise NotImplementedError("einconv currently supports only stride=1")

    kernel_dims = tuple(weight_names.index(name) for name in kernel_names)
    if padding is None:
        padding_values = _default_padding(weight, kernel_dims, dilation)
    else:
        padding_values = expand_family_mapping(padding, families, label="Padding")
        if isinstance(padding_values, dict):
            unknown = set(padding_values) - set(spatial_names)
            if unknown:
                raise ValueError(f"Padding axis {sorted(unknown)[0]!r} has no matching window")
            missing = [name for name in spatial_names if name not in padding_values]
            if missing:
                raise ValueError(f"Missing padding for window source axis {missing[0]!r}")
            padding_values = tuple(padding_values[name] for name in spatial_names)
        else:
            padding_values = _normalize_sequence(padding_values, ndim, label="padding")

    for name, value, kernel_dim, dil in zip(spatial_names, padding_values, kernel_dims, dilation):
        left, right = _normalize_halo(value)
        expected = dil * (weight.shape[kernel_dim] - 1)
        if left + right != expected:
            raise ValueError(
                f"Padding for axis {name!r} must preserve the spatial length; "
                f"got left + right = {left + right}, expected {expected}"
            )

    halos = {name: value for name, value in zip(spatial_names, padding_values)}
    conv = _conv_function(ndim)
    batch_names = [name for name in input_names if name not in spatial_names and name != channel_name]
    x_permutation = [input_names.index(name) for name in batch_names + [channel_name] + spatial_names]
    weight_permutation = [weight_names.index(name) for name in [out_channel_name, channel_name] + kernel_names]
    conv_output_names = batch_names + [out_channel_name] + spatial_names
    output_permutation = [conv_output_names.index(name) for name in output_names]

    def run_conv(x_arg, weight_arg, bias_arg=None):
        x_conv = x_arg.permute(x_permutation).contiguous()
        batch_shape = x_conv.shape[:len(batch_names)]
        x_conv = x_conv.reshape((-1, x_conv.shape[len(batch_names)], *x_conv.shape[len(batch_names) + 1:]))
        weight_conv = weight_arg.permute(weight_permutation).contiguous()
        result = conv(x_conv, weight_conv, bias_arg, stride=stride, padding=0, dilation=dilation, groups=groups)
        result = result.reshape((*batch_shape, *result.shape[1:]))
        return result.permute(output_permutation).contiguous()

    def run_full(x_arg, weight_arg, *optional_bias):
        bias_arg = optional_bias[0] if optional_bias else None
        haloed = _halo_tensor(input_spec, x_arg, halos, mesh=mesh, shapes=shapes, boundary=boundary, fill=fill)
        if checkpoint == "conv":
            tensors = (haloed, weight_arg) if bias_arg is None else (haloed, weight_arg, bias_arg)
            return _checkpoint(lambda *args: run_conv(*args), tensors, "conv", use_reentrant)
        return run_conv(haloed, weight_arg, bias_arg)

    tensors = (x, weight) if bias is None else (x, weight, bias)
    if checkpoint == "conv":
        return run_full(*tensors)
    return _checkpoint(lambda *args: run_full(*args), tensors, checkpoint, use_reentrant)

# FFT

`einfft` applies `torch.fft.fftn` or `torch.fft.ifftn` over named axes while
using sharding notation for input and output layout.

```python
z = es.einfft("b x c -> b k c", x, axes={"x": "k"})
```

The `axes` mapping names each transformed input axis and the corresponding
output frequency axis. Multiple axes use a single multidimensional FFT:

```python
z = es.einfft(
    "b x y c -> b kx ky c",
    x,
    axes={"x": "kx", "y": "ky"},
    norm="ortho",
)
```

Axis families can make one FFT expression cover both 2D and 3D FFTs. Family
entries may include sharding suffixes, and `axes` can map a family name to the
corresponding concrete output axis names:

```python
spatial_axes = ("x", "y", "z")[:dim]
freq_axes = ("kx", "ky", "kz")[:dim]
mesh_dims = ("sp", "dp", "mp")[:dim]

spatial = tuple(f"{axis}/{mesh_dim}" for axis, mesh_dim in zip(spatial_axes, mesh_dims))
freq = tuple(f"{axis}/{mesh_dim}" for axis, mesh_dim in zip(freq_axes, mesh_dims))
shapes = {
    mesh_dim: {axis: split_shapes[axis], freq_axis: split_shapes[axis]}
    for axis, freq_axis, mesh_dim in zip(spatial_axes, freq_axes, mesh_dims)
}

z = es.einfft(
    "b *spatial c -> b *freq c",
    x,
    axes={"spatial": freq_axes},
    families={"spatial": spatial, "freq": freq},
    mesh=mesh,
    shapes=shapes,
)
```

Use `inverse=True` for `torch.fft.ifftn`:

```python
x = es.einfft("b k c -> b x c", z, axes={"k": "x"}, inverse=True)
```

Use `real=True` for real FFT variants. Forward real mode calls
`torch.fft.rfftn`; inverse real mode calls `torch.fft.irfftn`. The last axis in
the `axes` mapping is the half-spectrum axis, matching PyTorch's
`rfftn`/`irfftn` convention:

```python
z = es.einfft("b x y -> b kx ky", x, axes={"x": "kx", "y": "ky"}, real=True)
x = es.einfft(
    "b kx ky -> b x y",
    z,
    axes={"kx": "x", "ky": "y"},
    inverse=True,
    real=True,
    signal_sizes={"y": y_size},
)
```

`signal_sizes` is only needed for inverse real FFTs when the original length
cannot be inferred from the half-spectrum size, such as odd-length signals.

## Sharded Fast Paths

Sharded transform axes are supported. The optimized path handles
complex-to-complex FFTs when each sharded transform axis stays on the same mesh
dimension with equal shard sizes and local shard size divisible by the mesh size.
Multiple sharded transform axes are supported when they use distinct mesh
dimensions. A sharded transform axis cannot share its mesh dimension with another
input or output axis.

```python
z = es.einfft(
    "b x/tp -> b k/tp",
    x_shard,
    axes={"x": "k"},
    mesh=mesh,
    shapes={"tp": {"x": x_shapes, "k": k_shapes}},
)
z = es.einfft(
    "b x/tp y -> b kx/tp ky",
    x_real_shard,
    axes={"x": "kx", "y": "ky"},
    real=True,
    mesh=mesh,
    shapes={"tp": {"x": x_shapes, "kx": x_shapes}},
)
```

Use `explain=True` to inspect whether a layout will use an optimized path
without running the FFT:

```python
info = es.einfft(
    "b x/tp -> b k/tp",
    x_real_shard,
    axes={"x": "k"},
    real=True,
    mesh=mesh,
    shapes={"tp": {"x": x_shapes, "k": k_half_shapes}},
    explain=True,
)
```

Unsupported sharded transform layouts fall back to gather the full transform
axis, run the local FFT, then split the output frequency axis if requested. That
path emits a `RuntimeWarning` because it materializes the full transform axis on
every rank. `einfft` currently supports explicit named axes; it does not support
ellipsis axes, factored axes, or partial tensor specs.

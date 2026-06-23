# Ghost Cells, Windows, And Convolution

`einhalo` extends one or more axes with ghost cells. Local axes are padded
directly. Sharded axes exchange only the needed boundary intervals with owning
ranks, then use autograd to route halo gradients back to those owners.

```python
xg = es.einhalo(
    "b h/sp_h w/sp_w c",
    x,
    {"h": 1, "w": 1},
    mesh=mesh,
    shapes={"sp_h": h_shapes, "sp_w": w_shapes},
)
```

Halo widths can be symmetric integers or `(left, right)` pairs:

```python
xg = es.einhalo("h/sp c", x, {"h": (2, 1)}, mesh=mesh, shapes=shapes)
```

Supported boundaries are `constant`, `periodic`, and `replicate`. Constant
boundaries use `fill=0` by default:

```python
xg = es.einhalo("h/sp c", x, {"h": 1}, boundary="constant", fill=-1, mesh=mesh, shapes=shapes)
```

## Windows

`einwindow` is a sharding-aware `unfold`/`im2col`. It first applies the needed
halo, then preserves the owned center axis and adds an explicit local window
axis:

```python
patches = es.einwindow(
    "b h/sp_h w/sp_w c -> b h/sp_h w/sp_w kh kw c",
    x,
    {"h": "kh", "w": "kw"},
    {"h": 1, "w": 1},
    mesh=mesh,
    shapes={"sp_h": h_shapes, "sp_w": w_shapes},
)
```

For a 3x3 convolution, the window axes can then be contracted normally:

```python
y = es.einshard(
    "b h/sp_h w/sp_w kh kw c, o kh kw c -> b h/sp_h w/sp_w o",
    patches,
    weight,
    mesh=mesh,
)
```

## Convolution

`einconv` provides the lower-memory convolution path directly. It applies the
needed `einhalo` padding, calls PyTorch `conv1d`/`conv2d`/`conv3d` without
materializing im2col windows, and checkpoints the full halo-plus-convolution
forward by default so backward recomputes activations instead of saving them:

```python
y = es.einconv(
    "b h/sp_h w/sp_w c, o c kh kw -> b h/sp_h w/sp_w o",
    x,
    weight,
    {"h": "kh", "w": "kw"},
    bias=bias,
    mesh=mesh,
    shapes={"sp_h": h_shapes, "sp_w": w_shapes},
)
```

The `checkpoint` option defaults to `"full"`. Use `checkpoint="conv"` to
checkpoint only the local convolution after halo exchange, or `checkpoint=False`
to disable checkpointing. The initial implementation supports 1D/2D/3D
convolutions with stride 1, one input-channel axis, one output-channel axis, and
local kernel axes. Padding must preserve each spatial length; omitted padding
defaults to same-padding for odd effective kernel sizes. Grouped convolutions are
not supported yet.

For neighborhood attention, window the key and value tensors, then contract the
query only against local neighborhood axes instead of a full sequence/spatial
axis.

Axis families work for both halo/window widths and window-axis names:

```python
patches = es.einwindow(
    "b *spatial c -> b *spatial *window c",
    x,
    {"spatial": ("kh", "kw")},
    {"spatial": (1, 1)},
    families={"spatial": ("h/sp_h", "w/sp_w"), "window": ("kh", "kw")},
    mesh=mesh,
    shapes={"sp_h": h_shapes, "sp_w": w_shapes},
)
```

Sharded halo exchange handles uneven shards, halos larger than a local shard,
and `periodic` wraparound by slicing the requested ghost interval into
rank-owned pieces.

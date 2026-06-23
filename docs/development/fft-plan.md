# FFT Follow-Up Plan

The current `einfft` implementation has useful distributed fast paths, but it is not a general distributed FFT engine yet.

## Optimized Backward

The forward path for eligible sharded FFTs uses a distributed Cooley-Tukey decomposition with all-to-all transposes and local factor FFTs. The backward path uses the same distributed factorization for the adjoint transform, with the inverse direction and normalization selected to match PyTorch FFT autograd semantics.

This avoids materializing the full transform axis during backward for layouts that qualify for the fast path. Backward for unsupported layouts still follows the slow fallback path through the existing gather/split autograd mappings.

## Broader Layouts

The fast path handles separable multi-axis FFTs when each sharded transform axis can use the distributed 1D kernel. Representative complex and real cases are:

```python
es.einfft("b x/tp -> b k/tp", x, axes={"x": "k"}, mesh=mesh, shapes=shapes)
es.einfft("b x/tp y -> b kx/tp ky", x, axes={"x": "kx", "y": "ky"}, real=True, mesh=mesh, shapes=shapes)
```

Current fast-path constraints:

- Each sharded transform axis must stay sharded on the same mesh dimension across input and output.
- Multiple sharded transform axes must use distinct mesh dimensions.
- A sharded transform axis cannot share its mesh dimension with another input or output axis.
- Full-complex transform axes have equal input and output shard sizes. Forward `rfftn` fast paths may use different output shard sizes for the sharded half-spectrum axis.
- Local shard size is divisible by the mesh size.
- Input is complex for complex FFTs and inverse real FFTs; forward real FFT fast paths require real `float32` or `float64` input.
- No factored axes, ellipsis axes, or partial specs.

Real FFT variants are supported through `real=True`. Forward real mode uses `torch.fft.rfftn`; inverse real mode uses `torch.fft.irfftn` and accepts `signal_sizes` for odd-length or otherwise ambiguous inverse lengths. The fast path supports real FFTs when the half-spectrum axis is local and sharded transform axes are full-complex axes handled by the distributed 1D kernel. Forward real FFTs can also fast-path when the half-spectrum axis itself is sharded and the full input axis satisfies the distributed 1D kernel constraints. Inverse real fast paths require the half-spectrum axis to be local and any specified non-half-axis `signal_sizes` to match the current global axis sizes; padding or cropping those axes falls back.

Use `explain=True` on `einfft` to inspect the selected optimized path or the fallback reason without running the FFT.

Future broader-layout work should consider:

- Uneven shard sizes.
- Local shard sizes that are not divisible by world size.
- Transform axes that move between mesh dimensions, such as `x/tp -> k/sp`.
- Multiple sharded transform axes on the same mesh dimension, if a clear distribution semantics is defined.
- Distributed inverse real FFT fast paths where the half-spectrum axis itself is sharded.
- Factored or reshaped transform axes once the intended semantics are clear.
- Alternative output layouts, such as block-cyclic or transposed frequency shards, when those are more efficient than contiguous `k/tp` shards.

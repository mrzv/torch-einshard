# FFT Follow-Up Plan

The current `einfft` implementation has useful distributed fast paths, but it is not a general distributed FFT engine yet.

## Optimized Backward

The forward path for eligible sharded FFTs uses a distributed Cooley-Tukey decomposition with all-to-all transposes and local factor FFTs. The backward path uses the same distributed factorization for the adjoint transform, with the inverse direction and normalization selected to match PyTorch FFT autograd semantics.

This avoids materializing the full transform axis during backward for layouts that qualify for the fast path. Backward for unsupported layouts still follows the slow fallback path through the existing gather/split autograd mappings.

## Broader Layouts

The fast path handles separable multi-axis FFTs when each sharded transform axis can use the distributed 1D kernel. A representative case is:

```python
es.einfft("b x/tp -> b k/tp", x, axes={"x": "k"}, mesh=mesh, shapes=shapes)
```

Current fast-path constraints:

- Each sharded input transform axis must have an output frequency axis sharded on the same mesh dimension.
- Multiple sharded transform axes must use distinct mesh dimensions.
- Input and output shard sizes are equal.
- Local shard size is divisible by the mesh size.
- Input is complex.
- No factored axes, ellipsis axes, partial specs, or real FFT fast paths.

Real FFT variants are supported through `real=True`. Forward real mode uses `torch.fft.rfftn`; inverse real mode uses `torch.fft.irfftn` and accepts `signal_sizes` for odd-length or otherwise ambiguous inverse lengths. Sharded real FFTs currently use the gather/local-FFT/split fallback because the half-spectrum axis changes shape and does not fit the complex Cooley-Tukey kernel yet.

Future broader-layout work should consider:

- Uneven shard sizes.
- Local shard sizes that are not divisible by world size.
- Transform axes that move between mesh dimensions, such as `x/tp -> k/sp`.
- Multiple sharded transform axes on the same mesh dimension, if a clear distribution semantics is defined.
- Distributed fast paths for real FFT variants such as `rfft` and `irfft`.
- Factored or reshaped transform axes once the intended semantics are clear.
- Alternative output layouts, such as block-cyclic or transposed frequency shards, when those are more efficient than contiguous `k/tp` shards.

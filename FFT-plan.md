# FFT Follow-Up Plan

The current `einfft` implementation has a useful first distributed fast path, but it is not a general distributed FFT engine yet.

## Optimized Backward

The forward path for eligible sharded FFTs uses a distributed Cooley-Tukey decomposition with all-to-all transposes and local factor FFTs. The backward path uses the same distributed factorization for the adjoint transform, with the inverse direction and normalization selected to match PyTorch FFT autograd semantics.

This avoids materializing the full transform axis during backward for layouts that qualify for the fast path. Backward for unsupported layouts still follows the slow fallback path through the existing gather/split autograd mappings.

## Broader Layouts

The fast path currently handles a narrow case like:

```python
es.einfft("b x/tp -> b k/tp", x, axes={"x": "k"}, mesh=mesh, shapes=shapes)
```

Current fast-path constraints:

- Exactly one transformed axis.
- The input transform axis is sharded.
- The output frequency axis is sharded on the same mesh dimension.
- Input and output shard sizes are equal.
- Local shard size is divisible by the mesh size.
- Input is complex.
- No factored axes, ellipsis axes, partial specs, `rfft`, or `irfft`.

Future broader-layout work should consider:

- Uneven shard sizes.
- Local shard sizes that are not divisible by world size.
- Transform axes that move between mesh dimensions, such as `x/tp -> k/sp`.
- Multiple transform axes for 2D or 3D distributed FFTs.
- Mixed local and sharded FFT axes.
- Real FFT variants such as `rfft` and `irfft`.
- Factored or reshaped transform axes once the intended semantics are clear.
- Alternative output layouts, such as block-cyclic or transposed frequency shards, when those are more efficient than contiguous `k/tp` shards.

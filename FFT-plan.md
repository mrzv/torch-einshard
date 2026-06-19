# FFT Follow-Up Plan

The current `einfft` implementation has a useful first distributed fast path, but it is not a general distributed FFT engine yet.

## Fully Optimized Backward

The forward path for eligible sharded FFTs uses a distributed Cooley-Tukey decomposition with all-to-all transposes and local factor FFTs. The backward path is correct, but it is still a correctness-first implementation: it gathers the full gradient FFT axis on every rank, applies the adjoint local FFT, and splits the result again.

This means backward currently has the same memory and communication profile as the slow gather/FFT/split fallback. A fully optimized backward should avoid materializing the full transform axis and instead implement the adjoint using the same distributed factorization/all-to-all structure as the forward path, with the correct inverse direction and normalization scaling.

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

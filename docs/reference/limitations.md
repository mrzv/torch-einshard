# Current Limitations

- The grammar supports at most two input tensors.
- Partial notation represents sum reductions only.
- Factored axes support one-level parenthesized groups; grouped transformations
  are local reshape operations around `torch.einsum`.
- Axis families are a pre-parse notation expansion; expanded expressions must
  still be valid `einshard` notation.
- Repartition and `einroll` still use correctness-first gather/split fallbacks
  for unsupported cases or missing shape metadata; performance-sensitive
  fallbacks emit `RuntimeWarning`.
- Multi-axis repartition swaps are currently limited to pure ownership swaps
  across equal-sized mesh dimensions with matching split metadata.
- `einhalo` and `einwindow` require explicit named axes; ellipsis and factored
  axes are not supported. `einwindow` also rejects partial specs, must preserve
  every input axis and its sharding in the output, and requires added window axes
  to be local.
- Optimized `einfft` paths require explicit named axes, equal full-complex shard
  sizes, local shard sizes divisible by mesh size, distinct mesh dimensions for
  multiple sharded transform axes, and no transform axis sharing its mesh
  dimension with another input or output axis. Inverse real FFTs with a sharded
  half-spectrum axis still fall back, and sharded FFT fallbacks emit
  `RuntimeWarning` because they materialize full transform axes.
- Parenthesized partial reductions over multiple mesh dimensions are applied
  sequentially; use `wrap_mesh` with a compound name for a single compound group
  reduction.

Invalid public `einshard` and `einroll` expressions raise `ValueError` with the
original expression included in the message.

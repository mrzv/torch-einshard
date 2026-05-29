# Distributed Pattern Plan

This document records notation ideas for distributed patterns observed in `../MachineLearning/SciGPT/scaling-transformers-physical-sciences` and how they could map into `torch-einshard`.

## Existing Notation Fits

Unary split:

```text
b h w c -> b h/sp1 w c
b h w c -> b h w/sp2 c
b h w c -> b/dp h w c
```

Unary gather:

```text
b h/sp1 w c -> b h w c
b h w/sp2 c -> b h w c
b/dp h w c -> b h w c
```

Chained multi-axis spatial split:

```text
b h w c -> b h/sp1 w/sp2 c
```

Chained multi-axis spatial gather:

```text
b h/sp1 w/sp2 c -> b h w c
```

Patch/token layout transforms are local permutations once the tensor is already sharded:

```text
b t c h/sp1 w/sp2 -> b t h/sp1 w/sp2 c
```

If factored axes are added later, patch expansion/head reshape patterns could be represented as:

```text
b t h w p q c -> b t c h p w q
```

## Distributed Contractions

Tensor-parallel column-sharded linear or QKV projection:

```text
b n c, o/tp c -> b n o/tp
```

This produces sharded output features and does not need a forward all-reduce.

Tensor-parallel row-sharded linear or output projection:

```text
b n c/tp, o c/tp -> b n o
```

The contracted `c/tp` axis disappears, so the local result must be all-reduced over `tp`.

MLP as two contractions:

```text
b n c, h/tp c -> b n h/tp
b n h/tp, c h/tp -> b n c
```

Attention Q/K/V projections:

```text
b l c, q/tp c -> b l q/tp
b l c, k/tp c -> b l k/tp
b l c, v/tp c -> b l v/tp
```

Attention output projection:

```text
b l v/tp, c v/tp -> b l c
```

Window attention can stay local per tensor-parallel shard when heads are sharded:

```text
b heads/tp l d, b heads/tp m d -> b heads/tp l m
b heads/tp l m, b heads/tp m d -> b heads/tp l d
```

## Autograd-Only Communication

SciGPT uses identity-forward/all-reduce-backward before tensor-parallel linears. The forward tensor layout does not change, so this needs an operation annotation rather than plain axis notation.

Possible extension:

```text
b n c -> b n c :: backward_reduce(tp)
```

Alternative compact form:

```text
b n c -> b n c / grad:tp
```

Meaning:

- forward: identity
- backward: all-reduce over `tp`

The reverse direction, all-reduce-forward/identity-backward, should eventually distinguish sharded axes from partial values:

```text
b n c / partial(tp) -> b n c
```

## Reduce-Scatter Patterns

Reduce-scatter forward / all-gather backward:

```text
b n c -> b n/tp c :: reduce_scatter(tp, n)
```

All-gather forward / reduce-scatter backward:

```text
b n/tp c -> b n c :: backward_reduce_scatter(tp, n)
```

These require a notion of partial values, since current gather/split semantics use split-backward rather than reduce-scatter-backward.

## Distributed Roll

Shifted-window attention needs cyclic shifts across spatial shards. This is a neighbor-exchange operation and should probably be a named operation instead of overloaded einsum syntax.

Possible API:

```python
einroll("b t h/sp1 w/sp2 c", x, shifts={"h": -sh, "w": -sw}, mesh=mesh)
```

Possible compact notation:

```text
roll[b t h/sp1 w/sp2 c; h=-s1,w=-s2]
```

## Distributed Transpose / Repartition

Repartitioning from one sharded axis to another over the same mesh dimension can be expressed directly:

```text
b h/sp1 w c -> b h w/sp1 c
```

Swapping ownership of two spatial mesh dimensions:

```text
b h/sp1 w/sp2 c -> b h/sp2 w/sp1 c
```

These require all-to-all-style redistribution.

## Compound Groups

SciGPT uses groups such as `sp1-sp2`, `tp-sp1-sp2`, and `dp-sp1-sp2`.

Axis-wise notation can represent many compound operations:

```text
b h/sp1 w/sp2 c -> b h w c
```

Scalar or loss reductions over compound groups need partial-value notation:

```text
loss partial(sp1-sp2) -> loss
```

## Parameter Metadata

Parameter sharding can be represented by tensor-axis notation:

```text
o/tp c
o c/tp
b t h/sp1 w/sp2 c
```

Shared/reduced parameter metadata likely needs module-level annotations:

```text
[o c] shared(tp-sp1-sp2) reduce(sp1-sp2)
[o/tp c] shared(sp1-sp2) reduce(sp1-sp2)
[o c/tp] shared(sp1-sp2) reduce(sp1-sp2)
```

## First Implementation Targets

1. Multi-axis unary split/gather using existing notation.
2. Identity-forward/all-reduce-backward mapping as a low-level primitive.
3. Reduce-scatter mappings as low-level primitives.
4. Repartition from `axis/p -> other_axis/p` using all-to-all.
5. Distributed roll as a separate named operation.
6. Higher-level tensor-parallel linear examples/tests using contraction notation.

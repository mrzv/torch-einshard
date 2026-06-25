# torch-einshard

`torch-einshard` expresses local and distributed PyTorch tensor operations with
einsum-like axis and sharding notation.

```{toctree}
:maxdepth: 2
:caption: Start Here

getting-started
notation
mesh-and-shapes
```

```{toctree}
:maxdepth: 2
:caption: User Guide

user-guide/local
user-guide/distributed-unary
user-guide/distributed-binary
user-guide/tensor-parallel
user-guide/fft
user-guide/halo-window-conv
user-guide/roll
user-guide/params
user-guide/policies
```

```{toctree}
:maxdepth: 2
:caption: Reference

reference/api
reference/autograd-mappings
reference/limitations
```

```{toctree}
:maxdepth: 2
:caption: Development

development/testing
development/performance
development/roadmap
development/symbolic-engine
development/parameter-inference
development/distributed-operations
development/fft-plan
```

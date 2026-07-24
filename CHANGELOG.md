# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Add user-visible changes to the `Unreleased` section as they are made, under
`Added`, `Changed`, `Deprecated`, `Removed`, `Fixed`, or `Security` headings.

## [Unreleased]

### Added

- Read the Docs build configuration for publishing the Sphinx documentation.

### Fixed

- Resolve Read the Docs dependencies from PyPI while retaining CPU-only
  PyTorch.

## [1.0] - 2026-07-24

### Added

- Initial public release.
- Einsum-like notation for local and distributed PyTorch tensor operations.
- Tensor sharding, partial-value, factored-axis, family, and ellipsis notation.
- Selected distributed split, gather, repartition, all-reduce, and
  tensor-parallel contraction patterns.
- Named-axis FFT, halo exchange, window, convolution, and roll operations.
- Optimization policy diagnostics and parameter metadata helpers.
- PyPI package metadata for the BSD-3-Clause-LBNL license.

[Unreleased]: https://github.com/lbnl-sciml/torch-einshard/compare/8e1a263fb6517338550735df6e58802715dfe8f2...HEAD
[1.0]: https://pypi.org/project/torch-einshard/1.0/

# AGENTS.md

## Repository Overview

`torch-einshard` is a Python package for expressing local and distributed PyTorch tensor operations with einsum-like sharding notation.

Core code lives in `src/torch_einshard/`:

- `__init__.py` exposes `einshard`, parses sharding notation, and dispatches to local or distributed execution.
- `grammar.py` defines the Parsley grammar for sharding expressions.
- `sharding.py` defines `Axis` and `Axes` data structures.
- `einsum.py` translates parsed axis notation into `torch.einsum` calls.
- `distributed.py` implements limited 1D distributed split/gather/all-reduce cases.
- `mappings.py` contains custom autograd mappings for distributed collectives.
- `helpers.py` contains process-group setup and tensor collective helpers.
- `mesh.py` is currently incomplete and should be treated cautiously.

Tests are in `tests/`. Examples are in `examples/`. The project uses Python 3.12, `uv`, PyTorch, NumPy, Parsley, and pytest.

## Development Instructions

- Prefer small, focused changes that match the existing style.
- Keep public behavior simple and explicit; do not add compatibility layers unless there is a concrete need.
- The README is currently empty, so source files and tests are the best references for behavior.
- Distributed tests are intended to run through `torchrun`; use `./run_tests.sh` when verifying the full suite.
- Be careful with distributed process-group initialization and PyTorch `DeviceMesh` behavior when changing tests or examples.

## Version Control

This repository uses Jujutsu (`jj`) for version control.

- Make your own logical commits with `jj`; do not leave unrelated changes mixed together.
- Inspect state with `jj status` before and after changes.
- Start each AI-authored commit title with `[ai]`.
- At the end of each AI-authored commit message, add a line recording the model used, formatted exactly like `Model: <model>`.

Model: openai/gpt-5.5

#!/usr/bin/env bash
set -euo pipefail

NPROC="${NPROC:-8}"

uv run python benchmarks/bench_local.py "$@"
uv run torchrun --nproc-per-node "${NPROC}" benchmarks/bench_distributed.py "$@"
uv run torchrun --nproc-per-node "${NPROC}" benchmarks/bench_ddp_hook.py --mode per-spec "$@"
uv run torchrun --nproc-per-node "${NPROC}" benchmarks/bench_ddp_hook.py --mode combined "$@"
uv run torchrun --nproc-per-node "${NPROC}" benchmarks/bench_ddp_hook.py --mode mixed "$@"

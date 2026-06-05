# Performance Test Plan

This repository should keep correctness tests as the default test suite and add performance coverage as explicit opt-in benchmarks. Performance tests should be stable enough to catch large regressions without making normal CI noisy.

## Goals

- Track Python overhead for parser/family/cache-heavy local operations.
- Track distributed collective choices for common tensor-parallel and spatial-parallel patterns.
- Compare DDP gradient reduction modes, especially the per-spec reduction path against the combined compound-group fast path.
- Keep benchmark output machine-readable so results can be compared across commits.

## Structure

Add a `benchmarks/` directory with small standalone scripts rather than pytest tests initially:

```text
benchmarks/
  bench_local.py
  bench_distributed.py
  bench_ddp_hook.py
  common.py
```

Each script should print JSON Lines records with at least:

```json
{"name": "local_mlp_einshard", "device": "cpu", "world_size": 1, "median_ms": 0.0123, "iters": 1000}
```

Use `time.perf_counter()` for CPU timings and CUDA events plus `torch.cuda.synchronize()` for CUDA timings. Avoid adding `pytest-benchmark` until there is a clear CI workflow that benefits from it.

## Local Benchmarks

Run with `uv run python benchmarks/bench_local.py`.

Cases:

- Tiny local `einshard("... c, c o -> ... o")` versus `torch.einsum` to track parser/cache overhead.
- Axis-family window partition and reverse for 2D and 3D shapes.
- Factored-axis pack/unpack with and without `sizes` inference.
- `einroll` fallback-free local roll patterns.

Report both absolute time and overhead versus the equivalent PyTorch primitive when practical.

## Distributed Benchmarks

Run with `torchrun`, for example:

```bash
uv run torchrun --nproc-per-node 8 benchmarks/bench_distributed.py --device cpu
```

Cases:

- Unary split/gather round trips on even and uneven shard sizes.
- Same-mesh axis-to-axis repartition with metadata, compared to gather/split fallback.
- Pure ownership swaps, for example `h/sp1 w/sp2 -> h/sp2 w/sp1`.
- Tensor-parallel MLP contractions with ellipses.
- `einroll` shard exchange with `shapes` metadata versus gather/roll/split fallback.

Each benchmark should run a warmup phase, time multiple iterations, and report median plus p90 if the implementation stores per-iteration timings.

## DDP Hook Benchmarks

Run with `torchrun`:

```bash
uv run torchrun --nproc-per-node 8 benchmarks/bench_ddp_hook.py --mode per-spec
uv run torchrun --nproc-per-node 8 benchmarks/bench_ddp_hook.py --mode combined
```

Cases:

- Uniform `reduce=("sp",)` specs where the combined fast path should apply.
- Mixed buckets where the combined option should fall back.
- Bucket-size sensitivity by varying `bucket_cap_mb`.

Primary metric: backward step wall time after warmup. Secondary metrics: number of buckets and whether the combined path was eligible for each bucket. If needed, expose lightweight debug counters from the hook state rather than relying on profiler traces.

## CI Policy

- Do not run performance benchmarks in the default `uv run pytest` or `./run_tests.sh` suites.
- Add a manual CI job or script later, for example `./run_benchmarks.sh`, once baseline hardware is known.
- Store baseline JSONL artifacts outside the repository or under a clearly named `benchmarks/baselines/` directory only if they are stable and useful.

## Regression Thresholds

Start without hard pass/fail thresholds. After collecting several runs on the same host, add optional comparison tooling:

```bash
uv run python benchmarks/compare.py baseline.jsonl current.jsonl --max-regression 1.25
```

Only enforce thresholds for low-variance cases such as local parser overhead. Distributed timings should remain informational until variance is understood.

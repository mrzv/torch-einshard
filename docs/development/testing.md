# Testing

Local and single-process tests:

```sh
uv run pytest
```

Full distributed test suite:

```sh
./run_tests.sh
```

`run_tests.sh` uses `torchrun --nproc-per-node 8`.

Performance benchmark structure and CI policy are described in
`performance.md`. Benchmarks should remain opt-in and separate from the
correctness test suites.

Quick local benchmark smoke test:

```sh
uv run python benchmarks/bench_local.py --warmup 1 --iters 2 --size small --device cpu
```

Full benchmark runner:

```sh
NPROC=8 ./run_benchmarks.sh --device cpu --size small --output results.jsonl
```

import argparse
import json
import statistics
import time
from contextlib import nullcontext

import torch
import torch.distributed as dist


def add_common_args(parser):
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--size", choices=("small", "medium", "large"), default="small")
    return parser


def common_parser(description):
    return add_common_args(argparse.ArgumentParser(description=description))


def resolve_device(device):
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    return torch.device(device)


def init_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def init_distributed(device):
    if dist.is_initialized():
        return
    backend = "nccl" if device.type == "cuda" else "gloo"
    dist.init_process_group(backend=backend)


def rank0():
    return not dist.is_initialized() or dist.get_rank() == 0


def world_size():
    return dist.get_world_size() if dist.is_initialized() else 1


def rank():
    return dist.get_rank() if dist.is_initialized() else 0


def sync(device):
    if dist.is_initialized():
        dist.barrier()
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def percentile(values, pct):
    if not values:
        return 0.0
    values = sorted(values)
    index = min(len(values) - 1, max(0, round((len(values) - 1) * pct)))
    return values[index]


def time_call(fn, *, warmup, iters, device):
    with torch.no_grad():
        for _ in range(warmup):
            fn()
        sync(device)

        times = []
        if device.type == "cuda":
            for _ in range(iters):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                fn()
                end.record()
                torch.cuda.synchronize(device)
                times.append(start.elapsed_time(end))
        else:
            for _ in range(iters):
                start = time.perf_counter()
                fn()
                times.append((time.perf_counter() - start) * 1000.0)
        sync(device)
    return times


def summarize_times(times):
    return {
        "median_ms": statistics.median(times),
        "p90_ms": percentile(times, 0.9),
        "min_ms": min(times),
        "max_ms": max(times),
    }


class JsonlWriter:
    def __init__(self, output=None):
        self.output = output
        self._handle = None

    def __enter__(self):
        if self.output is not None and rank0():
            self._handle = open(self.output, "a", encoding="utf-8")
        return self

    def __exit__(self, *args):
        if self._handle is not None:
            self._handle.close()
        return False

    def write(self, record):
        if not rank0():
            return
        line = json.dumps(record, sort_keys=True)
        print(line, flush=True)
        if self._handle is not None:
            self._handle.write(line + "\n")
            self._handle.flush()


def benchmark(name, fn, *, args, device, writer, extra=None, baselines=None):
    times = time_call(fn, warmup=args.warmup, iters=args.iters, device=device)
    record = {
        "name": name,
        "device": str(device),
        "world_size": world_size(),
        "rank": rank(),
        "iters": args.iters,
        "warmup": args.warmup,
        "size": args.size,
        **summarize_times(times),
    }
    if extra:
        record.update(extra)
    if baselines:
        for baseline_name, baseline_ms in baselines.items():
            record[f"baseline_{baseline_name}_ms"] = baseline_ms
            if baseline_ms:
                record[f"ratio_to_{baseline_name}"] = record["median_ms"] / baseline_ms
    writer.write(record)
    return record


def maybe_autocast(device):
    return nullcontext()


LOCAL_LINEAR_SHAPES = {
    "small": (4, 32, 128, 256),
    "medium": (8, 128, 512, 2048),
    "large": (16, 256, 1024, 4096),
}


WINDOW_SHAPES = {
    "small": (2, 1, 16, 16, 32, (4, 4)),
    "medium": (4, 2, 64, 64, 64, (8, 8)),
    "large": (4, 2, 128, 128, 128, (8, 8)),
}

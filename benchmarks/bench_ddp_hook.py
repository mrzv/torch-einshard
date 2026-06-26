import argparse

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel

import torch_einshard as es

from common import (
    DISTRIBUTED_SHAPES,
    JsonlWriter,
    add_common_args,
    benchmark,
    destroy_distributed,
    init_distributed,
    init_seed,
    mesh_2d,
    resolve_device,
)


class TinyMLP(nn.Module):
    def __init__(self, width, depth):
        super().__init__()
        layers = []
        for _ in range(depth):
            layers.append(nn.Linear(width, width))
            layers.append(nn.GELU())
        layers.append(nn.Linear(width, width))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def attach_states(model, mode):
    for name, param in model.named_parameters():
        if mode == "mixed" and name.endswith("bias"):
            continue
        layout = "out in" if param.ndim == 2 else "out"
        es.register_parameter_layout(param, layout, grad="sp")


def build_model(args, device):
    cfg = DISTRIBUTED_SHAPES[args.size]
    width = 128 if args.size == "small" else min(cfg["hidden"], 1024)
    depth = 2 if args.size == "small" else 6
    model = TinyMLP(width, depth).to(device)
    return model, width


def main():
    parser = add_common_args(argparse.ArgumentParser(description="DDP hook benchmarks"))
    parser.add_argument("--mode", choices=("per-spec", "combined", "mixed"), default="per-spec")
    parser.add_argument("--bucket-cap-mb", type=float, default=25.0)
    args = parser.parse_args()

    device = resolve_device(args.device)
    init_seed(args.seed)
    init_distributed(device)
    mesh = es.wrap_mesh(mesh_2d(device, names=("dp", "sp")))

    model, width = build_model(args, device)
    attach_states(model, "mixed" if args.mode == "mixed" else "uniform")
    ddp = DistributedDataParallel(
        model,
        process_group=mesh["dp"].get_group(),
        bucket_cap_mb=args.bucket_cap_mb,
        gradient_as_bucket_view=True,
    )
    hook_kwargs = {"ddp_group": "dp"}
    if args.mode in ("combined", "mixed"):
        hook_kwargs.update({"combined_reduce_group": "dp-sp", "combined_reduce": "sp"})
    es.register_grad_reduction_hook_(ddp, mesh, **hook_kwargs)

    batch = 16 if args.size == "small" else 64
    x = torch.randn(batch, width, device=device)

    def step():
        ddp.zero_grad(set_to_none=True)
        loss = ddp(x).square().mean()
        loss.backward()
        return loss

    with JsonlWriter(args.output) as writer:
        benchmark(
            f"ddp_hook_{args.mode}",
            step,
            args=args,
        device=device,
        writer=writer,
        extra={"mode": args.mode, "bucket_cap_mb": args.bucket_cap_mb, "width": width, "batch": batch},
        grad=True,
    )

    destroy_distributed()


if __name__ == "__main__":
    main()

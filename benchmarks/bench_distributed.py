import warnings

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

import torch_einshard as es

from common import (
    DISTRIBUTED_SHAPES,
    JsonlWriter,
    benchmark,
    common_parser,
    destroy_distributed,
    init_distributed,
    init_seed,
    mesh_1d,
    mesh_2d,
    resolve_device,
)


def split_local(full, sections, dim, group):
    return torch.split(full, sections, dim=dim)[dist.get_rank(group)].contiguous()


def bench_split_gather(args, device, writer, mesh):
    group = mesh["dp"].get_group()
    size = dist.get_world_size(group)
    rank = dist.get_rank(group)
    cfg = DISTRIBUTED_SHAPES[args.size]
    rows = cfg["rows"] + 1
    shapes = es.helpers.compute_split_shapes(rows, size)
    x = torch.randn(rows, cfg["cols"], device=device)
    extra = {"global_shape": [rows, cfg["cols"]], "local_rows": shapes[rank]}

    def round_trip():
        z = es.einshard("a b -> a/dp b", x, mesh=mesh, shapes=shapes)
        return es.einshard("a/dp b -> a b", z, mesh=mesh, shapes=shapes)

    benchmark("distributed_split_gather_round_trip", round_trip, args=args, device=device, writer=writer, extra=extra)


def bench_repartition(args, device, writer, mesh):
    group = mesh["dp"].get_group()
    size = dist.get_world_size(group)
    cfg = DISTRIBUTED_SHAPES[args.size]
    rows = cfg["rows"]
    cols = cfg["cols"]
    row_shapes = es.helpers.compute_split_shapes(rows, size)
    col_shapes = es.helpers.compute_split_shapes(cols, size)
    full = torch.randn(rows, cols, device=device)
    x = split_local(full, row_shapes, 0, group)
    extra = {"global_shape": [rows, cols]}

    benchmark(
        "distributed_repartition_axis_metadata",
        lambda: es.einshard(
            "a/dp b -> a b/dp",
            x,
            mesh=mesh,
            shapes={"dp": {"a": row_shapes, "b": col_shapes}},
        ),
        args=args,
        device=device,
        writer=writer,
        extra=extra,
    )

    def fallback():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return es.einshard("a/dp b -> a b/dp", x, mesh=mesh)

    benchmark("distributed_repartition_axis_fallback", fallback, args=args, device=device, writer=writer, extra=extra)


def bench_ownership_swap(args, device, writer, mesh):
    sp_group = mesh["sp1"].get_group()
    dp_group = mesh["sp2"].get_group()
    sp1_rank = dist.get_rank(sp_group)
    sp2_rank = dist.get_rank(dp_group)
    cfg = DISTRIBUTED_SHAPES[args.size]
    rows = cfg["rows"]
    cols = cfg["cols"]
    shapes = {
        "sp1": {
            "h": es.helpers.compute_split_shapes(rows, dist.get_world_size(sp_group)),
            "w": es.helpers.compute_split_shapes(cols, dist.get_world_size(dp_group)),
        },
        "sp2": {
            "h": es.helpers.compute_split_shapes(rows, dist.get_world_size(sp_group)),
            "w": es.helpers.compute_split_shapes(cols, dist.get_world_size(dp_group)),
        },
    }
    full = torch.randn(2, rows, cols, 8, device=device)
    x = torch.split(full, shapes["sp1"]["h"], dim=1)[sp1_rank]
    x = torch.split(x, shapes["sp2"]["w"], dim=2)[sp2_rank].contiguous()

    benchmark(
        "distributed_ownership_swap",
        lambda: es.einshard("n h/sp1 w/sp2 c -> n h/sp2 w/sp1 c", x, mesh=mesh, shapes=shapes),
        args=args,
        device=device,
        writer=writer,
        extra={"global_shape": [2, rows, cols, 8]},
    )


def bench_tp_mlp(args, device, writer, mesh):
    group = mesh["tp"].get_group()
    cfg = DISTRIBUTED_SHAPES[args.size]
    hidden = cfg["hidden"]
    shapes = es.helpers.compute_split_shapes(hidden, dist.get_world_size(group))
    local_hidden = shapes[dist.get_rank(group)]
    x = torch.randn(cfg["batch"], cfg["seq"], cfg["cols"], device=device)
    w = torch.randn(local_hidden, cfg["cols"], device=device)

    benchmark(
        "distributed_tp_mlp_contract",
        lambda: es.einshard("... c, h/tp c -> ... h/tp", x, w, mesh=mesh),
        args=args,
        device=device,
        writer=writer,
        extra={"input_shape": list(x.shape), "local_hidden": local_hidden},
    )


def bench_roll(args, device, writer, mesh):
    group = mesh["dp"].get_group()
    cfg = DISTRIBUTED_SHAPES[args.size]
    rows = cfg["rows"] + 1
    shapes = es.helpers.compute_split_shapes(rows, dist.get_world_size(group))
    full = torch.randn(rows, cfg["cols"], device=device)
    x = split_local(full, shapes, 0, group)

    benchmark(
        "distributed_roll_metadata",
        lambda: es.einroll("a/dp b", x, {"a": 7}, mesh=mesh, shapes=shapes),
        args=args,
        device=device,
        writer=writer,
        extra={"global_shape": [rows, cfg["cols"]]},
    )

    even_rows = cfg["rows"]
    even_shapes = es.helpers.compute_split_shapes(even_rows, dist.get_world_size(group))
    even_full = torch.randn(even_rows, cfg["cols"], device=device)
    even_x = split_local(even_full, even_shapes, 0, group)

    def fallback():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return es.einroll("a/dp b", even_x, {"a": 7}, mesh=mesh)

    benchmark(
        "distributed_roll_fallback",
        fallback,
        args=args,
        device=device,
        writer=writer,
        extra={"global_shape": [even_rows, cfg["cols"]]},
    )


def main():
    parser = common_parser("Distributed torch-einshard benchmarks")
    args = parser.parse_args()
    device = resolve_device(args.device)
    init_seed(args.seed)
    init_distributed(device)
    init_seed(args.seed + dist.get_rank())

    with JsonlWriter(args.output) as writer:
        dp_mesh = mesh_1d(device, "dp")
        bench_split_gather(args, device, writer, dp_mesh)
        bench_repartition(args, device, writer, dp_mesh)
        bench_roll(args, device, writer, dp_mesh)

        tp_mesh = mesh_1d(device, "tp")
        bench_tp_mlp(args, device, writer, tp_mesh)

        size = dist.get_world_size()
        if size == 1:
            swap_shape = (1, 1, 1)
        elif size % 4 == 0:
            swap_shape = (size // 4, 2, 2)
        else:
            writer.write({
                "name": "distributed_ownership_swap",
                "device": str(device),
                "world_size": size,
                "rank": dist.get_rank(),
                "skipped": True,
                "reason": "requires world size 1 or divisible by 4",
            })
            destroy_distributed()
            return
        swap_mesh = init_device_mesh(device.type, swap_shape, mesh_dim_names=("dp", "sp1", "sp2"))
        bench_ownership_swap(args, device, writer, swap_mesh)

    destroy_distributed()


if __name__ == "__main__":
    main()

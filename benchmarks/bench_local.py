import torch
import torch.nn.functional as F

import torch_einshard as es

from common import (
    LOCAL_LINEAR_SHAPES,
    WINDOW_SHAPES,
    JsonlWriter,
    benchmark,
    common_parser,
    init_seed,
    resolve_device,
)


def bench_linear(args, device, writer):
    batch, seq, in_features, out_features = LOCAL_LINEAR_SHAPES[args.size]
    x = torch.randn(batch, seq, in_features, device=device)
    w = torch.randn(out_features, in_features, device=device)
    w_t = w.t().contiguous()
    shape = {"batch": batch, "seq": seq, "in": in_features, "out": out_features}

    records = {}
    records["f_linear"] = benchmark(
        "local_linear_f_linear",
        lambda: F.linear(x, w),
        args=args,
        device=device,
        writer=writer,
        extra={"shape": shape},
    )
    records["matmul_oc"] = benchmark(
        "local_linear_matmul_oc",
        lambda: x @ w.t(),
        args=args,
        device=device,
        writer=writer,
        extra={"shape": shape},
        baselines={"f_linear": records["f_linear"]["median_ms"]},
    )
    records["matmul_co"] = benchmark(
        "local_linear_matmul_co",
        lambda: x @ w_t,
        args=args,
        device=device,
        writer=writer,
        extra={"shape": shape},
        baselines={"f_linear": records["f_linear"]["median_ms"]},
    )
    records["einsum_oc"] = benchmark(
        "local_linear_einsum_oc",
        lambda: torch.einsum("...c,oc->...o", x, w),
        args=args,
        device=device,
        writer=writer,
        extra={"shape": shape},
        baselines={"f_linear": records["f_linear"]["median_ms"]},
    )
    records["einsum_co"] = benchmark(
        "local_linear_einsum_co",
        lambda: torch.einsum("...c,co->...o", x, w_t),
        args=args,
        device=device,
        writer=writer,
        extra={"shape": shape},
        baselines={"f_linear": records["f_linear"]["median_ms"]},
    )
    benchmark(
        "local_linear_einshard_oc",
        lambda: es.einshard("... c, o c -> ... o", x, w),
        args=args,
        device=device,
        writer=writer,
        extra={"shape": shape},
        baselines={
            "f_linear": records["f_linear"]["median_ms"],
            "einsum": records["einsum_oc"]["median_ms"],
        },
    )
    benchmark(
        "local_linear_einshard_co",
        lambda: es.einshard("... c, c o -> ... o", x, w_t),
        args=args,
        device=device,
        writer=writer,
        extra={"shape": shape},
        baselines={
            "f_linear": records["f_linear"]["median_ms"],
            "einsum": records["einsum_co"]["median_ms"],
        },
    )


def bench_windows(args, device, writer):
    b, t, h, w, c, window = WINDOW_SHAPES[args.size]
    x = torch.randn(b, t, h, w, c, device=device)
    families = {"spatial": ("h", "w"), "window": ("wh", "ww")}
    sizes = {"window": window}
    shape = {"b": b, "t": t, "h": h, "w": w, "c": c, "window": window}

    def partition_view():
        return x.reshape(b, t, h // window[0], window[0], w // window[1], window[1], c).permute(
            0, 2, 4, 1, 3, 5, 6
        ).reshape(b * (h // window[0]) * (w // window[1]), t, window[0], window[1], c)

    partition = benchmark(
        "local_window_partition_view",
        partition_view,
        args=args,
        device=device,
        writer=writer,
        extra={"shape": shape},
    )
    benchmark(
        "local_window_partition_einshard",
        lambda: es.einshard(
            "b t [*spatial *window] c -> (b *spatial) t *window c",
            x,
            families=families,
            sizes=sizes,
        ),
        args=args,
        device=device,
        writer=writer,
        extra={"shape": shape},
        baselines={"view": partition["median_ms"]},
    )

    windows = partition_view().contiguous()
    img_by_win = (h // window[0], w // window[1])

    def reverse_view():
        return windows.reshape(b, img_by_win[0], img_by_win[1], t, window[0], window[1], c).permute(
            0, 3, 1, 4, 2, 5, 6
        ).reshape(b, t, h, w, c)

    reverse = benchmark(
        "local_window_reverse_view",
        reverse_view,
        args=args,
        device=device,
        writer=writer,
        extra={"shape": shape},
    )
    benchmark(
        "local_window_reverse_einshard",
        lambda: es.einshard(
            "(b *spatial) t *window c -> b t [*spatial *window] c",
            windows,
            families=families,
            sizes={"spatial": img_by_win, "window": window},
        ),
        args=args,
        device=device,
        writer=writer,
        extra={"shape": shape},
        baselines={"view": reverse["median_ms"]},
    )


def bench_factored(args, device, writer):
    b, t, h, w, c, window = WINDOW_SHAPES[args.size]
    x = torch.randn(b, t, h, w, c, device=device)
    shape = {"b": b, "t": t, "h": h, "w": w, "c": c, "window": window}

    baseline = benchmark(
        "local_factored_unpack_view",
        lambda: x.reshape(b, t, h // window[0], window[0], w // window[1], window[1], c),
        args=args,
        device=device,
        writer=writer,
        extra={"shape": shape},
    )
    benchmark(
        "local_factored_unpack_einshard",
        lambda: es.einshard(
            "b t (h wh) (w ww) c -> b t h wh w ww c",
            x,
            sizes={"wh": window[0], "ww": window[1]},
        ),
        args=args,
        device=device,
        writer=writer,
        extra={"shape": shape},
        baselines={"view": baseline["median_ms"]},
    )

    unpacked = x.reshape(b, t, h // window[0], window[0], w // window[1], window[1], c)
    pack = benchmark(
        "local_factored_pack_view",
        lambda: unpacked.reshape(b, t, h, w, c),
        args=args,
        device=device,
        writer=writer,
        extra={"shape": shape},
    )
    benchmark(
        "local_factored_pack_einshard",
        lambda: es.einshard("b t h wh w ww c -> b t (h wh) (w ww) c", unpacked),
        args=args,
        device=device,
        writer=writer,
        extra={"shape": shape},
        baselines={"view": pack["median_ms"]},
    )


def bench_roll(args, device, writer):
    _, _, h, w, c, _ = WINDOW_SHAPES[args.size]
    x = torch.randn(4, h, w, c, device=device)
    shape = {"b": 4, "h": h, "w": w, "c": c}
    baseline = benchmark(
        "local_roll_torch",
        lambda: torch.roll(x, shifts=(3, -5), dims=(1, 2)),
        args=args,
        device=device,
        writer=writer,
        extra={"shape": shape},
    )
    benchmark(
        "local_roll_einroll",
        lambda: es.einroll("b h w c", x, {"h": 3, "w": -5}),
        args=args,
        device=device,
        writer=writer,
        extra={"shape": shape},
        baselines={"torch_roll": baseline["median_ms"]},
    )


def main():
    parser = common_parser("Local torch-einshard benchmarks")
    args = parser.parse_args()
    device = resolve_device(args.device)
    init_seed(args.seed)

    with JsonlWriter(args.output) as writer:
        bench_linear(args, device, writer)
        bench_windows(args, device, writer)
        bench_factored(args, device, writer)
        bench_roll(args, device, writer)


if __name__ == "__main__":
    main()

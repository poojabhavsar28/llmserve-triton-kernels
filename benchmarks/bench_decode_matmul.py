#!/usr/bin/env python
"""Decode-matmul shootout (needs a CUDA GPU + Triton): the same y = x @ W^T four ways.

  cuBLAS fp16        F.linear                         reads 2 bytes/weight
  Triton fp16        kernels/fp16_linear.py           reads 2 bytes/weight   (control experiment)
  torch int4         dequantise to fp16, then matmul  materialises a full fp16 weight first
  Triton int4        quant/triton_int4.py             reads ~0.5 byte/weight

Reading the table: Triton-fp16 vs cuBLAS tells you how good your Triton kernel structure is. Triton-int4
vs Triton-fp16 tells you what dequantisation costs on top of that. The achieved-GB/s columns say how far each
is from the GPU's memory bandwidth (pass --peak-gbs to also get a percentage).

Weights are rotated across copies totalling >= 128 MiB so each call reads from DRAM, not the L2 cache.

    python benchmarks/bench_decode_matmul.py [--m 1] [--tune] [--peak-gbs 192]
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import torch
import torch.nn.functional as F

ROTATE_BYTES = 128 * 2**20
SHAPES = [(2048, 2048), (5632, 2048), (2048, 5632), (32000, 2048)]  # q/o, gate/up, down, lm_head
GROUP = 128


def time_rotating(closures, iters=200, warmup=30) -> float:
    """Mean microseconds per call, cycling through closures (each bound to its own weight copy)."""
    n = len(closures)
    for i in range(warmup):
        closures[i % n]()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for i in range(iters):
        closures[i % n]()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1e3


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=1, help="activation rows (decode batch size)")
    ap.add_argument("--tune", action="store_true", help="let the fp16 kernel pick a tile config per shape")
    ap.add_argument("--peak-gbs", type=float, default=None, help="your GPU's peak memory bandwidth, GB/s")
    args = ap.parse_args()
    if args.tune:
        os.environ["LLMSERVE_TUNE"] = "1"

    from llmserve.kernels import fp16_linear, tuned_configs
    from llmserve.quant.int4 import dequantize_int4, quantize_int4
    from llmserve.quant.triton_int4 import HAS_TRITON, int4_matmul

    if not (torch.cuda.is_available() and HAS_TRITON):
        print("skipped: needs CUDA + Triton (Linux, or `pip install triton-windows`).")
        return 0
    print(f"GPU: {torch.cuda.get_device_name(0)} | rows M={args.m} | int4 group={GROUP} | tune={args.tune}\n")
    cols = ["N x K", "cuBLAS fp16 us", "Triton fp16 us", "T-fp16 vs cuBLAS", "torch-int4 us",
            "Triton-int4 us", "GB/s cuBLAS", "GB/s T-fp16", "GB/s T-int4", "fp16 max err"]
    print("| " + " | ".join(cols) + " |")
    print("|" + "---|" * len(cols))

    for N, K in SHAPES:
        w16 = torch.randn(N, K, device="cuda", dtype=torch.float16)
        x = torch.randn(args.m, K, device="cuda", dtype=torch.float16)
        packed, scale, zero = quantize_int4(w16, GROUP)
        scale = scale.half()

        fp16_bytes = N * K * 2
        int4_bytes = packed.numel() + scale.numel() * 2 + zero.numel()
        c16 = max(2, math.ceil(ROTATE_BYTES / fp16_bytes))
        c4 = max(2, math.ceil(ROTATE_BYTES / int4_bytes))
        ws = [w16.clone() for _ in range(c16)]
        qs = [(packed.clone(), scale.clone(), zero.clone()) for _ in range(c4)]

        t_cublas = time_rotating([lambda w=w: F.linear(x, w) for w in ws])
        t_tri16 = time_rotating([lambda w=w: fp16_linear(x, w) for w in ws])
        t_torch4 = time_rotating([
            lambda q=q: F.linear(x, dequantize_int4(q[0], q[1], q[2], GROUP, torch.float16)) for q in qs])
        t_tri4 = time_rotating([lambda q=q: int4_matmul(x, q[0], q[1], q[2], GROUP) for q in qs])
        err = (fp16_linear(x, w16) - F.linear(x, w16)).abs().max().item()

        gbs = lambda nbytes, us: nbytes / (us * 1e-6) / 1e9  # noqa: E731
        g_c, g_t, g_4 = gbs(fp16_bytes, t_cublas), gbs(fp16_bytes, t_tri16), gbs(int4_bytes, t_tri4)
        pct = (lambda g: f" ({g / args.peak_gbs:.0%})") if args.peak_gbs else (lambda g: "")
        print(f"| {N}x{K} | {t_cublas:.1f} | {t_tri16:.1f} | {t_cublas / t_tri16:.2f}x | {t_torch4:.1f} | "
              f"{t_tri4:.1f} | {g_c:.0f}{pct(g_c)} | {g_t:.0f}{pct(g_t)} | {g_4:.0f}{pct(g_4)} | {err:.4f} |")
        del ws, qs

    if args.tune:
        print("\ntuned tile configs (N, K) -> config:")
        for (n, k, *_), cfg in sorted(tuned_configs().items()):
            print(f"  {n}x{k}: {cfg}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

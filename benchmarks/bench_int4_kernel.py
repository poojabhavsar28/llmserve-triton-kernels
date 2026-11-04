#!/usr/bin/env python
"""Phase 6 micro-benchmark (needs a CUDA GPU + Triton): the INT4 decode matmul three ways.

  fp16 linear        -> reads 2 bytes/weight                      (reference speed)
  torch int4 path    -> dequantise to fp16, then matmul           (what QuantLinear4 does without the kernel)
  triton int4 kernel -> fused dequant+matmul, reads 0.5 byte/weight
"""

from __future__ import annotations

import sys

import torch
import torch.nn.functional as F

from llmserve.quant.int4 import dequantize_int4, quantize_int4
from llmserve.quant.triton_int4 import HAS_TRITON, int4_matmul

SHAPES = [(2048, 2048), (5632, 2048), (2048, 5632), (4096, 4096)]  # (N, K): TinyLlama / 7B-ish
GROUP, M = 128, 1


def timeit(fn, iters=200, warmup=30):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1e3  # microseconds


def main():
    if not (torch.cuda.is_available() and HAS_TRITON):
        print("skipped: needs CUDA + Triton (Linux, or `pip install triton-windows`).")
        return 0
    print(f"GPU: {torch.cuda.get_device_name(0)} | decode rows M={M} | group={GROUP}\n")
    print("| N x K | fp16 us | torch-int4 us | triton-int4 us | triton vs torch-int4 | triton vs fp16 | max abs err |")
    print("|---|---|---|---|---|---|---|")
    for N, K in SHAPES:
        w = torch.randn(N, K, device="cuda", dtype=torch.float16)
        x = torch.randn(M, K, device="cuda", dtype=torch.float16)
        packed, scale, zero = quantize_int4(w, GROUP)
        scale = scale.half()
        t_fp16 = timeit(lambda: F.linear(x, w))
        t_torch = timeit(lambda: F.linear(x, dequantize_int4(packed, scale, zero, GROUP, torch.float16)))
        t_tri = timeit(lambda: int4_matmul(x, packed, scale, zero, GROUP))
        ref = F.linear(x, dequantize_int4(packed, scale, zero, GROUP, torch.float16))
        err = (int4_matmul(x, packed, scale, zero, GROUP) - ref).abs().max().item()
        print(f"| {N}x{K} | {t_fp16:.1f} | {t_torch:.1f} | {t_tri:.1f} | "
              f"{t_torch / t_tri:.2f}x | {t_fp16 / t_tri:.2f}x | {err:.4f} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())

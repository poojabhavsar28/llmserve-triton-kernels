#!/usr/bin/env python
"""Phase 6: profile decode steps with torch.profiler and export a Chrome/Perfetto trace.

    python benchmarks/profile_decode.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --quant fp16
    python benchmarks/profile_decode.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --quant int4

Open results/trace-<quant>.json at https://ui.perfetto.dev (or chrome://tracing). See docs/PROFILING.md.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, record_function

from llmserve.kvcache import KVCache
from llmserve.quant import quantize_model
from llmserve.weights import load_model, random_model

DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="tiny")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", choices=list(DTYPES), default=None)
    ap.add_argument("--quant", choices=["fp16", "int8", "int4"], default="fp16")
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--prompt-len", type=int, default=256)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--rows", type=int, default=15)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    dtype = DTYPES[args.dtype] if args.dtype else (
        torch.float16 if args.device.startswith("cuda") else torch.float32)
    model = (random_model(device=args.device, dtype=dtype) if args.model == "tiny"
             else load_model(args.model, device=args.device, dtype=dtype))
    if args.quant != "fp16":
        quantize_model(model, args.quant, group_size=32 if args.model == "tiny" else args.group_size)

    cfg, dev = model.config, torch.device(args.device)
    ids = torch.randint(100 % cfg.vocab_size, cfg.vocab_size, (1, args.prompt_len), device=dev)
    cache = KVCache(cfg, 1, args.prompt_len + args.steps + 8, dev, dtype)
    with torch.inference_mode():
        nxt = model(ids, cache=cache, start_pos=0, last_only=True)[:, -1].argmax(-1)[:, None]
        cache.advance(args.prompt_len)
        for _ in range(3):  # warmup (kernel compilation, allocator)
            nxt = model(nxt, cache=cache, start_pos=cache.seq_len, last_only=True)[:, -1].argmax(-1)[:, None]
            cache.advance(1)

        acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if dev.type == "cuda" else [])
        with profile(activities=acts, record_shapes=True) as prof:
            for _ in range(args.steps):
                with record_function("decode_step"):
                    logits = model(nxt, cache=cache, start_pos=cache.seq_len, last_only=True)
                    cache.advance(1)
                    nxt = logits[:, -1].argmax(-1)[:, None]
            if dev.type == "cuda":
                torch.cuda.synchronize()

    key = "self_cuda_time_total" if dev.type == "cuda" else "self_cpu_time_total"
    try:
        print(prof.key_averages().table(sort_by=key, row_limit=args.rows))
    except Exception:  # noqa: BLE001 - torch renamed sort keys across versions
        print(prof.key_averages().table(row_limit=args.rows))
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    trace = out / f"trace-{args.quant}.json"
    prof.export_chrome_trace(str(trace))
    print(f"\ntrace written to {trace}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Phase 3 + 5 benchmark: concurrent load with Poisson arrivals and mixed lengths.

Offline mode (default) drives the real engine in-process:
    python benchmarks/load_test.py --model tiny --rates 4 8 16
    policies:  sequential (batch=1) | static (drain-then-admit) | continuous (iteration-level)

HTTP mode measures the running server end to end (SSE streaming):
    python -m llmserve.server --model tiny &
    python benchmarks/load_test.py --url http://127.0.0.1:8000 --vocab 256 --rates 8

Reported per (policy, offered rate): throughput, TTFT / inter-token latency / end-to-end latency at
p50/p95/p99, preemptions (offline) and errors such as HTTP 429 (HTTP mode).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import time
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from llmserve.datatypes import SamplingParams
from llmserve.kvcache import PagedKVCache
from llmserve.scheduler import ContinuousBatchingEngine

DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}


def make_workload(n, rate, seed, prompt_range, out_range, vocab):
    rng = random.Random(seed)
    low, t, items = min(100, vocab // 4), 0.0, []
    for _ in range(n):
        t += rng.expovariate(rate)  # Poisson process: exponential inter-arrival times
        plen = rng.randint(*prompt_range)
        items.append({
            "arrival": t,
            "prompt": [rng.randrange(low, vocab) for _ in range(plen)],
            "max_new": rng.randint(*out_range),
        })
    return items


def pct(values, q):
    return float(np.percentile(values, q)) if len(values) else float("nan")


def summarize(label, rate, n, wall, ttft, itl, e2e, total_tokens, extra):
    return {
        "policy": label, "rate_req_s": rate, "requests": n,
        "tok_s": round(total_tokens / wall, 1),
        "ttft_p50_ms": round(pct(ttft, 50) * 1e3, 1), "ttft_p95_ms": round(pct(ttft, 95) * 1e3, 1),
        "ttft_p99_ms": round(pct(ttft, 99) * 1e3, 1),
        "itl_p50_ms": round(pct(itl, 50) * 1e3, 1), "itl_p95_ms": round(pct(itl, 95) * 1e3, 1),
        "itl_p99_ms": round(pct(itl, 99) * 1e3, 1),
        "e2e_p50_s": round(pct(e2e, 50), 2), "e2e_p99_s": round(pct(e2e, 99), 2),
        **extra,
    }


def run_offline(model, device, dtype, workload, label, args):
    policy, batch = ("continuous", 1) if label == "sequential" else (label, args.max_batch)
    cache = PagedKVCache(model.config, args.num_blocks, args.block_size, device, dtype)
    eng = ContinuousBatchingEngine(model, cache, max_batch_size=batch, policy=policy)
    n, i, t0 = len(workload), 0, perf_counter()
    while i < n or eng.has_unfinished():
        now = perf_counter() - t0
        while i < n and workload[i]["arrival"] <= now:
            w = workload[i]
            eng.add_request(w["prompt"], SamplingParams(max_new_tokens=w["max_new"]),
                            arrival_time=t0 + w["arrival"])
            i += 1
        if eng.has_unfinished():
            eng.step()
        else:
            time.sleep(max(0.0, workload[i]["arrival"] - (perf_counter() - t0)))
    wall = perf_counter() - t0
    reqs = list(eng.finished.values())
    ttft = [r.first_token_time - r.arrival_time for r in reqs]
    e2e = [r.finish_time - r.arrival_time for r in reqs]
    itl = [b - a for r in reqs for a, b in zip(r.token_times, r.token_times[1:])]
    total = sum(len(r.generated) for r in reqs)
    return summarize(label, args._rate, n, wall, ttft, itl, e2e, total,
                     {"preemptions": eng.stats.preemptions, "errors": 0})


async def _http(url, workload, model_name):
    import httpx

    async with httpx.AsyncClient(base_url=url, timeout=None) as client:
        t0 = perf_counter()

        async def one(w):
            await asyncio.sleep(max(0.0, w["arrival"] - (perf_counter() - t0)))
            start, times = perf_counter(), []
            body = {"model": model_name, "prompt": w["prompt"], "max_tokens": w["max_new"],
                    "temperature": 0, "stream": True}
            async with client.stream("POST", "/v1/completions", json=body) as r:
                if r.status_code != 200:
                    return {"error": r.status_code}
                async for line in r.aiter_lines():
                    if line.startswith("data: ") and line != "data: [DONE]":
                        if json.loads(line[6:])["choices"][0]["token_ids"]:
                            times.append(perf_counter())
            return {"start": start, "times": times}

        results = await asyncio.gather(*[one(w) for w in workload])
        wall = perf_counter() - t0
    return results, wall


def run_http(workload, args):
    results, wall = asyncio.run(_http(args.url, workload, "llmserve"))
    ok = [r for r in results if "times" in r and r["times"]]
    ttft = [r["times"][0] - r["start"] for r in ok]
    e2e = [r["times"][-1] - r["start"] for r in ok]
    itl = [b - a for r in ok for a, b in zip(r["times"], r["times"][1:])]
    total = sum(len(r["times"]) for r in ok)
    return summarize("http-server", args._rate, len(workload), wall, ttft, itl, e2e, total,
                     {"preemptions": "", "errors": len(results) - len(ok)})


def to_markdown(rows):
    cols = list(rows[0].keys())
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="tiny")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", choices=list(DTYPES), default=None)
    ap.add_argument("--quant", choices=["fp16", "fp16-triton", "int8", "int4"], default="fp16")
    ap.add_argument("--url", default=None, help="benchmark a running server instead of in-process")
    ap.add_argument("--vocab", type=int, default=256, help="vocab size for random prompts (HTTP mode)")
    ap.add_argument("--policies", nargs="+", default=["sequential", "static", "continuous"],
                    choices=["sequential", "static", "continuous"])
    ap.add_argument("--rates", type=float, nargs="+", default=[2.0, 4.0, 8.0], help="offered req/s")
    ap.add_argument("--num-requests", type=int, default=40)
    ap.add_argument("--prompt-range", type=int, nargs=2, default=[16, 128])
    ap.add_argument("--out-range", type=int, nargs=2, default=[16, 96])
    ap.add_argument("--max-batch", type=int, default=8)
    ap.add_argument("--num-blocks", type=int, default=1024)
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    rows = []
    if args.url:
        for rate in args.rates:
            args._rate = rate
            wl = make_workload(args.num_requests, rate, args.seed, args.prompt_range, args.out_range, args.vocab)
            rows.append(run_http(wl, args))
            print(f"done: http rate={rate}")
        meta = {"mode": "http", "url": args.url}
    else:
        from llmserve.weights import load_model, random_model

        dtype = DTYPES[args.dtype] if args.dtype else (
            torch.float16 if args.device.startswith("cuda") else torch.float32)
        model = (random_model(device=args.device, dtype=dtype) if args.model == "tiny"
                 else load_model(args.model, device=args.device, dtype=dtype))
        if args.quant != "fp16":
            from llmserve.quant import quantize_model  # lazy: only needed for quantized runs

            model = quantize_model(model, args.quant)
        vocab = model.config.vocab_size
        warm = make_workload(2, 1000.0, 99, (8, 8), (4, 4), vocab)  # CUDA/kernel warmup
        args._rate = 1000.0
        run_offline(model, args.device, dtype, warm, "continuous", args)
        for rate in args.rates:
            args._rate = rate
            wl = make_workload(args.num_requests, rate, args.seed, args.prompt_range, args.out_range, vocab)
            for label in args.policies:
                rows.append(run_offline(model, args.device, dtype, wl, label, args))
                print(f"done: {label} rate={rate}")
        meta = {"mode": "offline", "model": args.model, "quant": args.quant, "device": args.device}

    md = to_markdown(rows)
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    (out / "load_latest.md").write_text(md + "\n")
    (out / f"load-{time.strftime('%Y%m%d-%H%M%S')}.json").write_text(
        json.dumps({"meta": meta, "args": {k: v for k, v in vars(args).items() if not k.startswith("_")},
                    "rows": rows}, indent=2))
    print("\n" + md)


if __name__ == "__main__":
    main()

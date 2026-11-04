#!/usr/bin/env python
"""Single-stream benchmark: naive (no cache) vs KV-cache engine.

Reports TTFT, end-to-end latency, decode throughput, inter-token latency
percentiles (p50/p95/p99) and peak GPU memory. Every engine runs the same model
code and the same prompts; each config gets warmup runs, then N timed runs.

Later phases add: batched/concurrent load generation with Poisson arrivals.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

from llmserve.baseline.naive import generate_naive
from llmserve.config import ModelConfig
from llmserve.datatypes import SamplingParams
from llmserve.engine.generate import generate
from llmserve.weights import load_model, random_model

ENGINES = {"naive": generate_naive, "kv_cache": generate}
DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}


def make_prompt(vocab: int, n: int, seed: int = 0) -> list[int]:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(min(100, vocab // 4), vocab, (n,), generator=g).tolist()


def env_info(device: str) -> dict:
    info = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
    }
    if device.startswith("cuda"):
        info["gpu"] = torch.cuda.get_device_name(0)
        info["cuda"] = torch.version.cuda
    return info


def to_markdown(rows: list[dict]) -> str:
    cols = ["engine", "prompt_len", "new_tokens", "ttft_ms", "e2e_s", "tok_per_s",
            "itl_p50_ms", "itl_p95_ms", "itl_p99_ms", "peak_mem_mib", "speedup_vs_naive"]
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        out.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="tiny", help="'tiny' (random weights) or HF repo id / path")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", choices=list(DTYPES), default=None)
    ap.add_argument("--prompt-lens", type=int, nargs="+", default=[128, 512])
    ap.add_argument("--new-tokens", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    dtype = DTYPES[args.dtype] if args.dtype else (
        torch.float16 if args.device.startswith("cuda") else torch.float32
    )
    if args.model == "tiny":
        model = random_model(ModelConfig.tiny(), device=args.device, dtype=dtype)
    else:
        model = load_model(args.model, device=args.device, dtype=dtype)
    vocab = model.config.vocab_size
    params = SamplingParams(max_new_tokens=args.new_tokens)  # greedy, no EOS => fixed length

    rows: list[dict] = []
    for plen in args.prompt_lens:
        prompt = make_prompt(vocab, plen)
        baseline_e2e = None
        for name, fn in ENGINES.items():
            for _ in range(args.warmup):
                fn(model, prompt, params, device=args.device)
            runs = [fn(model, prompt, params, device=args.device) for _ in range(args.runs)]

            itl = np.concatenate([r.step_latencies_s for r in runs]) * 1e3
            e2e = float(np.median([r.total_s for r in runs]))
            if name == "naive":
                baseline_e2e = e2e
            rows.append({
                "engine": name,
                "prompt_len": plen,
                "new_tokens": args.new_tokens,
                "ttft_ms": round(float(np.median([r.ttft_s for r in runs])) * 1e3, 1),
                "e2e_s": round(e2e, 3),
                "tok_per_s": round(args.new_tokens / e2e, 1),
                "itl_p50_ms": round(float(np.percentile(itl, 50)), 2),
                "itl_p95_ms": round(float(np.percentile(itl, 95)), 2),
                "itl_p99_ms": round(float(np.percentile(itl, 99)), 2),
                "peak_mem_mib": round(max(r.peak_memory_bytes for r in runs) / 2**20, 0),
                "speedup_vs_naive": round(baseline_e2e / e2e, 2) if baseline_e2e else "",
            })
            print(f"done: {name} prompt_len={plen}")

    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    payload = {
        "model": args.model, "dtype": str(dtype), "device": args.device,
        "env": env_info(args.device), "args": vars(args), "rows": rows,
    }
    (out_dir / f"bench-{stamp}.json").write_text(json.dumps(payload, indent=2))
    md = to_markdown(rows)
    (out_dir / "latest.md").write_text(md + "\n")
    print("\n" + md)


if __name__ == "__main__":
    main()

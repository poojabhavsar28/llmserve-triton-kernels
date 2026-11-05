#!/usr/bin/env python
"""Phase 4 benchmark: fp16 vs INT8 vs INT4 -- weight memory, decode speed, perplexity.

    python benchmarks/run_quant_benchmark.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0
    python benchmarks/run_quant_benchmark.py --model tiny            # offline smoke test

Perplexity uses WikiText-2 (needs `pip install datasets` + internet) or --text-file. With
--model tiny the weights are random, so perplexity is only a code-path check.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch

from llmserve.datatypes import SamplingParams
from llmserve.engine.generate import generate
from llmserve.eval.perplexity import load_wikitext2_tokens, perplexity
from llmserve.quant import model_nbytes, quantize_model
from llmserve.weights import load_model, random_model, resolve_model_path

DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}


def build(args, dtype, scheme):
    model = (random_model(device=args.device, dtype=dtype) if args.model == "tiny"
             else load_model(args.model, device=args.device, dtype=dtype))
    if scheme == "fp16-triton":
        quantize_model(model, "fp16-triton")
    elif scheme == "int8":
        quantize_model(model, "int8")
    elif scheme in ("int4", "int4-nokernel"):
        quantize_model(model, "int4", group_size=args.group_size, use_kernel=(scheme == "int4"))
    return model


def get_eval_tokens(args, vocab):
    if args.no_ppl:
        return None
    if args.model == "tiny":
        g = torch.Generator().manual_seed(0)
        return torch.randint(0, vocab, (1, 2048), generator=g)
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(resolve_model_path(args.model))
    if args.text_file:
        return tok(Path(args.text_file).read_text(), return_tensors="pt").input_ids[:, : args.ppl_tokens]
    try:
        return load_wikitext2_tokens(tok, args.ppl_tokens)
    except Exception as e:  # noqa: BLE001
        print(f"[warn] perplexity skipped ({type(e).__name__}: {e}). Use --text-file or `pip install datasets`.")
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="tiny")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", choices=list(DTYPES), default=None)
    ap.add_argument("--schemes", nargs="+", default=["fp16", "int8", "int4"],
                    choices=["fp16", "fp16-triton", "int8", "int4", "int4-nokernel"])
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--prompt-len", type=int, default=128)
    ap.add_argument("--new-tokens", type=int, default=128)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--ppl-tokens", type=int, default=40000)
    ap.add_argument("--window", type=int, default=512)
    ap.add_argument("--text-file", default=None)
    ap.add_argument("--no-ppl", action="store_true")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    dtype = DTYPES[args.dtype] if args.dtype else (
        torch.float16 if args.device.startswith("cuda") else torch.float32)
    if args.model == "tiny" and args.group_size > 32:
        args.group_size = 32  # tiny model dims are too small for 128

    rows, tokens = [], None
    for scheme in args.schemes:
        model = build(args, dtype, scheme)
        vocab = model.config.vocab_size
        g = torch.Generator().manual_seed(0)
        prompt = torch.randint(min(100, vocab // 4), vocab, (args.prompt_len,), generator=g).tolist()
        params = SamplingParams(max_new_tokens=args.new_tokens)
        generate(model, prompt, params, device=args.device)  # warmup
        runs = [generate(model, prompt, params, device=args.device) for _ in range(args.runs)]
        if tokens is None and not args.no_ppl:
            tokens = get_eval_tokens(args, vocab)
        ppl = perplexity(model, tokens, args.window) if tokens is not None else None
        rows.append({
            "scheme": scheme,
            "weights_mib": round(model_nbytes(model) / 2**20, 1),
            "decode_tok_s": round(float(np.median([r.decode_tokens_per_s for r in runs])), 1),
            "peak_mem_mib": round(max(r.peak_memory_bytes for r in runs) / 2**20, 0),
            "perplexity": round(ppl, 3) if ppl is not None else "n/a",
        })
        print(f"done: {scheme}")
        del model
        gc.collect()
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    base = rows[0]
    for r in rows:
        r["mem_vs_first"] = f"{r['weights_mib'] / base['weights_mib']:.2f}x"
        r["speed_vs_first"] = f"{r['decode_tok_s'] / base['decode_tok_s']:.2f}x" if base["decode_tok_s"] else ""
        if isinstance(r["perplexity"], float) and isinstance(base["perplexity"], float):
            r["ppl_delta"] = f"{r['perplexity'] - base['perplexity']:+.3f}"
        else:
            r["ppl_delta"] = "n/a"
    cols = list(rows[0].keys())
    md = "\n".join(["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
                   + ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows])
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    (out / "quant_latest.md").write_text(md + "\n")
    (out / f"quant-{time.strftime('%Y%m%d-%H%M%S')}.json").write_text(
        json.dumps({"model": args.model, "device": args.device, "dtype": str(dtype),
                    "group_size": args.group_size, "rows": rows}, indent=2))
    print("\n" + md)
    if args.model == "tiny":
        print("\n(random weights: perplexity is only a code-path check)")


if __name__ == "__main__":
    main()

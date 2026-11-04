"""Quick demo:  python -m llmserve.cli --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --prompt "..." """

from __future__ import annotations

import argparse

import torch

from llmserve.baseline.naive import generate_naive
from llmserve.datatypes import SamplingParams
from llmserve.engine.generate import generate
from llmserve.weights import load_model, random_model, resolve_model_path

DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF repo id or local directory")
    ap.add_argument("--prompt", default="Explain KV caching in one paragraph.")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--mode", choices=["naive", "kv_cache"], default="kv_cache")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", choices=list(DTYPES), default=None)
    args = ap.parse_args()

    dtype = DTYPES[args.dtype] if args.dtype else (
        torch.float16 if args.device.startswith("cuda") else torch.float32
    )
    if args.model == "tiny":  # offline smoke test: random weights, token ids in/out
        tok, model = None, random_model(device=args.device, dtype=dtype)
        prompt_ids, eos = list(range(10, 26)), None
    else:
        from transformers import AutoTokenizer

        path = resolve_model_path(args.model)
        tok = AutoTokenizer.from_pretrained(path)
        model = load_model(str(path), device=args.device, dtype=dtype)
        prompt_ids, eos = tok(args.prompt).input_ids, tok.eos_token_id

    params = SamplingParams(
        temperature=args.temperature, max_new_tokens=args.max_new_tokens, eos_token_id=eos
    )
    fn = generate if args.mode == "kv_cache" else generate_naive
    result = fn(model, prompt_ids, params, device=args.device)

    print(tok.decode(result.tokens, skip_special_tokens=True) if tok else result.tokens)
    print(
        f"\n[{args.mode}] ttft={result.ttft_s * 1e3:.1f} ms | "
        f"decode={result.decode_tokens_per_s:.1f} tok/s | "
        f"peak mem={result.peak_memory_bytes / 2**20:.0f} MiB"
    )


if __name__ == "__main__":
    main()

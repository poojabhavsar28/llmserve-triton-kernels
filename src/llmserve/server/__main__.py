"""python -m llmserve.server --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 [--quant int4]"""

from __future__ import annotations

import argparse

import torch
import uvicorn

from llmserve.config import ModelConfig
from llmserve.kvcache.paged import num_blocks_for_budget
from llmserve.quant import quantize_model
from llmserve.server.app import create_app
from llmserve.weights import load_model, random_model, resolve_model_path

DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}


def main() -> None:
    ap = argparse.ArgumentParser(description="llmserve OpenAI-compatible server")
    ap.add_argument("--model", default="tiny", help="'tiny' (random weights), HF repo id, or path")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", choices=list(DTYPES), default=None)
    ap.add_argument("--quant", choices=["fp16", "fp16-triton", "int8", "int4"], default="fp16")
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--num-blocks", type=int, default=512)
    ap.add_argument("--kv-mem-mib", type=int, default=None, help="size the KV pool by memory")
    ap.add_argument("--max-batch-size", type=int, default=8)
    ap.add_argument("--max-pending", type=int, default=64)
    ap.add_argument("--policy", choices=["continuous", "static"], default="continuous")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    dtype = DTYPES[args.dtype] if args.dtype else (
        torch.float16 if args.device.startswith("cuda") else torch.float32
    )
    tokenizer = None
    if args.model == "tiny":
        model = random_model(ModelConfig.tiny(), device=args.device, dtype=dtype)
    else:
        from transformers import AutoTokenizer

        path = resolve_model_path(args.model)
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = load_model(str(path), device=args.device, dtype=dtype)
    if args.quant != "fp16":
        model = quantize_model(model, args.quant, group_size=args.group_size)

    num_blocks = args.num_blocks
    if args.kv_mem_mib:
        num_blocks = num_blocks_for_budget(
            model.config, args.block_size, dtype, args.kv_mem_mib * 2**20
        )
    print(f"KV pool: {num_blocks} blocks x {args.block_size} tokens = {num_blocks * args.block_size} tokens")

    app = create_app(
        model, tokenizer, model_name=args.model, num_blocks=num_blocks,
        block_size=args.block_size, max_batch_size=args.max_batch_size,
        max_pending=args.max_pending, policy=args.policy,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()

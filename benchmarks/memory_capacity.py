#!/usr/bin/env python
"""Phase 2 benchmark: how many concurrent sequences fit in a fixed KV-cache memory budget?

  contiguous: every sequence reserves max_seq_len slots up front (what a pre-allocated cache does)
  paged:      sequences take ceil(len / block_size) blocks, using the real BlockAllocator

Sequence lengths (prompt + output) are drawn from a lognormal around --mean-len, which matches the
"many short, few long" shape of real traffic. This is a capacity *simulation* (pure bookkeeping, no
model run): it measures the memory-management policy, not GPU speed.
"""

from __future__ import annotations

import argparse
import math
import random

from llmserve.kvcache import BlockAllocator, OutOfBlocks

PRESETS = {  # kv-cache geometry of common small models
    "tinyllama-1.1b": dict(layers=22, kv_heads=4, head_dim=64, max_seq=2048),
    "qwen2.5-0.5b": dict(layers=24, kv_heads=2, head_dim=64, max_seq=4096),
    "tiny": dict(layers=2, kv_heads=2, head_dim=16, max_seq=512),
}


def simulate(args, rng: random.Random) -> dict:
    g = PRESETS[args.preset]
    bytes_per_token = 2 * g["layers"] * g["kv_heads"] * g["head_dim"] * args.dtype_bytes
    budget = args.budget_mib * 2**20
    max_seq = args.max_seq or g["max_seq"]

    contiguous = budget // (max_seq * bytes_per_token)

    num_blocks = budget // (args.block_size * bytes_per_token)
    alloc, held, used_tokens = BlockAllocator(num_blocks), 0, 0
    sigma = 0.8
    mu = math.log(args.mean_len) - sigma**2 / 2
    while True:
        n = int(min(max(rng.lognormvariate(mu, sigma), 16), max_seq))
        need = -(-n // args.block_size)
        try:
            for _ in range(need):
                alloc.allocate()
        except OutOfBlocks:
            break
        held += 1
        used_tokens += n
    frag = 1 - used_tokens / max(1, alloc.num_used * args.block_size)
    return {"contiguous": contiguous, "paged": held, "frag": frag, "kib_per_token": bytes_per_token / 1024}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=list(PRESETS), default="tinyllama-1.1b")
    ap.add_argument("--budget-mib", type=int, default=1024, help="memory reserved for the KV-cache")
    ap.add_argument("--max-seq", type=int, default=None, help="per-sequence reservation (contiguous)")
    ap.add_argument("--mean-len", type=int, default=300, help="mean prompt+output length")
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--dtype-bytes", type=int, default=2)
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    runs = [simulate(args, rng) for _ in range(args.trials)]
    c = runs[0]["contiguous"]
    p = sum(r["paged"] for r in runs) / len(runs)
    frag = sum(r["frag"] for r in runs) / len(runs)
    g = PRESETS[args.preset]
    print(f"model geometry: {args.preset} | KV = {runs[0]['kib_per_token']:.1f} KiB/token | "
          f"budget = {args.budget_mib} MiB | reservation = {args.max_seq or g['max_seq']} tokens | "
          f"mean length = {args.mean_len} | block = {args.block_size}")
    print("\n| cache | concurrent sequences | vs contiguous | internal fragmentation |")
    print("|---|---|---|---|")
    print(f"| contiguous | {c} | 1.00x | n/a (reserves max_seq) |")
    print(f"| paged | {p:.1f} | {p / max(c, 1):.2f}x | {frag:.1%} |")


if __name__ == "__main__":
    main()

"""Single-request generation with a KV-cache: one prefill pass, then O(1)-per-token decode."""

from __future__ import annotations

from time import perf_counter

import torch

from llmserve.datatypes import GenerationResult, SamplingParams
from llmserve.kvcache.contiguous import KVCache
from llmserve.sampling import make_generator, sample
from llmserve.utils import check_length, peak_memory_bytes, reset_peak_memory


@torch.inference_mode()
def generate(
    model, prompt_ids: list[int], params: SamplingParams, *, device: torch.device | str
) -> GenerationResult:
    device = torch.device(device)
    cfg = model.config
    total_len = len(prompt_ids) + params.max_new_tokens
    check_length(cfg, total_len)

    dtype = next(model.parameters()).dtype
    cache = KVCache(cfg, batch_size=1, max_seq_len=total_len, device=device, dtype=dtype)
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    gen = make_generator(params, device)

    reset_peak_memory(device)
    t_start = perf_counter()

    # Prefill: process the whole prompt once, populate the cache.
    logits = model(ids, cache=cache, start_pos=0, last_only=True)[:, -1]
    cache.advance(ids.shape[1])
    nxt = sample(logits, params, gen)
    tok = int(nxt.item())  # .item() blocks until the GPU is done, so timings are honest
    ttft = perf_counter() - t_start

    tokens = [tok]
    step_latencies: list[float] = []

    # Decode: feed only the newest token; keys/values for the past come from the cache.
    while len(tokens) < params.max_new_tokens and tok != params.eos_token_id:
        t0 = perf_counter()
        logits = model(nxt[:, None], cache=cache, start_pos=cache.seq_len, last_only=True)[:, -1]
        cache.advance(1)
        nxt = sample(logits, params, gen)
        tok = int(nxt.item())
        step_latencies.append(perf_counter() - t0)
        tokens.append(tok)

    return GenerationResult(
        tokens=tokens,
        ttft_s=ttft,
        step_latencies_s=step_latencies,
        total_s=perf_counter() - t_start,
        peak_memory_bytes=peak_memory_bytes(device),
    )

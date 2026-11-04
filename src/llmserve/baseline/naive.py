"""The baseline every optimisation is measured against.

No KV-cache, batch size 1: each new token re-runs the *entire* sequence through
the model, so total work grows quadratically with generated length. Deliberately
simple, deliberately slow, and shares the exact same model code as the fast path
so any speedup is attributable to the engine, not to a different model.
"""

from __future__ import annotations

from time import perf_counter

import torch

from llmserve.datatypes import GenerationResult, SamplingParams
from llmserve.sampling import make_generator, sample
from llmserve.utils import check_length, peak_memory_bytes, reset_peak_memory


@torch.inference_mode()
def generate_naive(
    model, prompt_ids: list[int], params: SamplingParams, *, device: torch.device | str
) -> GenerationResult:
    device = torch.device(device)
    check_length(model.config, len(prompt_ids) + params.max_new_tokens)

    ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    gen = make_generator(params, device)

    reset_peak_memory(device)
    t_start = perf_counter()

    logits = model(ids, cache=None, start_pos=0, last_only=True)[:, -1]
    nxt = sample(logits, params, gen)
    tok = int(nxt.item())
    ttft = perf_counter() - t_start

    tokens = [tok]
    step_latencies: list[float] = []

    while len(tokens) < params.max_new_tokens and tok != params.eos_token_id:
        t0 = perf_counter()
        ids = torch.cat([ids, nxt[:, None]], dim=1)
        logits = model(ids, cache=None, start_pos=0, last_only=True)[:, -1]  # full recompute
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

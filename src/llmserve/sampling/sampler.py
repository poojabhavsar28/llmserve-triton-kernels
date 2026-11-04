from __future__ import annotations

import torch

from llmserve.datatypes import SamplingParams


def make_generator(params: SamplingParams, device: torch.device) -> torch.Generator | None:
    if params.seed is None:
        return None
    g = torch.Generator(device=device)
    g.manual_seed(params.seed)
    return g


def sample(
    logits: torch.Tensor, params: SamplingParams, generator: torch.Generator | None = None
) -> torch.Tensor:
    """logits [B, V] -> token ids [B]. Greedy when temperature <= 0."""
    if params.temperature <= 0:
        return logits.argmax(dim=-1)

    logits = logits.float() / params.temperature

    if params.top_k > 0:
        k = min(params.top_k, logits.size(-1))
        kth = torch.topk(logits, k, dim=-1).values[..., -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))

    if params.top_p < 1.0:
        sorted_logits, idx = torch.sort(logits, descending=True, dim=-1)
        probs = torch.softmax(sorted_logits, dim=-1)
        # drop a token if the mass *before* it already exceeds top_p (always keeps the top-1)
        remove = (probs.cumsum(dim=-1) - probs) > params.top_p
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(-1, idx, sorted_logits)

    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1, generator=generator).squeeze(-1)

"""Perplexity evaluation (non-overlapping windows)."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


@torch.inference_mode()
def perplexity(model, token_ids: torch.Tensor, window: int = 512, max_windows: int | None = None):
    """token_ids [1, N] (any device). exp(mean NLL) over non-overlapping windows of ``window``.

    Each window is scored independently (no context carried across windows), so the number is
    comparable *between schemes on the same data* but is higher than a stride-overlapped PPL.
    """
    device = next(model.parameters()).device
    n = token_ids.shape[1]
    nll, count, done = 0.0, 0, 0
    for s in range(0, n - 1, window):
        chunk = token_ids[:, s : s + window].to(device)
        if chunk.shape[1] < 2:
            break
        logits = model(chunk)[0, :-1].float()
        nll += F.cross_entropy(logits, chunk[0, 1:], reduction="sum").item()
        count += chunk.shape[1] - 1
        done += 1
        if max_windows and done >= max_windows:
            break
    return math.exp(nll / count)


def load_wikitext2_tokens(tokenizer, max_tokens: int | None = 40_000) -> torch.Tensor:
    """WikiText-2 (raw) test split -> token ids [1, N]. Needs `pip install datasets`."""
    try:
        from datasets import load_dataset
    except ImportError as e:  # pragma: no cover
        raise ImportError("pip install datasets   (or pass --text-file)") from e
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    ids = tokenizer("\n\n".join(ds["text"]), return_tensors="pt").input_ids
    return ids[:, :max_tokens] if max_tokens else ids

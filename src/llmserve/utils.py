from __future__ import annotations

import torch

from llmserve.config import ModelConfig


def check_length(cfg: ModelConfig, total_len: int) -> None:
    if total_len > cfg.max_position_embeddings:
        raise ValueError(
            f"prompt + max_new_tokens = {total_len} exceeds "
            f"max_position_embeddings = {cfg.max_position_embeddings}"
        )


def reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)


def peak_memory_bytes(device: torch.device) -> int:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        return int(torch.cuda.max_memory_allocated(device))
    return 0

"""Phase 1 KV-cache: one pre-allocated contiguous buffer per cache.

Layout: [num_layers, batch, num_kv_heads, max_seq_len, head_dim] for K and V.

Limitation (motivates Phase 2): every sequence reserves max_seq_len up front, so
memory is wasted on unused tail positions and sequences of different lengths
cannot share a batch. The paged cache (kvcache/paged.py) fixes both.
"""

from __future__ import annotations

import torch

from llmserve.config import ModelConfig


class KVCache:
    def __init__(
        self,
        cfg: ModelConfig,
        batch_size: int,
        max_seq_len: int,
        device: torch.device | str,
        dtype: torch.dtype,
    ):
        shape = (cfg.num_layers, batch_size, cfg.num_kv_heads, max_seq_len, cfg.head_dim)
        self.k = torch.zeros(shape, device=device, dtype=dtype)
        self.v = torch.zeros(shape, device=device, dtype=dtype)
        self.max_seq_len = max_seq_len
        self.seq_len = 0  # number of positions already committed

    def update(
        self, layer: int, k: torch.Tensor, v: torch.Tensor, start_pos: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Write k/v ([B, kv_heads, T, D]) at start_pos; return all keys/values so far."""
        T = k.shape[2]
        end = start_pos + T
        if end > self.max_seq_len:
            raise ValueError(
                f"KV-cache overflow: need {end} positions, capacity {self.max_seq_len}"
            )
        self.k[layer, :, :, start_pos:end] = k
        self.v[layer, :, :, start_pos:end] = v
        return self.k[layer, :, :, :end], self.v[layer, :, :, :end]

    def advance(self, n: int) -> None:
        self.seq_len += n

    @property
    def nbytes(self) -> int:
        return self.k.numel() * self.k.element_size() * 2

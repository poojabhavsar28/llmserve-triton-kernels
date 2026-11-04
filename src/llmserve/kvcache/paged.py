"""Phase 2: paged KV-cache (PagedAttention-style memory management).

Instead of reserving ``max_seq_len`` positions per sequence, K/V live in a shared pool of
fixed-size *blocks*. Each sequence owns a *block table* (list of block ids); blocks are
allocated on demand as the sequence grows and returned to the free list when it finishes.

Consequences (measured by ``benchmarks/memory_capacity.py``):
  * internal fragmentation is bounded by one partially-filled block per sequence
  * sequences of different lengths share one batch
  * freed memory is immediately reusable by other requests

Attention over a paged cache is done here by *gathering* each sequence's blocks into a
contiguous tensor and running masked SDPA. That gather copy is the cost a fused paged-attention
kernel removes (see docs/DESIGN.md).

Pool layout: [num_layers, num_blocks, num_kv_heads, block_size, head_dim]
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from llmserve.config import ModelConfig


class OutOfBlocks(RuntimeError):
    """The block pool is exhausted."""


class BlockAllocator:
    """Free-list allocator over ``num_blocks`` integer block ids."""

    def __init__(self, num_blocks: int):
        self.num_blocks = num_blocks
        self._free: deque[int] = deque(range(num_blocks))
        self._used: set[int] = set()

    @property
    def num_free(self) -> int:
        return len(self._free)

    @property
    def num_used(self) -> int:
        return len(self._used)

    def allocate(self) -> int:
        if not self._free:
            raise OutOfBlocks("no free KV-cache blocks")
        b = self._free.popleft()
        self._used.add(b)
        return b

    def free(self, blocks: list[int]) -> None:
        for b in blocks:
            if b not in self._used:
                raise ValueError(f"double free or invalid block id {b}")
            self._used.remove(b)
            self._free.append(b)


@dataclass
class PagedBatch:
    """Everything one forward pass needs to read/write the paged cache.

    B sequences, right-padded to T new tokens each. Built by :meth:`PagedKVCache.build_batch`.
    """

    cache: PagedKVCache
    positions: torch.Tensor  # [B, T] absolute position of each (padded) query token
    valid: torch.Tensor  # [B, T] bool, False for padding
    write_blocks: torch.Tensor  # [N] block id for each valid token (flattened in valid order)
    write_offsets: torch.Tensor  # [N] offset within block
    block_tables: torch.Tensor  # [B, M] padded with 0 (masked out by context_lens)
    context_lens: torch.Tensor  # [B] number of keys visible after this step's write
    last_idx: torch.Tensor  # [B] index of each sequence's last real token within T

    def attend(
        self, layer: int, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        """q [B,H,T,D], k/v [B,KV,T,D] (already RoPE'd) -> attention output [B,H,T,D]."""
        pool_k, pool_v = self.cache.k_pool[layer], self.cache.v_pool[layer]

        # 1) write this step's K/V into their slots (padding tokens are skipped)
        kw = k.transpose(1, 2)[self.valid]  # [N, KV, D]
        vw = v.transpose(1, 2)[self.valid]
        pool_k[self.write_blocks, :, self.write_offsets] = kw
        pool_v[self.write_blocks, :, self.write_offsets] = vw

        # 2) gather every sequence's blocks into contiguous [B, KV, M*block_size, D]
        B, M = self.block_tables.shape
        _, KV, bs, D = pool_k.shape
        kg = pool_k[self.block_tables].permute(0, 2, 1, 3, 4).reshape(B, KV, M * bs, D)
        vg = pool_v[self.block_tables].permute(0, 2, 1, 3, 4).reshape(B, KV, M * bs, D)

        H = q.shape[1]
        if H != KV:
            rep = H // KV
            kg = kg.repeat_interleave(rep, dim=1)
            vg = vg.repeat_interleave(rep, dim=1)

        # 3) causal + length mask: key j is visible to query i iff j <= pos(i) and j < ctx_len
        key_pos = torch.arange(M * bs, device=q.device)
        mask = (key_pos[None, None, :] <= self.positions[:, :, None]) & (
            key_pos[None, None, :] < self.context_lens[:, None, None]
        )
        return F.scaled_dot_product_attention(q, kg, vg, attn_mask=mask[:, None])


class PagedKVCache:
    def __init__(
        self,
        cfg: ModelConfig,
        num_blocks: int,
        block_size: int,
        device: torch.device | str,
        dtype: torch.dtype,
    ):
        self.cfg = cfg
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.device = torch.device(device)
        shape = (cfg.num_layers, num_blocks, cfg.num_kv_heads, block_size, cfg.head_dim)
        self.k_pool = torch.zeros(shape, device=device, dtype=dtype)
        self.v_pool = torch.zeros(shape, device=device, dtype=dtype)
        self.allocator = BlockAllocator(num_blocks)
        self.tables: dict[int, list[int]] = {}
        self.lengths: dict[int, int] = {}

    # ---- sizing helpers -------------------------------------------------
    def blocks_for(self, n_tokens: int) -> int:
        return -(-n_tokens // self.block_size)

    @property
    def num_free_blocks(self) -> int:
        return self.allocator.num_free

    @property
    def nbytes(self) -> int:
        return (self.k_pool.numel() + self.v_pool.numel()) * self.k_pool.element_size()

    def utilization(self) -> float:
        """Fraction of blocks currently allocated."""
        return self.allocator.num_used / self.num_blocks

    def internal_fragmentation(self) -> float:
        """Fraction of allocated slots that hold no token (tail waste of partially-filled blocks)."""
        slots = self.allocator.num_used * self.block_size
        used = sum(self.lengths.values())
        return 0.0 if slots == 0 else 1.0 - used / slots

    # ---- sequence lifecycle --------------------------------------------
    def add_sequence(self, seq_id: int) -> None:
        if seq_id in self.tables:
            raise ValueError(f"sequence {seq_id} already exists")
        self.tables[seq_id] = []
        self.lengths[seq_id] = 0

    def extra_blocks_needed(self, seq_id: int, new_tokens: int) -> int:
        need = self.blocks_for(self.lengths[seq_id] + new_tokens)
        return max(0, need - len(self.tables[seq_id]))

    def free_sequence(self, seq_id: int) -> None:
        self.allocator.free(self.tables.pop(seq_id))
        del self.lengths[seq_id]

    def has_sequence(self, seq_id: int) -> bool:
        return seq_id in self.tables

    # ---- batch construction ---------------------------------------------
    def build_batch(self, seq_ids: list[int], token_counts: list[int]) -> PagedBatch:
        """Allocate blocks and describe a forward pass appending ``token_counts[i]`` tokens
        to sequence ``seq_ids[i]``. Atomic: raises OutOfBlocks before changing any state."""
        total_extra = sum(self.extra_blocks_needed(s, n) for s, n in zip(seq_ids, token_counts))
        if total_extra > self.allocator.num_free:
            raise OutOfBlocks(f"need {total_extra} blocks, {self.allocator.num_free} free")

        B, T = len(seq_ids), max(token_counts)
        positions = torch.zeros(B, T, dtype=torch.long)
        valid = torch.zeros(B, T, dtype=torch.bool)
        w_blocks: list[torch.Tensor] = []
        w_offsets: list[torch.Tensor] = []
        ctx_lens, last_idx = [], []

        for b, (sid, n) in enumerate(zip(seq_ids, token_counts)):
            table = self.tables[sid]
            for _ in range(self.extra_blocks_needed(sid, n)):
                table.append(self.allocator.allocate())
            start = self.lengths[sid]
            pos = torch.arange(start, start + n)
            positions[b, :n] = pos
            valid[b, :n] = True
            w_blocks.append(torch.tensor(table)[pos // self.block_size])
            w_offsets.append(pos % self.block_size)
            ctx_lens.append(start + n)
            last_idx.append(n - 1)
            self.lengths[sid] = start + n

        M = max(len(self.tables[s]) for s in seq_ids)
        tables = torch.zeros(B, M, dtype=torch.long)
        for b, sid in enumerate(seq_ids):
            t = self.tables[sid]
            tables[b, : len(t)] = torch.tensor(t)

        dev = self.device
        return PagedBatch(
            cache=self,
            positions=positions.to(dev),
            valid=valid.to(dev),
            write_blocks=torch.cat(w_blocks).to(dev),
            write_offsets=torch.cat(w_offsets).to(dev),
            block_tables=tables.to(dev),
            context_lens=torch.tensor(ctx_lens, dtype=torch.long, device=dev),
            last_idx=torch.tensor(last_idx, dtype=torch.long, device=dev),
        )


def num_blocks_for_budget(
    cfg: ModelConfig, block_size: int, dtype: torch.dtype, budget_bytes: int
) -> int:
    """How many blocks fit in ``budget_bytes`` (K and V, all layers)."""
    elem = torch.empty(0, dtype=dtype).element_size()
    per_block = 2 * cfg.num_layers * cfg.num_kv_heads * block_size * cfg.head_dim * elem
    return max(1, budget_bytes // per_block)

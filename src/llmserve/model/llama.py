"""Llama-style decoder (RMSNorm, RoPE, GQA, SwiGLU).

Parameter names match Hugging Face checkpoints exactly so weights load without
renaming:  model.embed_tokens / model.layers.N.self_attn.q_proj / ... / lm_head.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from llmserve.config import ModelConfig
from llmserve.kvcache.contiguous import KVCache
from llmserve.kvcache.paged import PagedBatch


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x.to(dtype)


def rope_cos_sin(
    positions: torch.Tensor, head_dim: int, theta: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """cos/sin for absolute positions ([T] or [B, T]), computed in fp32. -> [..., head_dim]."""
    exponent = torch.arange(0, head_dim, 2, device=positions.device, dtype=torch.float32) / head_dim
    inv_freq = 1.0 / (theta**exponent)
    freqs = positions.float()[..., None] * inv_freq
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rope(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    if cos.dim() == 2:  # shared positions [T, D]
        cos, sin = cos[None, None], sin[None, None]
    else:  # per-sequence positions [B, T, D]
        cos, sin = cos[:, None], sin[:, None]
    return q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.num_heads = cfg.num_heads
        self.num_kv_heads = cfg.num_kv_heads
        self.head_dim = cfg.head_dim
        h = cfg.hidden_size
        self.q_proj = nn.Linear(h, self.num_heads * self.head_dim, bias=cfg.qkv_bias)
        self.k_proj = nn.Linear(h, self.num_kv_heads * self.head_dim, bias=cfg.qkv_bias)
        self.v_proj = nn.Linear(h, self.num_kv_heads * self.head_dim, bias=cfg.qkv_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, h, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cache: KVCache | None,
        layer_idx: int,
        start_pos: int,
        paged: PagedBatch | None = None,
    ) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q, k, cos, sin)

        if paged is not None:  # paged KV-cache: write, gather, masked attention (see kvcache/paged.py)
            out = paged.attend(layer_idx, q, k, v)
            out = out.transpose(1, 2).reshape(B, T, self.num_heads * self.head_dim)
            return self.o_proj(out)

        if cache is not None:
            k, v = cache.update(layer_idx, k, v, start_pos)
        S = k.shape[2]  # total key length

        if self.num_kv_heads != self.num_heads:  # GQA: expand KV heads to match query heads
            rep = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)

        if T == 1:  # decode: the single query attends to every cached key
            out = F.scaled_dot_product_attention(q, k, v)
        elif S == T:  # full prefill starting at position 0
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:  # chunked prefill: query i sits at absolute position start_pos + i
            q_pos = start_pos + torch.arange(T, device=x.device)
            mask = torch.arange(S, device=x.device)[None, :] <= q_pos[:, None]
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

        out = out.transpose(1, 2).reshape(B, T, self.num_heads * self.head_dim)
        return self.o_proj(out)


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.self_attn = Attention(cfg)
        self.mlp = MLP(cfg)
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

    def forward(self, x, cos, sin, cache, layer_idx, start_pos, paged=None):
        h = self.input_layernorm(x)
        x = x + self.self_attn(h, cos, sin, cache, layer_idx, start_pos, paged)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class LlamaModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(DecoderLayer(cfg) for _ in range(cfg.num_layers))
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)


class LlamaForCausalLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.config = cfg
        self.model = LlamaModel(cfg)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        cache: KVCache | None = None,
        start_pos: int = 0,
        last_only: bool = False,
        paged: PagedBatch | None = None,
    ) -> torch.Tensor:
        """input_ids [B, T] -> logits [B, T, V] (or [B, 1, V] when last_only).

        With ``paged`` (continuous batching) sequences may have different lengths; positions come
        from the batch, input_ids are right-padded, and logits are returned only for each
        sequence's last real token: [B, 1, V].
        """
        cfg = self.config
        T = input_ids.shape[1]
        x = self.model.embed_tokens(input_ids)
        if paged is not None:
            positions = paged.positions
        else:
            positions = torch.arange(start_pos, start_pos + T, device=input_ids.device)
        cos, sin = rope_cos_sin(positions, cfg.head_dim, cfg.rope_theta)
        cos, sin = cos.to(x.dtype), sin.to(x.dtype)
        for i, layer in enumerate(self.model.layers):
            x = layer(x, cos, sin, cache, i, start_pos, paged)
        if paged is not None:
            x = x[torch.arange(x.shape[0], device=x.device), paged.last_idx][:, None, :]
        x = self.model.norm(x)
        if last_only:
            x = x[:, -1:, :]
        return self.lm_head(x)

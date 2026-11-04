import pytest
import torch

from llmserve.config import ModelConfig
from llmserve.kvcache import KVCache


def test_update_returns_prefix_and_stores_values():
    cfg = ModelConfig.tiny()
    cache = KVCache(cfg, batch_size=1, max_seq_len=8, device="cpu", dtype=torch.float32)
    k = torch.randn(1, cfg.num_kv_heads, 3, cfg.head_dim)
    v = torch.randn_like(k)
    ks, vs = cache.update(0, k, v, start_pos=0)
    assert ks.shape == (1, cfg.num_kv_heads, 3, cfg.head_dim)
    assert torch.equal(ks, k) and torch.equal(vs, v)

    k2 = torch.randn(1, cfg.num_kv_heads, 1, cfg.head_dim)
    ks, _ = cache.update(0, k2, k2, start_pos=3)
    assert ks.shape[2] == 4 and torch.equal(ks[:, :, :3], k)


def test_overflow_raises():
    cfg = ModelConfig.tiny()
    cache = KVCache(cfg, 1, 4, device="cpu", dtype=torch.float32)
    x = torch.zeros(1, cfg.num_kv_heads, 5, cfg.head_dim)
    with pytest.raises(ValueError):
        cache.update(0, x, x, start_pos=0)

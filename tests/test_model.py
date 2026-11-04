import pytest
import torch

from llmserve.baseline.naive import generate_naive
from llmserve.config import ModelConfig
from llmserve.datatypes import SamplingParams
from llmserve.engine.generate import generate
from llmserve.kvcache import KVCache

ATOL = 1e-4


def _ids(cfg, B, T, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, cfg.vocab_size, (B, T), generator=g)


def _cache(model, B, T):
    return KVCache(model.config, B, T, device="cpu", dtype=torch.float32)


@torch.inference_mode()
def test_prefill_then_decode_matches_full_forward(tiny_model):
    ids = _ids(tiny_model.config, 1, 20)
    full = tiny_model(ids)  # [1, 20, V], no cache
    cache = _cache(tiny_model, 1, 20)

    pre = tiny_model(ids[:, :12], cache=cache, start_pos=0)
    cache.advance(12)
    assert torch.allclose(pre, full[:, :12], atol=ATOL)

    for pos in range(12, 20):
        step = tiny_model(ids[:, pos : pos + 1], cache=cache, start_pos=cache.seq_len)
        cache.advance(1)
        assert torch.allclose(step[:, 0], full[:, pos], atol=ATOL), f"mismatch at pos {pos}"


@torch.inference_mode()
def test_chunked_prefill_matches_full_forward(tiny_model):
    """Exercises the explicit-mask path (T > 1 with start_pos > 0)."""
    ids = _ids(tiny_model.config, 1, 25)
    full = tiny_model(ids)
    cache = _cache(tiny_model, 1, 25)

    a = tiny_model(ids[:, :9], cache=cache, start_pos=0)
    cache.advance(9)
    b = tiny_model(ids[:, 9:], cache=cache, start_pos=9)
    assert torch.allclose(torch.cat([a, b], dim=1), full, atol=ATOL)


@torch.inference_mode()
def test_batched_cache_matches_unbatched(tiny_model):
    ids = _ids(tiny_model.config, 2, 15)
    cache = _cache(tiny_model, 2, 15)
    batched = tiny_model(ids, cache=cache, start_pos=0)
    for b in range(2):
        solo = tiny_model(ids[b : b + 1])
        assert torch.allclose(batched[b], solo[0], atol=ATOL)


def test_greedy_kvcache_equals_naive(tiny_model):
    prompt = _ids(tiny_model.config, 1, 10)[0].tolist()
    params = SamplingParams(max_new_tokens=24)
    a = generate_naive(tiny_model, prompt, params, device="cpu")
    b = generate(tiny_model, prompt, params, device="cpu")
    assert a.tokens == b.tokens
    assert len(b.step_latencies_s) == 23


def test_eos_stops_generation(tiny_model):
    prompt = _ids(tiny_model.config, 1, 10)[0].tolist()
    first = generate(tiny_model, prompt, SamplingParams(max_new_tokens=1), device="cpu").tokens[0]
    params = SamplingParams(max_new_tokens=50, eos_token_id=first)
    r = generate(tiny_model, prompt, params, device="cpu")
    assert r.tokens == [first]


def test_length_guard(tiny_model):
    cfg = tiny_model.config
    with pytest.raises(ValueError):
        too_long = [1] * cfg.max_position_embeddings
        generate(tiny_model, too_long, SamplingParams(max_new_tokens=5), device="cpu")


def test_hf_parity():
    """Our forward pass must reproduce Hugging Face's Llama logits for identical weights."""
    transformers = pytest.importorskip("transformers")
    hf_cfg = transformers.LlamaConfig(
        vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512,
        rms_norm_eps=1e-5, rope_theta=10000.0, tie_word_embeddings=False,
    )
    torch.manual_seed(0)
    hf = transformers.LlamaForCausalLM(hf_cfg).float().eval()

    from llmserve.model.llama import LlamaForCausalLM

    ours = LlamaForCausalLM(ModelConfig.from_hf_dict(hf_cfg.to_dict())).float().eval()
    sd = {k: v for k, v in hf.state_dict().items() if "rotary_emb" not in k}
    res = ours.load_state_dict(sd, strict=False)
    assert not res.missing_keys and not res.unexpected_keys

    ids = _ids(ModelConfig.tiny(), 1, 21)
    with torch.inference_mode():
        assert torch.allclose(ours(ids), hf(ids).logits, atol=1e-4)


def test_hf_parity_qwen2_bias_and_tied_embeddings():
    """Covers QKV bias + tied word embeddings (Qwen2.5 style). Biases are randomised
    because HF initialises them to zero, which would make the check vacuous."""
    transformers = pytest.importorskip("transformers")
    hf_cfg = transformers.Qwen2Config(
        vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512,
        tie_word_embeddings=True,
    )
    torch.manual_seed(0)
    hf = transformers.Qwen2ForCausalLM(hf_cfg).float().eval()
    for name, p in hf.named_parameters():
        if name.endswith("bias"):
            p.data.normal_(0, 0.1)

    from llmserve.model.llama import LlamaForCausalLM

    cfg = ModelConfig.from_hf_dict(hf_cfg.to_dict())
    assert cfg.qkv_bias and cfg.tie_word_embeddings
    ours = LlamaForCausalLM(cfg).float().eval()
    sd = {k: v for k, v in hf.state_dict().items() if "rotary_emb" not in k}
    res = ours.load_state_dict(sd, strict=False)
    assert not [k for k in res.missing_keys if k != "lm_head.weight"]
    assert not res.unexpected_keys

    ids = _ids(ModelConfig.tiny(), 1, 19)
    with torch.inference_mode():
        assert torch.allclose(ours(ids), hf(ids).logits, atol=1e-4)

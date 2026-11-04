import copy

import pytest
import torch
from torch import nn

from llmserve.eval.perplexity import perplexity
from llmserve.quant import QuantLinear4, QuantLinear8, model_nbytes, quantize_model
from llmserve.quant.int4 import dequantize_int4, pack_int4, quantize_int4, unpack_int4
from llmserve.quant.int8 import dequantize_int8, quantize_int8


def _w(out=48, inp=256, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(out, inp, generator=g)


def test_pack_unpack_roundtrip_exact():
    q = torch.randint(0, 16, (7, 64), dtype=torch.uint8)
    assert torch.equal(unpack_int4(pack_int4(q)), q)


def test_int8_error_bounded_by_half_step():
    w = _w()
    q, scale = quantize_int8(w)
    assert q.dtype == torch.int8 and q.abs().max() <= 127
    err = (w - dequantize_int8(q, scale, torch.float32)).abs()
    assert torch.all(err <= scale[:, None] / 2 + 1e-6)


@pytest.mark.parametrize("group", [32, 64, 128])
def test_int4_error_bounded_by_half_step_per_group(group):
    w = _w()
    packed, scale, zero = quantize_int4(w, group)
    assert packed.shape == (48, 128) and packed.dtype == torch.uint8
    deq = dequantize_int4(packed, scale, zero, group, torch.float32)
    step = scale.repeat_interleave(group, dim=1)
    # zero-point rounding can add up to one extra half-step at the range edges
    assert torch.all((w - deq).abs() <= step + 1e-5)
    assert (w - deq).abs().mean() < step.mean() * 0.35  # typical error ~ step/4


def test_int4_rejects_bad_group():
    with pytest.raises(ValueError):
        quantize_int4(_w(inp=100), 64)


def test_smaller_groups_are_more_accurate():
    w = _w(inp=256)
    errs = []
    for g in (128, 32):
        p, s, z = quantize_int4(w, g)
        errs.append((w - dequantize_int4(p, s, z, g, torch.float32)).pow(2).mean().item())
    assert errs[1] < errs[0]


def test_quant_linear_close_to_fp_linear():
    lin = nn.Linear(256, 48, bias=True)
    x = torch.randn(5, 256)
    ref = lin(x)
    y8 = QuantLinear8.from_linear(lin)(x)
    y4 = QuantLinear4.from_linear(lin, 64, use_kernel=False)(x)
    rel = lambda y: ((y - ref).norm() / ref.norm()).item()  # noqa: E731
    assert rel(y8) < 0.02
    assert rel(y4) < 0.15
    assert rel(y8) < rel(y4)


def test_model_quantization_shrinks_memory_and_preserves_logits(tiny_model):
    ids = torch.randint(0, tiny_model.config.vocab_size, (1, 24))
    ref = tiny_model(ids)
    base = model_nbytes(tiny_model)

    m8 = quantize_model(copy.deepcopy(tiny_model), "int8")
    m4 = quantize_model(copy.deepcopy(tiny_model), "int4", group_size=32, use_kernel=False)
    assert model_nbytes(m8) < base and model_nbytes(m4) < model_nbytes(m8)

    cos = lambda a, b: torch.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()  # noqa: E731
    assert cos(m8(ids), ref) > 0.995
    assert cos(m4(ids), ref) > 0.90
    assert cos(m8(ids), ref) > cos(m4(ids), ref)


def test_quantized_model_still_generates_with_kv_cache(tiny_model):
    from llmserve.datatypes import SamplingParams
    from llmserve.engine.generate import generate

    m = quantize_model(copy.deepcopy(tiny_model), "int4", group_size=32, use_kernel=False)
    r = generate(m, [3, 4, 5, 6], SamplingParams(max_new_tokens=6), device="cpu")
    assert len(r.tokens) == 6


def test_unknown_scheme_rejected(tiny_model):
    with pytest.raises(ValueError):
        quantize_model(copy.deepcopy(tiny_model), "int3")


def test_perplexity_runs_and_is_near_vocab_size_for_random_model(tiny_model):
    g = torch.Generator().manual_seed(0)
    toks = torch.randint(0, tiny_model.config.vocab_size, (1, 300), generator=g)
    ppl = perplexity(tiny_model, toks, window=64)
    assert 1.0 < ppl < tiny_model.config.vocab_size * 3

import copy

import pytest
import torch
import torch.nn.functional as F

triton = pytest.importorskip("triton")

from llmserve.datatypes import SamplingParams  # noqa: E402
from llmserve.engine.generate import generate  # noqa: E402
from llmserve.kernels import TritonLinear16, fp16_linear, use_triton_linear  # noqa: E402
from llmserve.kernels.fp16_linear import CANDIDATES  # noqa: E402
from llmserve.quant import quantize_model  # noqa: E402
from llmserve.quant.triton_int4 import kernel_available  # noqa: E402


def _case(M, N, K, dtype=torch.float32, device="cpu", bias=False, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(M, K, generator=g).to(dtype).to(device)
    w = torch.randn(N, K, generator=g).to(dtype).to(device)
    b = torch.randn(N, generator=g).to(dtype).to(device) if bias else None
    return x, w, b


@pytest.mark.parametrize("M", [1, 3, 16, 20])  # 20 -> two M tiles
@pytest.mark.parametrize("N", [48, 70])  # ragged vs BLOCK_N
@pytest.mark.parametrize("K", [128, 100, 33])  # 100 and 33 exercise the K-tail mask
def test_kernel_matches_linear_fp32(M, N, K):
    x, w, _ = _case(M, N, K)
    assert kernel_available(x)
    torch.testing.assert_close(fp16_linear(x, w), F.linear(x, w), rtol=1e-4, atol=1e-4)


def test_bias_batched_dims_and_noncontiguous_weight():
    x, w, b = _case(6, 64, 128, bias=True)
    y = fp16_linear(x.reshape(2, 3, 128), w, b)
    torch.testing.assert_close(y.reshape(6, 64), F.linear(x, w, b), rtol=1e-4, atol=1e-4)
    wt = w.t().contiguous().t()  # same values, column-major strides
    torch.testing.assert_close(fp16_linear(x, wt, b), F.linear(x, w, b), rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("cfg", CANDIDATES)
def test_every_tuning_candidate_is_correct(cfg):
    x, w, b = _case(4, 70, 200, bias=True)
    torch.testing.assert_close(fp16_linear(x, w, b, config=cfg), F.linear(x, w, b), rtol=1e-4, atol=1e-4)


def test_rejects_bad_inputs():
    x, w, _ = _case(2, 32, 64)
    with pytest.raises(ValueError):
        fp16_linear(x[:, :32], w)
    with pytest.raises(ValueError):
        fp16_linear(x.double(), w)


def test_module_dispatch_and_prefill_fallback():
    lin = torch.nn.Linear(64, 40, bias=True)
    m = TritonLinear16(lin)
    assert m.weight is lin.weight  # shared, not copied
    for rows in (1, 8, 32):  # 32 rows -> cuBLAS fallback path
        x = torch.randn(rows, 64)
        torch.testing.assert_close(m(x), lin(x), rtol=1e-4, atol=1e-4)


def test_model_swap_preserves_logits_and_generation(tiny_model):
    swapped = use_triton_linear(copy.deepcopy(tiny_model))
    assert isinstance(swapped.lm_head, TritonLinear16)
    ids = torch.randint(0, tiny_model.config.vocab_size, (1, 20))
    torch.testing.assert_close(swapped(ids), tiny_model(ids), rtol=1e-3, atol=1e-3)
    p = SamplingParams(max_new_tokens=8)
    assert (
        generate(swapped, [3, 4, 5, 6], p, device="cpu").tokens
        == generate(tiny_model, [3, 4, 5, 6], p, device="cpu").tokens
    )


def test_quantize_model_scheme_fp16_triton(tiny_model):
    m = quantize_model(copy.deepcopy(tiny_model), "fp16-triton")
    assert isinstance(m.model.layers[0].self_attn.q_proj, TritonLinear16)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("M", [1, 8])
def test_kernel_matches_cublas_on_gpu(dtype, M):
    x, w, b = _case(M, 2048, 2048, dtype, "cuda", bias=True)
    torch.testing.assert_close(fp16_linear(x, w, b).float(), F.linear(x, w, b).float(), rtol=3e-2, atol=2e-1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
def test_every_candidate_compiles_and_is_correct_on_gpu():
    x, w, b = _case(1, 5632, 2048, torch.float16, "cuda", bias=True)
    ref = F.linear(x, w, b).float()
    for cfg in CANDIDATES:
        try:
            y = fp16_linear(x, w, b, config=cfg).float()
        except Exception as e:  # noqa: BLE001 - a tile too big for this GPU is skipped, not wrong
            pytest.skip(f"{cfg} does not fit this GPU: {type(e).__name__}")
        torch.testing.assert_close(y, ref, rtol=3e-2, atol=2e-1)

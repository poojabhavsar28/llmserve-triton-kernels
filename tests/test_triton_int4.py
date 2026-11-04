import pytest
import torch

triton = pytest.importorskip("triton")

from llmserve.quant.int4 import QuantLinear4, dequantize_int4, quantize_int4  # noqa: E402
from llmserve.quant.triton_int4 import int4_matmul, kernel_available  # noqa: E402


def _case(M, N, K, group, dtype, device="cpu", seed=0):
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(N, K, generator=g)
    x = torch.randn(M, K, generator=g).to(dtype)
    packed, scale, zero = quantize_int4(w, group)
    packed, zero = packed.to(device), zero.to(device)
    scale = scale.to(dtype).to(device)
    ref = x.to(device) @ dequantize_int4(packed, scale, zero, group, dtype).T
    return x.to(device), packed, scale, zero, ref


@pytest.mark.parametrize("M", [1, 3, 16, 20])  # 20 exercises multiple M tiles + masking
@pytest.mark.parametrize("N", [48, 70])  # not multiples of BLOCK_N -> masking
@pytest.mark.parametrize("group", [32, 64])
def test_kernel_matches_dequant_reference_fp32(M, N, group):
    """Runs in Triton's CPU interpreter when no GPU is present (fp32 -> tight tolerance)."""
    x, packed, scale, zero, ref = _case(M, N, 128, group, torch.float32)
    assert kernel_available(x)
    y = int4_matmul(x, packed, scale, zero, group)
    assert y.shape == ref.shape
    torch.testing.assert_close(y, ref, rtol=1e-4, atol=1e-4)


def test_kernel_handles_batched_leading_dims():
    x, packed, scale, zero, ref = _case(6, 64, 128, 64, torch.float32)
    y = int4_matmul(x.reshape(2, 3, 128), packed, scale, zero, 64)
    torch.testing.assert_close(y.reshape(6, 64), ref, rtol=1e-4, atol=1e-4)


def test_kernel_rejects_bad_shapes():
    x, packed, scale, zero, _ = _case(2, 32, 128, 64, torch.float32)
    with pytest.raises(ValueError):
        int4_matmul(x[:, :64], packed, scale, zero, 64)
    with pytest.raises(ValueError):
        int4_matmul(x, packed, scale, zero, 48)  # not a power of two


def test_quant_linear_dispatches_to_kernel_for_decode_rows():
    lin = torch.nn.Linear(128, 40, bias=True)
    q = QuantLinear4.from_linear(lin, group_size=64, use_kernel=True)
    ref = QuantLinear4.from_linear(lin, group_size=64, use_kernel=False)
    x = torch.randn(4, 128)
    torch.testing.assert_close(q(x), ref(x), rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_kernel_matches_reference_on_gpu(dtype):
    x, packed, scale, zero, ref = _case(8, 2048, 2048, 128, dtype, device="cuda")
    y = int4_matmul(x, packed, scale, zero, 128)
    torch.testing.assert_close(y.float(), ref.float(), rtol=3e-2, atol=3e-1)

"""Phase 6: fused INT4 dequantize + matmul in Triton (decode regime: few activation rows).

Why: decode is memory-bound. A pure-PyTorch INT4 linear first materialises the dequantised
fp16 weight (writing 4x the packed bytes, then reading them again) before the matmul, which
cancels the bandwidth win of 4-bit weights. This kernel reads the packed nibbles, applies
scale/zero-point in registers, and feeds the tensor-core dot directly: weights cross HBM once,
as 4 bits.

    y[M, N] = x[M, K] @ W_hat[N, K]^T,   W_hat = (q - zero[n, g]) * scale[n, g]

Each program computes a BLOCK_M x BLOCK_N output tile and walks K one quantization group at a
time (one scale/zero per column per step). Packed layout matches quant/int4.py: byte j of a
row holds columns 2j (low nibble) and 2j+1 (high nibble).

Needs a CUDA GPU + Triton (Linux, or ``pip install triton-windows`` on Windows). With
TRITON_INTERPRET=1 it also runs on CPU, which is how CI validates the logic. If Triton is
absent, ``kernel_available`` is False and QuantLinear4 silently uses the PyTorch path.
"""

from __future__ import annotations

import os

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except Exception:  # pragma: no cover - triton missing or broken (e.g. plain Windows)
    HAS_TRITON = False


def kernel_available(x: torch.Tensor) -> bool:
    if not HAS_TRITON:
        return False
    return x.is_cuda or os.environ.get("TRITON_INTERPRET") == "1"


if HAS_TRITON:

    @triton.jit
    def _int4_matmul_kernel(
        x_ptr, w_ptr, s_ptr, z_ptr, y_ptr,
        M, N, K,
        stride_xm, stride_xk,
        stride_wn, stride_wk,
        stride_sn, stride_sg,
        stride_zn, stride_zg,
        stride_ym, stride_yn,
        GROUP: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, GROUP)
        m_mask = offs_m < M
        n_mask = offs_n < N

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for g in range(0, K // GROUP):
            k = g * GROUP + offs_k  # [GROUP] absolute column indices

            x = tl.load(
                x_ptr + offs_m[:, None] * stride_xm + k[None, :] * stride_xk,
                mask=m_mask[:, None], other=0.0,
            )  # [BLOCK_M, GROUP]

            packed = tl.load(
                w_ptr + offs_n[None, :] * stride_wn + (k[:, None] // 2) * stride_wk,
                mask=n_mask[None, :], other=0,
            )  # [GROUP, BLOCK_N] bytes (each byte is fetched twice: once per nibble)
            shift = (k[:, None] % 2) * 4
            q = (packed.to(tl.int32) >> shift) & 0xF

            s = tl.load(s_ptr + offs_n * stride_sn + g * stride_sg, mask=n_mask, other=1.0)
            z = tl.load(z_ptr + offs_n * stride_zn + g * stride_zg, mask=n_mask, other=0)
            w = (q.to(tl.float32) - z.to(tl.float32)[None, :]) * s.to(tl.float32)[None, :]

            acc = tl.dot(x, w.to(x.dtype), acc)

        y = acc.to(y_ptr.dtype.element_ty)
        tl.store(
            y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn,
            y, mask=m_mask[:, None] & n_mask[None, :],
        )


def int4_matmul(
    x: torch.Tensor,
    packed: torch.Tensor,
    scale: torch.Tensor,
    zero: torch.Tensor,
    group_size: int,
    block_m: int = 16,
    block_n: int = 64,
) -> torch.Tensor:
    """x [..., K] @ dequant(packed [N, K/2])^T -> [..., N], same dtype as x."""
    if not HAS_TRITON:
        raise RuntimeError("Triton is not available")
    K = x.shape[-1]
    N = packed.shape[0]
    if packed.shape[1] * 2 != K:
        raise ValueError(f"packed weight has K={packed.shape[1] * 2}, activations have K={K}")
    if K % group_size or group_size < 16 or group_size & (group_size - 1):
        raise ValueError("group_size must be a power of two >= 16 that divides K")

    lead = x.shape[:-1]
    x2 = x.reshape(-1, K).contiguous()
    M = x2.shape[0]
    y = torch.empty((M, N), device=x.device, dtype=x.dtype)
    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))
    _int4_matmul_kernel[grid](
        x2, packed, scale, zero, y,
        M, N, K,
        x2.stride(0), x2.stride(1),
        packed.stride(0), packed.stride(1),
        scale.stride(0), scale.stride(1),
        zero.stride(0), zero.stride(1),
        y.stride(0), y.stride(1),
        GROUP=group_size, BLOCK_M=block_m, BLOCK_N=block_n,
    )
    return y.reshape(*lead, N)

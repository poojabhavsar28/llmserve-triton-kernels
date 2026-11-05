"""Fused FP16/BF16 decode linear in Triton:  y[M, N] = x[M, K] @ W[N, K]^T (+ bias).

Why it exists
  * It is the FP16 counterpart of ``quant/triton_int4.py`` with the *same* tiling idea, so the two
    can be compared like for like. cuBLAS FP16 at M=1 is already close to memory bandwidth;
    this kernel is a control experiment that shows how much of the INT4 kernel's gap to FP16 comes
    from Triton/kernel structure and how much from dequantisation work.
  * It fuses the bias add, and it is the template a faster INT4 kernel would follow.

Design (decode regime, M <= 16 rows)
  * One program owns a BLOCK_N slab of output columns and walks K in BLOCK_K chunks, so each weight
    byte crosses HBM exactly once (BLOCK_M = 16 is the tensor-core minimum; extra rows are masked).
  * Weights are row-major [N, K], contiguous along K, so the [BLOCK_K, BLOCK_N] tile is loaded
    K-contiguous (coalesced). ``num_stages`` software-pipelines the loads to keep bytes in flight.
  * fp32 accumulation; output cast back to the activation dtype.

Honest expectation: matching cuBLAS here is a good result; beating it is not the goal. Needs a CUDA
GPU + Triton (Linux, or ``pip install triton-windows``); runs on CPU under TRITON_INTERPRET=1.
Set LLMSERVE_TUNE=1 to time a few tile configs per (N, K) on first use (one-off compile cost).
"""

from __future__ import annotations

import math
import os

import torch
import torch.nn.functional as F
from torch import nn

from llmserve.quant.triton_int4 import HAS_TRITON, kernel_available

if HAS_TRITON:
    import triton
    import triton.language as tl

BLOCK_M = 16
DEFAULT_CONFIG = {"block_n": 32, "block_k": 128, "num_warps": 4, "num_stages": 3}
CANDIDATES = [
    DEFAULT_CONFIG,
    {"block_n": 64, "block_k": 128, "num_warps": 4, "num_stages": 3},
    {"block_n": 32, "block_k": 256, "num_warps": 4, "num_stages": 3},
    {"block_n": 16, "block_k": 256, "num_warps": 2, "num_stages": 4},
    {"block_n": 16, "block_k": 128, "num_warps": 2, "num_stages": 4},
    {"block_n": 64, "block_k": 64, "num_warps": 4, "num_stages": 4},
    {"block_n": 128, "block_k": 64, "num_warps": 8, "num_stages": 3},
    {"block_n": 32, "block_k": 64, "num_warps": 4, "num_stages": 4},
]
_TUNED: dict[tuple, dict] = {}

if HAS_TRITON:

    @triton.jit
    def _fp16_linear_kernel(
        x_ptr, w_ptr, b_ptr, y_ptr,
        M, N, K,
        stride_xm, stride_xk,
        stride_wn, stride_wk,
        stride_ym, stride_yn,
        HAS_BIAS: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        pid_m = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk  # [BM, BK]
        w_ptrs = w_ptr + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk  # [BK, BN]

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BLOCK_K)):
            k_left = K - k * BLOCK_K  # masks the ragged tail when K % BLOCK_K != 0
            x = tl.load(x_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < k_left), other=0.0)
            w = tl.load(w_ptrs, mask=(offs_n[None, :] < N) & (offs_k[:, None] < k_left), other=0.0)
            if x.dtype == tl.bfloat16:
                x = x.to(tl.float16)
            if w.dtype == tl.bfloat16:
                w = w.to(tl.float16)
            acc = tl.dot(x, w, acc)
            x_ptrs += BLOCK_K * stride_xk
            w_ptrs += BLOCK_K * stride_wk

        if HAS_BIAS:
            bias = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0)
            acc += bias.to(tl.float32)[None, :]

        tl.store(
            y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn,
            acc.to(y_ptr.dtype.element_ty),
            mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
        )


def tuning_enabled() -> bool:
    return os.environ.get("LLMSERVE_TUNE") == "1"


def tuned_configs() -> dict[tuple, dict]:
    """(N, K, dtype, device) -> chosen tile config, for reporting what the tuner picked."""
    return dict(_TUNED)


def _launch(x2, weight, bias, y, cfg) -> None:
    M, K = x2.shape
    N = weight.shape[0]
    grid = (triton.cdiv(N, cfg["block_n"]), triton.cdiv(M, BLOCK_M))
    _fp16_linear_kernel[grid](
        x2, weight, bias if bias is not None else x2, y,
        M, N, K,
        x2.stride(0), x2.stride(1),
        weight.stride(0), weight.stride(1),
        y.stride(0), y.stride(1),
        HAS_BIAS=bias is not None,
        BLOCK_M=BLOCK_M, BLOCK_N=cfg["block_n"], BLOCK_K=cfg["block_k"],
        num_warps=cfg["num_warps"], num_stages=cfg["num_stages"],
    )


def _time_cuda(fn, iters: int = 20) -> float:
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def _select_config(x2, weight, bias, y) -> dict:
    if not (tuning_enabled() and x2.is_cuda):
        return DEFAULT_CONFIG
    key = (weight.shape[0], weight.shape[1], x2.dtype, x2.device.index)
    if key not in _TUNED:
        best, best_t = DEFAULT_CONFIG, math.inf
        for cfg in CANDIDATES:
            try:
                _launch(x2, weight, bias, y, cfg)  # compiles; skips configs that don't fit
                torch.cuda.synchronize()
                t = _time_cuda(lambda cfg=cfg: _launch(x2, weight, bias, y, cfg))
            except Exception:  # noqa: BLE001 - e.g. OutOfResources for a big tile
                continue
            if t < best_t:
                best, best_t = cfg, t
        _TUNED[key] = best
    return _TUNED[key]


def fp16_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    config: dict | None = None,
) -> torch.Tensor:
    """x [..., K] @ weight[N, K]^T (+ bias) -> [..., N], same dtype as x. Intended for few rows."""
    if not HAS_TRITON:
        raise RuntimeError("Triton is not available")
    K = x.shape[-1]
    if weight.shape[1] != K:
        raise ValueError(f"weight has K={weight.shape[1]}, activations have K={K}")
    if x.dtype != weight.dtype:
        raise ValueError(f"dtype mismatch: x is {x.dtype}, weight is {weight.dtype}")
    lead = x.shape[:-1]
    x2 = x.reshape(-1, K).contiguous()
    y = torch.empty((x2.shape[0], weight.shape[0]), device=x.device, dtype=x.dtype)
    _launch(x2, weight, bias, y, config or _select_config(x2, weight, bias, y))
    return y.reshape(*lead, weight.shape[0])


class TritonLinear16(nn.Module):
    """Drop-in nn.Linear replacement: Triton kernel for decode-sized inputs, cuBLAS otherwise.
    Shares the original weight tensor (no copy, no extra memory)."""

    ROWS_MAX = 16

    def __init__(self, linear: nn.Linear):
        super().__init__()
        self.weight = linear.weight
        self.bias = linear.bias
        self.out_features, self.in_features = linear.weight.shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rows = x.numel() // x.shape[-1]
        if rows <= self.ROWS_MAX and kernel_available(x):
            return fp16_linear(x, self.weight, self.bias)
        return F.linear(x, self.weight, self.bias)


_LAYER_NAMES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def use_triton_linear(model, include_lm_head: bool = True):
    """Swap the decoder's projections (and optionally lm_head) for TritonLinear16, in place."""
    for layer in model.model.layers:
        for parent in (layer.self_attn, layer.mlp):
            for name in _LAYER_NAMES:
                lin = getattr(parent, name, None)
                if isinstance(lin, nn.Linear):
                    setattr(parent, name, TritonLinear16(lin))
    if include_lm_head and isinstance(model.lm_head, nn.Linear):
        model.lm_head = TritonLinear16(model.lm_head)
    return model

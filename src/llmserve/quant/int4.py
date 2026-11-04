"""Group-wise asymmetric INT4 weight-only quantization (GPTQ/AWQ-style storage, RTN rounding).

For each output row, the input dimension is split into groups of ``group_size`` columns; each
group gets its own scale and zero-point:

    scale = (max - min) / 15        zero = round(-min / scale) in [0, 15]
    q     = clamp(round(W / scale) + zero, 0, 15)          W_hat = (q - zero) * scale

Two 4-bit codes are packed per byte along the input dimension: column 2j -> low nibble,
column 2j+1 -> high nibble. Storage: 0.5 byte/weight + (scale fp16 + zero uint8) per group.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def pack_int4(q: torch.Tensor) -> torch.Tensor:
    """q uint8 [out, in] with values 0..15 -> packed uint8 [out, in // 2]."""
    return (q[:, 0::2] | (q[:, 1::2] << 4)).contiguous()


def unpack_int4(packed: torch.Tensor) -> torch.Tensor:
    lo, hi = packed & 0xF, packed >> 4
    return torch.stack((lo, hi), dim=-1).reshape(packed.shape[0], -1)


def quantize_int4(
    w: torch.Tensor, group_size: int = 128
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """w [out, in] -> (packed uint8 [out, in/2], scale float [out, ng], zero uint8 [out, ng])."""
    out_f, in_f = w.shape
    if in_f % group_size or group_size % 2:
        raise ValueError(f"in_features={in_f} must be divisible by an even group_size={group_size}")
    ng = in_f // group_size
    wg = w.float().reshape(out_f, ng, group_size)
    wmin, wmax = wg.amin(dim=-1, keepdim=True), wg.amax(dim=-1, keepdim=True)
    scale = ((wmax - wmin) / 15.0).clamp(min=1e-8)
    zero = torch.round(-wmin / scale).clamp(0, 15)
    q = torch.round(wg / scale + zero).clamp(0, 15).to(torch.uint8).reshape(out_f, in_f)
    return pack_int4(q), scale.squeeze(-1), zero.squeeze(-1).to(torch.uint8)


def dequantize_int4(
    packed: torch.Tensor,
    scale: torch.Tensor,
    zero: torch.Tensor,
    group_size: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    out_f = packed.shape[0]
    ng = scale.shape[1]
    q = unpack_int4(packed).reshape(out_f, ng, group_size).float()
    w = (q - zero.float()[..., None]) * scale.float()[..., None]
    return w.reshape(out_f, ng * group_size).to(dtype)


class QuantLinear4(nn.Module):
    """INT4 linear. Decode-sized inputs use the fused Triton kernel when available;
    everything else dequantises to the activation dtype and uses a regular matmul."""

    KERNEL_MAX_ROWS = 16  # kernel is for the memory-bound decode regime (few rows)

    def __init__(self, packed, scale, zero, bias, group_size: int, dtype, use_kernel: bool = True):
        super().__init__()
        self.out_features, half = packed.shape
        self.in_features = half * 2
        self.group_size = group_size
        self.use_kernel = use_kernel
        self.register_buffer("qweight", packed)
        self.register_buffer("scale", scale.to(dtype))
        self.register_buffer("zero", zero)
        self.register_buffer("bias", bias)

    @classmethod
    def from_linear(cls, lin: nn.Linear, group_size: int = 128, use_kernel: bool = True):
        packed, scale, zero = quantize_int4(lin.weight.data, group_size)
        bias = lin.bias.data if lin.bias is not None else None
        return cls(packed, scale, zero, bias, group_size, lin.weight.dtype, use_kernel)

    def dequantized_weight(self, dtype: torch.dtype) -> torch.Tensor:
        return dequantize_int4(self.qweight, self.scale, self.zero, self.group_size, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rows = x.numel() // x.shape[-1]
        y = None
        if self.use_kernel and rows <= self.KERNEL_MAX_ROWS:
            from llmserve.quant import triton_int4

            if triton_int4.kernel_available(x):
                y = triton_int4.int4_matmul(x, self.qweight, self.scale, self.zero, self.group_size)
        if y is None:
            y = F.linear(x, self.dequantized_weight(x.dtype))
        return y if self.bias is None else y + self.bias.to(x.dtype)

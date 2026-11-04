"""Per-output-channel symmetric INT8 weight-only quantization.

    scale[o] = max|W[o, :]| / 127        q = round(W / scale) in [-127, 127]
    y = (x @ q^T) * scale                # scale commutes with the matmul (per output channel)

Weights are stored as int8 (2x smaller than fp16); activations stay in fp16/bf16.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def quantize_int8(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """w [out, in] -> (q int8 [out, in], scale float [out])."""
    wf = w.float()
    scale = (wf.abs().amax(dim=1) / 127.0).clamp(min=1e-8)
    q = torch.round(wf / scale[:, None]).clamp(-127, 127).to(torch.int8)
    return q, scale


def dequantize_int8(q: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return (q.float() * scale[:, None]).to(dtype)


class QuantLinear8(nn.Module):
    def __init__(self, q: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor | None, dtype):
        super().__init__()
        self.out_features, self.in_features = q.shape
        self.register_buffer("qweight", q)
        self.register_buffer("scale", scale.to(dtype))
        self.register_buffer("bias", bias)
        self.dtype = dtype

    @classmethod
    def from_linear(cls, lin: nn.Linear) -> QuantLinear8:
        q, scale = quantize_int8(lin.weight.data)
        bias = lin.bias.data if lin.bias is not None else None
        return cls(q, scale, bias, lin.weight.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.qweight.to(x.dtype)) * self.scale.to(x.dtype)
        return y if self.bias is None else y + self.bias.to(x.dtype)

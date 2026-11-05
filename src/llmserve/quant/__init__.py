"""Weight-only quantization (INT8 per-channel, INT4 group-wise) for the decoder's linear layers."""

from __future__ import annotations

from torch import nn

from llmserve.quant.int4 import QuantLinear4, dequantize_int4, quantize_int4
from llmserve.quant.int8 import QuantLinear8, dequantize_int8, quantize_int8

_LAYER_NAMES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")

__all__ = [
    "QuantLinear4",
    "QuantLinear8",
    "dequantize_int4",
    "dequantize_int8",
    "model_nbytes",
    "quantize_int4",
    "quantize_int8",
    "quantize_model",
]


def quantize_model(model, scheme: str, group_size: int = 128, use_kernel: bool = True):
    """Quantize every attention/MLP projection in place. Embeddings, norms and lm_head stay in
    the original dtype (standard practice: they are a small share of bytes and sensitive)."""
    if scheme == "fp16" or scheme == "none":
        return model
    if scheme == "fp16-triton":  # not quantization: same fp16 weights, Triton decode kernel
        from llmserve.kernels import use_triton_linear

        return use_triton_linear(model)
    if scheme not in ("int8", "int4"):
        raise ValueError(f"unknown scheme {scheme!r}; use fp16 | fp16-triton | int8 | int4")
    for layer in model.model.layers:
        for parent in (layer.self_attn, layer.mlp):
            for name in _LAYER_NAMES:
                lin = getattr(parent, name, None)
                if isinstance(lin, nn.Linear):
                    if scheme == "int8":
                        q = QuantLinear8.from_linear(lin)
                    else:
                        q = QuantLinear4.from_linear(lin, group_size, use_kernel)
                    setattr(parent, name, q)
    return model


def model_nbytes(model) -> int:
    """Bytes held by parameters and buffers (shared/tied tensors counted once)."""
    seen: set[int] = set()
    total = 0
    for t in list(model.parameters()) + list(model.buffers()):
        if t.data_ptr() in seen:
            continue
        seen.add(t.data_ptr())
        total += t.numel() * t.element_size()
    return total

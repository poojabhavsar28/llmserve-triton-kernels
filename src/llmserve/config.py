"""Model configuration (Llama / Qwen2 family, HF-compatible field mapping)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 32000
    hidden_size: int = 2048
    intermediate_size: int = 5632
    num_layers: int = 22
    num_heads: int = 32
    num_kv_heads: int = 4
    max_position_embeddings: int = 2048
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-5
    qkv_bias: bool = False  # True for Qwen2-style attention
    tie_word_embeddings: bool = False

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    @classmethod
    def tiny(cls) -> ModelConfig:
        """Small config for tests and smoke benchmarks."""
        return cls(
            vocab_size=256,
            hidden_size=64,
            intermediate_size=128,
            num_layers=2,
            num_heads=4,
            num_kv_heads=2,
            max_position_embeddings=512,
        )

    @classmethod
    def from_hf_dict(cls, d: dict) -> ModelConfig:
        rs = d.get("rope_scaling")
        if rs and rs.get("rope_type", rs.get("type")) not in (None, "default"):
            raise NotImplementedError(
                f"rope_scaling={rs!r} is not supported yet "
                "(works with TinyLlama, Qwen2.5; not Llama-3.x)."
            )
        rope_theta = d.get("rope_theta") or d.get("rope_parameters", {}).get("rope_theta", 10000.0)
        n_heads = d["num_attention_heads"]
        return cls(
            vocab_size=d["vocab_size"],
            hidden_size=d["hidden_size"],
            intermediate_size=d["intermediate_size"],
            num_layers=d["num_hidden_layers"],
            num_heads=n_heads,
            num_kv_heads=d.get("num_key_value_heads", n_heads),
            max_position_embeddings=d.get("max_position_embeddings", 2048),
            rope_theta=float(rope_theta),
            rms_norm_eps=d.get("rms_norm_eps", 1e-5),
            qkv_bias=d.get("model_type") == "qwen2" or bool(d.get("attention_bias", False)),
            tie_word_embeddings=bool(d.get("tie_word_embeddings", False)),
        )

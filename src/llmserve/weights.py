"""Loading weights: Hugging Face safetensors checkpoints, or a random-init model for tests."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from llmserve.config import ModelConfig
from llmserve.model.llama import LlamaForCausalLM


def resolve_model_path(name_or_path: str) -> Path:
    path = Path(name_or_path)
    if path.exists():
        return path
    from huggingface_hub import snapshot_download

    patterns = ["*.json", "*.safetensors", "*.model", "*.txt"]
    return Path(snapshot_download(name_or_path, allow_patterns=patterns))


def load_model(
    name_or_path: str, device: str | torch.device = "cuda", dtype: torch.dtype = torch.float16
) -> LlamaForCausalLM:
    """Load a Llama/Qwen2-architecture checkpoint (HF repo id or local dir)."""
    from safetensors import safe_open

    path = resolve_model_path(name_or_path)
    cfg = ModelConfig.from_hf_dict(json.loads((path / "config.json").read_text()))

    files = sorted(path.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no .safetensors files found in {path}")

    state: dict[str, torch.Tensor] = {}
    for f in files:
        with safe_open(str(f), framework="pt", device="cpu") as sf:
            for key in sf.keys():  # noqa: SIM118 - safetensors handle isn't a dict
                if key.endswith("rotary_emb.inv_freq"):
                    continue  # legacy buffer, we compute RoPE on the fly
                state[key] = sf.get_tensor(key).to(device=device, dtype=dtype)

    with torch.device("meta"):  # allocate no memory; weights are assigned below
        model = LlamaForCausalLM(cfg)
    result = model.load_state_dict(state, strict=False, assign=True)

    if cfg.tie_word_embeddings:
        model.lm_head.weight = model.model.embed_tokens.weight
    tied_ok = {"lm_head.weight"} if cfg.tie_word_embeddings else set()
    missing = [k for k in result.missing_keys if k not in tied_ok]
    if missing or result.unexpected_keys:
        raise RuntimeError(
            f"checkpoint mismatch. missing={missing[:5]} unexpected={result.unexpected_keys[:5]}"
        )
    return model.eval().requires_grad_(False)


def random_model(
    cfg: ModelConfig | None = None,
    seed: int = 0,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
    std: float = 0.05,
) -> LlamaForCausalLM:
    """Deterministic random weights: lets tests and CI run with no downloads."""
    cfg = cfg or ModelConfig.tiny()
    g = torch.Generator().manual_seed(seed)
    model = LlamaForCausalLM(cfg)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.ndim >= 2:
                p.copy_(torch.randn(p.shape, generator=g) * std)
            elif name.endswith("norm.weight"):
                p.fill_(1.0)
            else:
                p.zero_()
    return model.to(device=device, dtype=dtype).eval().requires_grad_(False)

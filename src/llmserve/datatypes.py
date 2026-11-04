from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SamplingParams:
    temperature: float = 0.0  # <= 0 means greedy
    top_k: int = 0  # 0 disables
    top_p: float = 1.0  # 1.0 disables
    max_new_tokens: int = 128
    eos_token_id: int | None = None
    seed: int | None = None


@dataclass
class GenerationResult:
    tokens: list[int]
    ttft_s: float  # time to first token (prefill + first sample)
    step_latencies_s: list[float] = field(default_factory=list)  # per decode step
    total_s: float = 0.0
    peak_memory_bytes: int = 0

    @property
    def decode_tokens_per_s(self) -> float:
        total = sum(self.step_latencies_s)
        return len(self.step_latencies_s) / total if total > 0 else 0.0

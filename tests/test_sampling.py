import torch

from llmserve.datatypes import SamplingParams
from llmserve.sampling import make_generator, sample


def _logits():
    return torch.tensor([[0.1, 3.0, 0.2, 2.9, -1.0]])


def test_greedy_is_argmax():
    assert sample(_logits(), SamplingParams()).item() == 1


def test_top_k_1_is_argmax():
    p = SamplingParams(temperature=1.0, top_k=1)
    assert all(sample(_logits(), p).item() == 1 for _ in range(20))


def test_tiny_top_p_is_argmax():
    p = SamplingParams(temperature=1.0, top_p=1e-6)
    assert all(sample(_logits(), p).item() == 1 for _ in range(20))


def test_top_k_restricts_support():
    p = SamplingParams(temperature=1.0, top_k=2)
    seen = {sample(_logits(), p).item() for _ in range(200)}
    assert seen <= {1, 3}


def test_seed_is_reproducible():
    p = SamplingParams(temperature=1.0, seed=42)
    a = [sample(_logits(), p, make_generator(p, torch.device("cpu"))).item() for _ in range(5)]
    b = [sample(_logits(), p, make_generator(p, torch.device("cpu"))).item() for _ in range(5)]
    assert a == b

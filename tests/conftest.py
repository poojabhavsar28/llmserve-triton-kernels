import os

import pytest
import torch

# Triton's CPU interpreter lets us validate kernel logic without a GPU. Must be set before
# triton is first imported, so do it here, before any llmserve import.
if not torch.cuda.is_available():
    os.environ.setdefault("TRITON_INTERPRET", "1")

from llmserve.weights import random_model  # noqa: E402


@pytest.fixture(scope="session")
def tiny_model():
    return random_model(seed=0)

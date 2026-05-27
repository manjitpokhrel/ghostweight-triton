"""
conftest.py — shared pytest fixtures for ghostweight-triton test suite.

Fixtures
--------
device          : str  — "cuda" (skips whole session if CUDA unavailable)
sample_linear   : nn.Linear on CUDA, float16, shape (768, 3072)
toy_model       : tiny 2-layer MLP on CUDA, float16, used for replace_linears tests
"""

import pytest
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require_cuda():
    """Return 'cuda' or skip the test if no CUDA device is available."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required — no GPU found.")
    return "cuda"


# ---------------------------------------------------------------------------
# Session-scoped: GPU availability guard
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def device() -> str:
    """
    Session-scoped fixture.
    Returns the string 'cuda' or skips the entire session.
    All kernel and module tests depend on this.
    """
    return _require_cuda()


# ---------------------------------------------------------------------------
# Function-scoped: a single nn.Linear on CUDA / float16
# ---------------------------------------------------------------------------

@pytest.fixture()
def sample_linear(device: str) -> nn.Linear:
    """
    A fresh nn.Linear(768, 3072, bias=True) in float16 on CUDA.
    Re-created for every test function to avoid state leakage.
    """
    torch.manual_seed(42)
    linear = nn.Linear(768, 3072, bias=True)
    linear = linear.to(device=device, dtype=torch.float16)
    return linear


# ---------------------------------------------------------------------------
# Function-scoped: a tiny 2-layer MLP (used by replace_linears / report tests)
# ---------------------------------------------------------------------------

class _ToyMLP(nn.Module):
    """
    Minimal MLP:
        Linear(128, 512)  -> ReLU
        Linear(512, 128)  -> ReLU
        Linear(128, 10)   <- deliberately small (< min_features threshold for some tests)

    The third layer has in_features=128 which is right on the border, so tests
    can exercise both the skip and no-skip paths of replace_linears.
    """

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(128, 512, bias=True)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(512, 128, bias=True)
        self.act2 = nn.ReLU()
        self.fc3 = nn.Linear(128, 10, bias=False)  # out_features < min_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc3(self.act2(self.fc2(self.act1(self.fc1(x)))))


@pytest.fixture()
def toy_model(device: str) -> nn.Module:
    """
    A fresh _ToyMLP in float16 on CUDA.
    Re-created for every test function.
    """
    torch.manual_seed(0)
    model = _ToyMLP()
    model = model.to(device=device, dtype=torch.float16)
    return model


# ---------------------------------------------------------------------------
# Parametrize helpers (re-used across multiple test files via indirect use)
# ---------------------------------------------------------------------------

#: dtype pairs tested across kernel tests
DTYPES = [torch.float16, torch.float32]

#: (M, K, N) shape triples exercised in matmul / module tests
SHAPES = [
    (1,   128,  256),   # single-token, small
    (1,   768,  3072),  # GPT-2 MLP single token
    (16,  768,  3072),  # small batch
    (128, 768,  3072),  # typical batch
    (128, 3072, 768),   # GPT-2 MLP down-projection
]

#: sparsity levels used in parametrized tests
SPARSITY_LEVELS = [0.0, 0.3, 0.5, 0.7, 0.95]
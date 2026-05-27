"""
test_gather.py — correctness tests for gather_compact kernel.

Verified properties
-------------------
1.  A_compact shape is [M, K_alive].
2.  W_compact shape is [K_alive, N].
3.  A_compact[m, i] == A[m, alive_indices[i]] for all m, i.
4.  W_compact[i, n] == W[alive_indices[i], n] for all i, n.
5.  K_alive == K (dense, no sparsity) — full pass-through.
6.  K_alive == 1 (extreme sparsity).
7.  Identity permutation → outputs equal inputs.
8.  Reversed index order → correct reordering.
9.  Both float16 and float32 preserved in output.
10. Outputs are on CUDA and are contiguous.
"""

import pytest
import torch

from ghostweight_triton.kernels import gather_compact
from tests.conftest import DTYPES


# ---------------------------------------------------------------------------
# Reference
# ---------------------------------------------------------------------------

def _ref_gather(
    A: torch.Tensor,             # [M, K]
    W: torch.Tensor,             # [K, N]
    alive_indices: torch.Tensor, # [K_alive] int64
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure-PyTorch reference via index-select."""
    A_compact = A[:, alive_indices]   # [M, K_alive]
    W_compact = W[alive_indices, :]   # [K_alive, N]
    return A_compact, W_compact


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_inputs(
    M: int, K: int, N: int,
    dtype: torch.dtype, device: str,
    sparsity: float, seed: int = 7,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    A = torch.randn(M, K, dtype=dtype, device=device)
    W = torch.randn(K, N, dtype=dtype, device=device)
    n_dead = int(K * sparsity)
    perm = torch.randperm(K, device=device)
    dead_set = set(perm[:n_dead].tolist())
    alive_idx = torch.tensor(
        [k for k in range(K) if k not in dead_set],
        dtype=torch.int64, device=device,
    )
    return A, W, alive_idx


# ---------------------------------------------------------------------------
# 1–2 : Output shapes
# ---------------------------------------------------------------------------

class TestGatherShape:

    @pytest.mark.parametrize("M,K,N,sparsity", [
        (1,   128,  256,  0.0),
        (1,   128,  256,  0.5),
        (16,  768,  3072, 0.3),
        (128, 768,  3072, 0.7),
        (128, 3072, 768,  0.5),
    ])
    def test_output_shapes(self, device, M, K, N, sparsity):
        A, W, alive_idx = _make_inputs(M, K, N, torch.float16, device, sparsity)
        K_alive = alive_idx.shape[0]
        A_c, W_c = gather_compact(A, W, alive_idx)
        assert A_c.shape == (M, K_alive), \
            f"A_compact shape mismatch: expected ({M},{K_alive}), got {A_c.shape}"
        assert W_c.shape == (K_alive, N), \
            f"W_compact shape mismatch: expected ({K_alive},{N}), got {W_c.shape}"

    def test_dense_k_alive_equals_k(self, device):
        """No dead neurons → K_alive == K → shapes unchanged."""
        M, K, N = 8, 64, 128
        A = torch.randn(M, K, dtype=torch.float16, device=device)
        W = torch.randn(K, N, dtype=torch.float16, device=device)
        alive_idx = torch.arange(K, dtype=torch.int64, device=device)
        A_c, W_c = gather_compact(A, W, alive_idx)
        assert A_c.shape == (M, K)
        assert W_c.shape == (K, N)

    def test_extreme_k_alive_equals_one(self, device):
        """Only one alive neuron — minimum possible K_alive."""
        M, K, N = 4, 128, 64
        A = torch.randn(M, K, dtype=torch.float16, device=device)
        W = torch.randn(K, N, dtype=torch.float16, device=device)
        alive_idx = torch.tensor([42], dtype=torch.int64, device=device)
        A_c, W_c = gather_compact(A, W, alive_idx)
        assert A_c.shape == (M, 1)
        assert W_c.shape == (1, N)


# ---------------------------------------------------------------------------
# 3–4 : Value correctness
# ---------------------------------------------------------------------------

class TestGatherValues:

    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize("M,K,N,sparsity", [
        (1,  64,   128,  0.0),
        (1,  64,   128,  0.5),
        (8,  256,  512,  0.3),
        (32, 768,  3072, 0.6),
    ])
    def test_a_compact_values(self, device, dtype, M, K, N, sparsity):
        A, W, alive_idx = _make_inputs(M, K, N, dtype, device, sparsity)
        A_c, _ = gather_compact(A, W, alive_idx)
        A_ref, _ = _ref_gather(A, W, alive_idx)
        torch.testing.assert_close(
            A_c, A_ref,
            msg=f"A_compact mismatch M={M},K={K},N={N},sp={sparsity},dtype={dtype}",
        )

    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize("M,K,N,sparsity", [
        (1,  64,   128,  0.0),
        (1,  64,   128,  0.5),
        (8,  256,  512,  0.3),
        (32, 768,  3072, 0.6),
    ])
    def test_w_compact_values(self, device, dtype, M, K, N, sparsity):
        A, W, alive_idx = _make_inputs(M, K, N, dtype, device, sparsity)
        _, W_c = gather_compact(A, W, alive_idx)
        _, W_ref = _ref_gather(A, W, alive_idx)
        torch.testing.assert_close(
            W_c, W_ref,
            msg=f"W_compact mismatch M={M},K={K},N={N},sp={sparsity},dtype={dtype}",
        )

    def test_exact_value_single_alive(self, device):
        """M=1, K_alive=1: exact scalar check."""
        M, K, N = 1, 16, 8
        # A = [[0, 1, 2, ..., 15]]
        A = torch.arange(K, dtype=torch.float32, device=device).unsqueeze(0)
        W = torch.eye(K, N, dtype=torch.float32, device=device)
        alive_idx = torch.tensor([5], dtype=torch.int64, device=device)
        A_c, W_c = gather_compact(A, W, alive_idx)
        assert A_c[0, 0].item() == pytest.approx(5.0)
        torch.testing.assert_close(W_c[0], W[5])

    def test_identity_permutation(self, device):
        """alive_indices = [0, …, K-1] → outputs identical to inputs."""
        M, K, N = 4, 32, 16
        A = torch.randn(M, K, dtype=torch.float32, device=device)
        W = torch.randn(K, N, dtype=torch.float32, device=device)
        alive_idx = torch.arange(K, dtype=torch.int64, device=device)
        A_c, W_c = gather_compact(A, W, alive_idx)
        torch.testing.assert_close(A_c, A)
        torch.testing.assert_close(W_c, W)

    def test_reversed_indices(self, device):
        """Reversed alive_indices → correct mirrored gather."""
        M, K, N = 2, 8, 4
        A = torch.randn(M, K, dtype=torch.float32, device=device)
        W = torch.randn(K, N, dtype=torch.float32, device=device)
        alive_idx = torch.arange(K - 1, -1, -1, dtype=torch.int64, device=device)
        A_c, W_c = gather_compact(A, W, alive_idx)
        A_ref, W_ref = _ref_gather(A, W, alive_idx)
        torch.testing.assert_close(A_c, A_ref)
        torch.testing.assert_close(W_c, W_ref)


# ---------------------------------------------------------------------------
# 9–10 : Output properties
# ---------------------------------------------------------------------------

class TestGatherProperties:

    def test_dtype_preserved_float16(self, device):
        M, K, N = 4, 64, 32
        A = torch.randn(M, K, dtype=torch.float16, device=device)
        W = torch.randn(K, N, dtype=torch.float16, device=device)
        alive_idx = torch.arange(K // 2, dtype=torch.int64, device=device)
        A_c, W_c = gather_compact(A, W, alive_idx)
        assert A_c.dtype == torch.float16
        assert W_c.dtype == torch.float16

    def test_dtype_preserved_float32(self, device):
        M, K, N = 4, 64, 32
        A = torch.randn(M, K, dtype=torch.float32, device=device)
        W = torch.randn(K, N, dtype=torch.float32, device=device)
        alive_idx = torch.arange(K // 2, dtype=torch.int64, device=device)
        A_c, W_c = gather_compact(A, W, alive_idx)
        assert A_c.dtype == torch.float32
        assert W_c.dtype == torch.float32

    def test_outputs_on_cuda(self, device):
        M, K, N = 4, 64, 32
        A = torch.randn(M, K, dtype=torch.float16, device=device)
        W = torch.randn(K, N, dtype=torch.float16, device=device)
        alive_idx = torch.arange(K // 2, dtype=torch.int64, device=device)
        A_c, W_c = gather_compact(A, W, alive_idx)
        assert A_c.is_cuda, "A_compact must be on CUDA"
        assert W_c.is_cuda, "W_compact must be on CUDA"

    def test_outputs_contiguous(self, device):
        M, K, N = 4, 64, 32
        A = torch.randn(M, K, dtype=torch.float16, device=device)
        W = torch.randn(K, N, dtype=torch.float16, device=device)
        alive_idx = torch.arange(K // 2, dtype=torch.int64, device=device)
        A_c, W_c = gather_compact(A, W, alive_idx)
        assert A_c.is_contiguous(), "A_compact must be contiguous"
        assert W_c.is_contiguous(), "W_compact must be contiguous"
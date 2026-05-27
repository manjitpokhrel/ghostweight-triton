"""
test_matmul.py — correctness tests for ghostweight_matmul kernel.

Verified properties
-------------------
1.  Output shape is [M, N].
2.  Output dtype matches input dtype.
3.  Output is on CUDA and contiguous.
4.  Numerical correctness vs reference: C = A[:, alive] @ W[alive, :].
5.  Both float16 (atol=1e-2) and float32 (atol=1e-4) inputs.
6.  K_alive == K (no sparsity) matches plain torch.matmul.
7.  K_alive == 1 (extreme sparsity) exact outer-product check.
8.  Non-contiguous / unsorted alive_indices work correctly.
9.  High sparsity (90%) works correctly.
10. Multiple BLOCK_M / BLOCK_N / BLOCK_K combinations give same result.
11. GPT-2 shapes do not produce NaN or Inf.
"""

import pytest
import torch

from ghostweight_triton.kernels import ghostweight_matmul
from tests.conftest import DTYPES


# ---------------------------------------------------------------------------
# Reference
# ---------------------------------------------------------------------------

def _ref_matmul(
    A: torch.Tensor,             # [M, K]
    W: torch.Tensor,             # [K, N]
    alive_indices: torch.Tensor, # [K_alive] int64
) -> torch.Tensor:
    """
    Reference implementation in float32 accumulation.
    Gathers alive columns/rows then performs a standard matmul.
    """
    A_f = A.float()
    W_f = W.float()
    C = (A_f[:, alive_indices] @ W_f[alive_indices, :]).to(A.dtype)
    return C


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_inputs(
    M: int, K: int, N: int,
    dtype: torch.dtype, device: str,
    sparsity: float, seed: int = 99,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    # Small scale (0.1) keeps fp16 sums in a safe range
    A = torch.randn(M, K, dtype=dtype, device=device) * 0.1
    W = torch.randn(K, N, dtype=dtype, device=device) * 0.1
    n_dead = int(K * sparsity)
    perm = torch.randperm(K, device=device)
    dead_set = set(perm[:n_dead].tolist())
    alive_idx = torch.tensor(
        sorted(k for k in range(K) if k not in dead_set),
        dtype=torch.int64, device=device,
    )
    return A, W, alive_idx


def _tols(dtype: torch.dtype) -> dict:
    """Per-dtype tolerances (fp16 accumulation has larger rounding error)."""
    if dtype == torch.float16:
        return {"atol": 1e-2, "rtol": 1e-2}
    return {"atol": 1e-4, "rtol": 1e-4}


# ---------------------------------------------------------------------------
# 1–3 : Shape, dtype, device
# ---------------------------------------------------------------------------

class TestMatmulBasic:

    @pytest.mark.parametrize("M,K,N", [
        (1,   128,  256),
        (1,   768,  3072),
        (16,  768,  3072),
        (128, 768,  3072),
        (128, 3072, 768),
    ])
    def test_output_shape(self, device, M, K, N):
        A, W, alive_idx = _make_inputs(M, K, N, torch.float16, device, 0.5)
        C = ghostweight_matmul(A, W, alive_idx)
        assert C.shape == (M, N), f"Expected ({M},{N}), got {C.shape}"

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_output_dtype(self, device, dtype):
        A, W, alive_idx = _make_inputs(8, 64, 32, dtype, device, 0.5)
        C = ghostweight_matmul(A, W, alive_idx)
        assert C.dtype == dtype, f"Expected {dtype}, got {C.dtype}"

    def test_output_on_cuda(self, device):
        A, W, alive_idx = _make_inputs(4, 64, 32, torch.float16, device, 0.5)
        C = ghostweight_matmul(A, W, alive_idx)
        assert C.is_cuda, "Output must be on CUDA"

    def test_output_contiguous(self, device):
        A, W, alive_idx = _make_inputs(4, 64, 32, torch.float16, device, 0.5)
        C = ghostweight_matmul(A, W, alive_idx)
        assert C.is_contiguous(), "Output must be contiguous"


# ---------------------------------------------------------------------------
# 4–5 : Numerical correctness
# ---------------------------------------------------------------------------

class TestMatmulCorrectness:

    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize("M,K,N,sparsity", [
        (1,   128,  256,  0.0),
        (1,   128,  256,  0.5),
        (1,   768,  3072, 0.5),
        (16,  768,  3072, 0.3),
        (128, 768,  3072, 0.7),
        (128, 3072, 768,  0.5),
    ])
    def test_matches_reference(self, device, dtype, M, K, N, sparsity):
        A, W, alive_idx = _make_inputs(M, K, N, dtype, device, sparsity)
        C_got = ghostweight_matmul(A, W, alive_idx)
        C_ref = _ref_matmul(A, W, alive_idx)
        torch.testing.assert_close(
            C_got, C_ref, **_tols(dtype),
            msg=(f"M={M},K={K},N={N},sparsity={sparsity},dtype={dtype}"),
        )

    def test_no_sparsity_equals_dense_matmul(self, device):
        """alive_indices = [0…K-1] → result == torch.matmul(A, W)."""
        M, K, N = 32, 128, 64
        torch.manual_seed(1)
        A = torch.randn(M, K, dtype=torch.float32, device=device) * 0.1
        W = torch.randn(K, N, dtype=torch.float32, device=device) * 0.1
        alive_idx = torch.arange(K, dtype=torch.int64, device=device)
        C_ghost = ghostweight_matmul(A, W, alive_idx)
        C_dense = A @ W
        torch.testing.assert_close(C_ghost, C_dense, atol=1e-4, rtol=1e-4)

    def test_k_alive_one_outer_product(self, device):
        """K_alive=1: result must equal outer product of one column × one row."""
        M, K, N = 4, 128, 16
        torch.manual_seed(2)
        A = torch.randn(M, K, dtype=torch.float32, device=device)
        W = torch.randn(K, N, dtype=torch.float32, device=device)
        alive_idx = torch.tensor([7], dtype=torch.int64, device=device)
        C_got = ghostweight_matmul(A, W, alive_idx)
        C_ref = A[:, [7]] @ W[[7], :]
        torch.testing.assert_close(C_got, C_ref, atol=1e-4, rtol=1e-4)

    def test_unsorted_alive_indices(self, device):
        """alive_indices in random order (not sorted) must still be correct."""
        M, K, N = 8, 64, 32
        torch.manual_seed(3)
        A = torch.randn(M, K, dtype=torch.float32, device=device) * 0.1
        W = torch.randn(K, N, dtype=torch.float32, device=device) * 0.1
        alive_idx = torch.randperm(K, device=device)[: K // 2].to(torch.int64)
        C_got = ghostweight_matmul(A, W, alive_idx)
        C_ref = _ref_matmul(A, W, alive_idx)
        torch.testing.assert_close(C_got, C_ref, atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_high_sparsity_90_pct(self, device, dtype):
        """90 % sparsity: only 10 % of neurons alive."""
        A, W, alive_idx = _make_inputs(16, 256, 128, dtype, device, 0.9)
        C_got = ghostweight_matmul(A, W, alive_idx)
        C_ref = _ref_matmul(A, W, alive_idx)
        torch.testing.assert_close(C_got, C_ref, **_tols(dtype))

    def test_block_size_variants_agree(self, device):
        """Different BLOCK_M/N/K values must all produce the same result."""
        M, K, N = 64, 128, 64
        A, W, alive_idx = _make_inputs(M, K, N, torch.float32, "cuda", 0.5)
        C_ref = _ref_matmul(A, W, alive_idx)
        for bm, bn, bk in [(32, 32, 16), (64, 64, 32), (128, 64, 32)]:
            C_got = ghostweight_matmul(
                A, W, alive_idx, BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk
            )
            torch.testing.assert_close(
                C_got, C_ref, atol=1e-4, rtol=1e-4,
                msg=f"Block variant BLOCK_M={bm},BLOCK_N={bn},BLOCK_K={bk}",
            )


# ---------------------------------------------------------------------------
# 11 : GPT-2 shapes stress test
# ---------------------------------------------------------------------------

class TestMatmulStress:

    @pytest.mark.parametrize("M,K,N", [
        (1,   768,  3072),
        (128, 768,  3072),
        (128, 3072, 768),
        (1,   768,  2304),  # attention Wqkv
    ])
    def test_gpt2_shapes_no_nan_inf(self, device, M, K, N):
        A, W, alive_idx = _make_inputs(M, K, N, torch.float16, device, 0.5)
        C = ghostweight_matmul(A, W, alive_idx)
        assert C.shape == (M, N)
        assert not torch.isnan(C).any(), "NaN in output"
        assert not torch.isinf(C).any(), "Inf in output"
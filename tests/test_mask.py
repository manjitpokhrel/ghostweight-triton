"""
test_mask.py — correctness tests for dead_neuron_mask kernel.

Verified properties
-------------------
1.  Output shape is [K].
2.  Output dtype is int32.
3.  Output lives on the same CUDA device as input.
4.  All values are binary (0 or 1).
5.  All-alive case  : activations all above threshold → mask all 1.
6.  All-dead case   : activations all zero            → mask all 0.
7.  Values exactly at threshold are dead (alive requires strict >).
8.  Values just above threshold are alive.
9.  A neuron is alive if ANY row exceeds threshold (multi-row semantics).
10. Mixed sparsity: alive count matches constructed pattern.
11. Custom threshold works correctly.
12. Both float16 and float32 inputs work.
13. Large K (up to 8192) does not crash.
"""

import pytest
import torch

from ghostweight_triton.kernels import dead_neuron_mask
from tests.conftest import DTYPES


# ---------------------------------------------------------------------------
# Reference implementation (pure PyTorch)
# ---------------------------------------------------------------------------

def _ref_mask(activations: torch.Tensor, threshold: float) -> torch.Tensor:
    """
    Neuron k is alive (1) when its absolute maximum across all rows
    strictly exceeds the threshold; dead (0) otherwise.
    """
    max_abs = activations.abs().amax(dim=0)   # [K]
    return (max_abs > threshold).to(torch.int32)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_activations(
    M: int, K: int, dtype: torch.dtype, device: str,
    sparsity: float = 0.0, seed: int = 42,
) -> torch.Tensor:
    """
    Random activation tensor with a controlled fraction of dead columns.
    Dead neurons have every value set to 0 across all rows.
    """
    torch.manual_seed(seed)
    a = torch.randn(M, K, dtype=dtype, device=device)
    if sparsity > 0.0:
        n_dead = int(K * sparsity)
        dead_cols = torch.randperm(K, device=device)[:n_dead]
        a[:, dead_cols] = 0.0
    return a


# ---------------------------------------------------------------------------
# 1–4 : Shape, dtype, device, binary values
# ---------------------------------------------------------------------------

class TestMaskBasic:

    @pytest.mark.parametrize("M,K", [(1, 128), (4, 256), (128, 768), (1, 3072)])
    def test_output_shape(self, device, M, K):
        a = torch.randn(M, K, dtype=torch.float16, device=device)
        mask = dead_neuron_mask(a, threshold=1e-6)
        assert mask.shape == (K,), \
            f"Expected shape ({K},), got {mask.shape}"

    @pytest.mark.parametrize("M,K", [(1, 64), (32, 512)])
    def test_output_dtype_is_int32(self, device, M, K):
        a = torch.randn(M, K, dtype=torch.float16, device=device)
        mask = dead_neuron_mask(a, threshold=1e-6)
        assert mask.dtype == torch.int32, \
            f"Expected int32, got {mask.dtype}"

    def test_output_on_cuda(self, device):
        a = torch.randn(4, 128, dtype=torch.float16, device=device)
        mask = dead_neuron_mask(a, threshold=1e-6)
        assert mask.is_cuda, "Mask must reside on CUDA"

    def test_values_are_binary(self, device):
        a = torch.randn(8, 256, dtype=torch.float16, device=device)
        mask = dead_neuron_mask(a, threshold=1e-6)
        unique = set(mask.unique().tolist())
        assert unique.issubset({0, 1}), \
            f"Non-binary values found: {unique - {0, 1}}"


# ---------------------------------------------------------------------------
# 5–6 : Extreme cases
# ---------------------------------------------------------------------------

class TestMaskExtremes:

    def test_all_alive_ones_input(self, device):
        """Every activation = 1.0, well above 1e-6 → all alive."""
        a = torch.ones(4, 128, dtype=torch.float16, device=device)
        mask = dead_neuron_mask(a, threshold=1e-6)
        assert mask.sum().item() == 128, "All 128 neurons should be alive"

    def test_all_dead_zeros_input(self, device):
        """Every activation = 0 → all dead."""
        a = torch.zeros(4, 128, dtype=torch.float16, device=device)
        mask = dead_neuron_mask(a, threshold=1e-6)
        assert mask.sum().item() == 0, "All neurons should be dead"

    def test_single_row_all_alive(self, device):
        a = torch.ones(1, 64, dtype=torch.float32, device=device)
        mask = dead_neuron_mask(a, threshold=1e-6)
        assert mask.sum().item() == 64

    def test_single_row_all_dead(self, device):
        a = torch.zeros(1, 64, dtype=torch.float32, device=device)
        mask = dead_neuron_mask(a, threshold=1e-6)
        assert mask.sum().item() == 0


# ---------------------------------------------------------------------------
# 7–8 : Threshold boundary
# ---------------------------------------------------------------------------

class TestMaskThreshold:

    def test_value_exactly_at_threshold_is_dead(self, device):
        """
        |value| == threshold → dead.
        Alive requires strict |value| > threshold.
        """
        threshold = 0.01
        K = 64
        a = torch.full((4, K), threshold, dtype=torch.float32, device=device)
        mask = dead_neuron_mask(a, threshold=threshold)
        assert mask.sum().item() == 0, \
            "Values exactly at threshold must be dead"

    def test_value_just_above_threshold_is_alive(self, device):
        threshold = 0.01
        K = 64
        a = torch.full((4, K), threshold + 1e-4, dtype=torch.float32, device=device)
        mask = dead_neuron_mask(a, threshold=threshold)
        assert mask.sum().item() == K, \
            "Values just above threshold must be alive"

    def test_split_threshold(self, device):
        """
        First half of columns: 0.3 < 0.5 threshold → dead.
        Second half:           0.7 > 0.5 threshold → alive.
        """
        K = 32
        threshold = 0.5
        a = torch.zeros(1, K, dtype=torch.float32, device=device)
        a[0, :K // 2] = 0.3   # below threshold
        a[0, K // 2:] = 0.7   # above threshold
        mask = dead_neuron_mask(a, threshold=threshold)
        assert mask[:K // 2].sum().item() == 0,      "First half should be dead"
        assert mask[K // 2:].sum().item() == K // 2, "Second half should be alive"


# ---------------------------------------------------------------------------
# 9 : Multi-row semantics
# ---------------------------------------------------------------------------

class TestMaskMultiRow:

    def test_alive_if_any_row_nonzero(self, device):
        """
        Column k is alive if at least ONE row has |value| > threshold,
        even if all other rows are zero for that column.
        """
        M, K = 8, 64
        a = torch.zeros(M, K, dtype=torch.float32, device=device)
        a[-1, 0] = 1.0   # only last row, only column 0
        mask = dead_neuron_mask(a, threshold=1e-6)
        assert mask[0].item() == 1,              "Column 0 should be alive"
        assert mask[1:].sum().item() == 0,       "All other columns should be dead"

    def test_dead_only_when_all_rows_zero(self, device):
        M, K = 4, 32
        a = torch.zeros(M, K, dtype=torch.float32, device=device)
        a[0, :16] = 1.0   # columns 0–15 alive via row 0
        mask = dead_neuron_mask(a, threshold=1e-6)
        assert mask[:16].sum().item() == 16, "Columns 0-15 should be alive"
        assert mask[16:].sum().item() == 0,  "Columns 16-31 should be dead"


# ---------------------------------------------------------------------------
# 10–12 : Numerical correctness vs reference
# ---------------------------------------------------------------------------

class TestMaskCorrectness:

    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize("M,K,sparsity", [
        (1,   128,  0.0),
        (1,   128,  0.5),
        (4,   256,  0.3),
        (32,  768,  0.7),
        (128, 3072, 0.5),
    ])
    def test_matches_reference(self, device, dtype, M, K, sparsity):
        a = _make_activations(M, K, dtype, device, sparsity=sparsity)
        threshold = 1e-6
        ref = _ref_mask(a, threshold)
        got = dead_neuron_mask(a, threshold=threshold)
        torch.testing.assert_close(
            got.cpu(), ref.cpu(),
            msg=(f"Mask mismatch: M={M}, K={K}, "
                 f"sparsity={sparsity}, dtype={dtype}"),
        )

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_alive_count_matches_pattern(self, device, dtype):
        """Verify #alive neurons matches the exact pattern we constructed."""
        M, K = 8, 200
        sparsity = 0.4   # exactly 80 dead columns constructed
        a = _make_activations(M, K, dtype, device, sparsity=sparsity)
        mask = dead_neuron_mask(a, threshold=1e-6)
        n_alive = mask.sum().item()
        n_expected_alive = K - int(K * sparsity)
        assert n_alive == n_expected_alive, (
            f"Expected {n_expected_alive} alive neurons, got {n_alive}"
        )


# ---------------------------------------------------------------------------
# 13 : Large-K stress
# ---------------------------------------------------------------------------

class TestMaskLargeK:

    @pytest.mark.parametrize("K", [3072, 4096, 8192])
    def test_large_k_no_crash(self, device, K):
        a = torch.randn(64, K, dtype=torch.float16, device=device)
        mask = dead_neuron_mask(a, threshold=1e-6)
        assert mask.shape == (K,)
        assert mask.dtype == torch.int32
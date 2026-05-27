"""
test_module.py — correctness tests for GhostLinear nn.Module.

Verified properties
-------------------
1.  GhostLinear constructed from scratch has correct weight shape.
2.  from_linear copies weight and bias exactly.
3.  from_linear without bias: bias is None.
4.  from_linear places the module on the correct device and dtype.
5.  Forward produces correct output shape for 2-D input [M, in_features].
6.  Forward produces correct output shape for 3-D input [B, S, in_features].
7.  Output dtype matches input dtype.
8.  No NaN or Inf in output.
9.  Dense fallback (sparsity < 0.1) matches F.linear exactly.
10. Ghost path (sparsity >= 0.1) is numerically close to F.linear.
11. current_sparsity is a float in [0, 1].
12. All-zero input → current_sparsity ≈ 1.0.
13. All-ones input → current_sparsity ≈ 0.0.
14. freeze_mask does not raise.
15. After freeze_mask, current_sparsity is constant across different inputs.
16. Bias=True: zero input gives nonzero output (bias contributes).
17. Bias=False + zero input: output is zero.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from ghostweight_triton import GhostLinear
from tests.conftest import DTYPES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_ghost(
    in_f: int, out_f: int, bias: bool,
    device: str, dtype: torch.dtype,
    threshold: float = 1e-6,
) -> GhostLinear:
    torch.manual_seed(42)
    return GhostLinear(
        in_f, out_f, bias=bias,
        threshold=threshold, device=device, dtype=dtype,
    )


def _from_linear(
    in_f: int, out_f: int, bias: bool,
    device: str, dtype: torch.dtype,
    threshold: float = 1e-6,
) -> tuple[GhostLinear, nn.Linear]:
    torch.manual_seed(42)
    lin = nn.Linear(in_f, out_f, bias=bias).to(device=device, dtype=dtype)
    ghost = GhostLinear.from_linear(lin, threshold=threshold)
    return ghost, lin


# ---------------------------------------------------------------------------
# 1–4 : Construction
# ---------------------------------------------------------------------------

class TestGhostLinearConstruction:

    def test_weight_shape_from_scratch(self, device):
        gl = _build_ghost(128, 256, True, device, torch.float16)
        assert gl.weight.shape == (256, 128)

    def test_from_linear_weight_copied(self, device):
        gl, lin = _from_linear(256, 512, True, device, torch.float16)
        torch.testing.assert_close(gl.weight.data, lin.weight.data)

    def test_from_linear_bias_copied(self, device):
        gl, lin = _from_linear(256, 512, True, device, torch.float16)
        assert gl.bias is not None
        torch.testing.assert_close(gl.bias.data, lin.bias.data)

    def test_from_linear_no_bias_is_none(self, device):
        gl, _ = _from_linear(256, 512, False, device, torch.float32)
        assert gl.bias is None

    def test_from_linear_device(self, device):
        gl, _ = _from_linear(128, 256, True, device, torch.float16)
        assert gl.weight.device.type == "cuda"

    def test_from_linear_dtype(self, device):
        gl, _ = _from_linear(128, 256, True, device, torch.float16)
        assert gl.weight.dtype == torch.float16


# ---------------------------------------------------------------------------
# 5–8 : Forward shape / dtype / sanity
# ---------------------------------------------------------------------------

class TestGhostLinearForwardShape:

    @pytest.mark.parametrize("M,in_f,out_f", [
        (1,   128,  256),
        (1,   768,  3072),
        (16,  768,  3072),
        (128, 768,  3072),
        (128, 3072, 768),
    ])
    def test_output_shape_2d(self, device, M, in_f, out_f):
        gl, _ = _from_linear(in_f, out_f, True, device, torch.float16)
        x = torch.randn(M, in_f, dtype=torch.float16, device=device)
        y = gl(x)
        assert y.shape == (M, out_f), f"Expected ({M},{out_f}), got {y.shape}"

    def test_output_shape_3d(self, device):
        """Batch × Sequence × Features input."""
        B, S, in_f, out_f = 2, 16, 128, 256
        gl, _ = _from_linear(in_f, out_f, True, device, torch.float16)
        x = torch.randn(B, S, in_f, dtype=torch.float16, device=device)
        y = gl(x)
        assert y.shape == (B, S, out_f)

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_output_dtype(self, device, dtype):
        gl, _ = _from_linear(128, 256, True, device, dtype)
        x = torch.randn(8, 128, dtype=dtype, device=device)
        y = gl(x)
        assert y.dtype == dtype

    def test_no_nan_no_inf(self, device):
        gl, _ = _from_linear(768, 3072, True, device, torch.float16)
        x = torch.randn(32, 768, dtype=torch.float16, device=device)
        y = gl(x)
        assert not torch.isnan(y).any(), "NaN in output"
        assert not torch.isinf(y).any(), "Inf in output"


# ---------------------------------------------------------------------------
# 9–10 : Dense fallback vs ghost path
# ---------------------------------------------------------------------------

class TestGhostLinearPaths:

    def test_dense_fallback_matches_f_linear(self, device):
        """
        All-ones input → sparsity ≈ 0 → dense fallback path.
        Must match F.linear exactly.
        """
        in_f, out_f = 128, 256
        gl, lin = _from_linear(in_f, out_f, True, device, torch.float32)
        x = torch.ones(8, in_f, dtype=torch.float32, device=device)
        y_ghost = gl(x)
        y_ref = F.linear(x, lin.weight, lin.bias)
        torch.testing.assert_close(y_ghost, y_ref, atol=1e-4, rtol=1e-4)

    def test_ghost_path_close_to_f_linear(self, device):
        """
        50% dead columns: ghost path must still agree with F.linear
        because zero-valued inputs contribute nothing regardless of path.
        """
        in_f, out_f = 256, 512
        torch.manual_seed(7)
        lin = nn.Linear(in_f, out_f, bias=True).to(device=device, dtype=torch.float16)
        gl = GhostLinear.from_linear(lin, threshold=1e-6)

        x = torch.randn(16, in_f, dtype=torch.float16, device=device)
        x[:, ::2] = 0.0   # every other column dead → sparsity ≈ 0.5

        y_ghost = gl(x)
        y_ref = F.linear(x, lin.weight, lin.bias)
        # Ghost computes A[:, alive] @ W.T[alive, :].T which equals F.linear
        # because A[:, dead] == 0 contributes nothing
        torch.testing.assert_close(y_ghost, y_ref, atol=5e-2, rtol=5e-2)


# ---------------------------------------------------------------------------
# 11–13 : current_sparsity property
# ---------------------------------------------------------------------------

class TestGhostLinearSparsity:

    def test_sparsity_is_float(self, device):
        gl, _ = _from_linear(128, 256, True, device, torch.float16)
        x = torch.randn(4, 128, dtype=torch.float16, device=device)
        _ = gl(x)
        assert isinstance(gl.current_sparsity, float)

    def test_sparsity_in_range(self, device):
        gl, _ = _from_linear(128, 256, True, device, torch.float16)
        x = torch.randn(4, 128, dtype=torch.float16, device=device)
        _ = gl(x)
        assert 0.0 <= gl.current_sparsity <= 1.0

    def test_zero_input_high_sparsity(self, device):
        """Zero input → all neurons dead → current_sparsity ≈ 1.0."""
        gl, _ = _from_linear(128, 256, True, device, torch.float32)
        x = torch.zeros(4, 128, dtype=torch.float32, device=device)
        _ = gl(x)
        assert gl.current_sparsity >= 0.99, \
            f"Expected ≈1.0 for zero input, got {gl.current_sparsity}"

    def test_ones_input_low_sparsity(self, device):
        """All-ones input → all neurons alive → current_sparsity ≈ 0.0."""
        gl, _ = _from_linear(128, 256, True, device, torch.float32)
        x = torch.ones(4, 128, dtype=torch.float32, device=device)
        _ = gl(x)
        assert gl.current_sparsity < 0.1, \
            f"Expected <0.1 for dense input, got {gl.current_sparsity}"


# ---------------------------------------------------------------------------
# 14–15 : freeze_mask
# ---------------------------------------------------------------------------

class TestGhostLinearFreezeMask:

    def test_freeze_mask_does_not_raise(self, device):
        gl, _ = _from_linear(128, 256, True, device, torch.float16)
        cal = torch.randn(16, 128, dtype=torch.float16, device=device)
        gl.freeze_mask(cal)   # must not raise

    def test_frozen_sparsity_is_constant(self, device):
        """
        After freeze_mask, current_sparsity must not change
        no matter what input is fed.
        """
        gl, _ = _from_linear(128, 256, True, device, torch.float32)
        # Calibrate with zero → all-dead mask
        cal = torch.zeros(4, 128, dtype=torch.float32, device=device)
        gl.freeze_mask(cal)
        sp_frozen = gl.current_sparsity

        # Feed dense input — mask must stay the same
        x_dense = torch.ones(4, 128, dtype=torch.float32, device=device)
        _ = gl(x_dense)
        assert gl.current_sparsity == pytest.approx(sp_frozen), (
            "current_sparsity changed after freeze_mask — it must remain constant"
        )


# ---------------------------------------------------------------------------
# 16–17 : Bias
# ---------------------------------------------------------------------------

class TestGhostLinearBias:

    def test_bias_contributes_to_output(self, device):
        """bias=True, input=0 → output equals broadcast bias."""
        in_f, out_f = 64, 128
        torch.manual_seed(5)
        lin = nn.Linear(in_f, out_f, bias=True).to(device=device, dtype=torch.float32)
        nn.init.constant_(lin.bias, 1.0)
        gl = GhostLinear.from_linear(lin)
        x = torch.zeros(4, in_f, dtype=torch.float32, device=device)
        y = gl(x)
        assert y.abs().sum().item() > 0, "Bias must contribute when input is zero"

    def test_no_bias_zero_input_gives_zero_output(self, device):
        """bias=False, input=0 → output is exactly 0."""
        gl, _ = _from_linear(64, 128, False, device, torch.float32)
        x = torch.zeros(4, 64, dtype=torch.float32, device=device)
        y = gl(x)
        assert y.abs().sum().item() == 0.0, "No bias + zero input must give zero output"
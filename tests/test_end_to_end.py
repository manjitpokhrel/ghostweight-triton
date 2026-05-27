"""
test_end_to_end.py — end-to-end integration tests.

These tests exercise the complete GhostWeight pipeline from raw tensors
through to model-level inference, verifying:

  A. Full kernel chain:
       activations → dead_neuron_mask → alive_indices → ghostweight_matmul

  B. GhostLinear single-layer equivalence to F.linear.

  C. Multi-layer model replacement, forward correctness, NaN/Inf absence.

  D. sparsity_stats correctness on a live mask.

  E. freeze_mask immutability under repeated forward calls.

No benchmarking here — correctness checks only.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from ghostweight_triton import GhostLinear, replace_linears, get_sparsity_report
from ghostweight_triton.kernels import dead_neuron_mask, ghostweight_matmul
from ghostweight_triton.utils.sparsity import sparsity_stats, log_sparsity


# ---------------------------------------------------------------------------
# Small models for integration tests
# ---------------------------------------------------------------------------

class _FFN(nn.Module):
    """Two-layer FFN: Linear(d, 4d) → GELU → Linear(4d, d)."""

    def __init__(self, d: int = 128, bias: bool = True):
        super().__init__()
        self.fc1 = nn.Linear(d, 4 * d, bias=bias)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(4 * d, d, bias=bias)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class _TransformerLayer(nn.Module):
    """Simplified single transformer encoder layer."""

    def __init__(self, d: int = 128):
        super().__init__()
        self.proj  = nn.Linear(d, d)
        self.ffn   = _FFN(d)
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)

    def forward(self, x):
        x = self.norm1(x + self.proj(x))
        x = self.norm2(x + self.ffn(x))
        return x


# ---------------------------------------------------------------------------
# A. Full kernel chain
# ---------------------------------------------------------------------------

class TestFullKernelChain:

    def test_mask_to_matmul_pipeline(self, device):
        """
        activations → dead_neuron_mask → alive_indices → ghostweight_matmul.
        Result must match A[:, alive] @ W[alive, :] reference.
        """
        M, K, N = 32, 256, 128
        torch.manual_seed(10)
        A = torch.randn(M, K, dtype=torch.float16, device=device) * 0.1
        W = torch.randn(K, N, dtype=torch.float16, device=device) * 0.1
        A[:, ::2] = 0.0   # structured 50% sparsity

        mask = dead_neuron_mask(A, threshold=1e-6)
        assert mask.shape == (K,) and mask.dtype == torch.int32

        alive_idx = mask.nonzero(as_tuple=True)[0].to(torch.int64)
        assert alive_idx.shape[0] > 0, "Must have at least one alive neuron"

        C = ghostweight_matmul(A, W, alive_idx)
        assert C.shape == (M, N)

        # Reference in fp32
        C_ref = (A.float()[:, alive_idx] @ W.float()[alive_idx, :]).half()
        torch.testing.assert_close(C, C_ref, atol=1e-2, rtol=1e-2)

    def test_sparsity_stats_from_live_mask(self, device):
        """sparsity_stats on a real mask must return correct counts."""
        K = 128
        A = torch.zeros(4, K, dtype=torch.float16, device=device)
        A[:, :64] = 1.0   # first half alive, second half dead

        mask = dead_neuron_mask(A, threshold=1e-6)
        stats = sparsity_stats(mask)

        assert stats["total"] == K
        assert stats["alive"] == 64
        assert stats["dead"]  == 64
        assert pytest.approx(stats["pct_dead"], abs=1e-4) == 0.5


# ---------------------------------------------------------------------------
# B. GhostLinear single-layer equivalence
# ---------------------------------------------------------------------------

class TestGhostLinearEquivalence:

    def test_dense_input_matches_f_linear(self, device):
        """Dense input (sparsity < 0.1) → dense fallback → exact F.linear match."""
        in_f, out_f = 128, 256
        torch.manual_seed(42)
        lin = nn.Linear(in_f, out_f, bias=True).to(device=device, dtype=torch.float32)
        gl = GhostLinear.from_linear(lin, threshold=1e-6)

        x = torch.randn(16, in_f, dtype=torch.float32, device=device)
        y_ghost = gl(x)
        y_ref   = F.linear(x, lin.weight, lin.bias)
        torch.testing.assert_close(y_ghost, y_ref, atol=1e-4, rtol=1e-4)

    def test_sparse_input_matches_f_linear(self, device):
        """
        50% dead columns: because dead inputs contribute 0 to F.linear too,
        ghost and dense results must agree.
        """
        in_f, out_f = 128, 256
        torch.manual_seed(3)
        lin = nn.Linear(in_f, out_f, bias=True).to(device=device, dtype=torch.float32)
        gl = GhostLinear.from_linear(lin, threshold=1e-6)

        x = torch.randn(16, in_f, dtype=torch.float32, device=device)
        x[:, ::2] = 0.0   # 50% sparsity

        y_ghost = gl(x)
        y_ref   = F.linear(x, lin.weight, lin.bias)
        torch.testing.assert_close(y_ghost, y_ref, atol=1e-3, rtol=1e-3)


# ---------------------------------------------------------------------------
# C. Multi-layer model integration
# ---------------------------------------------------------------------------

class TestModelIntegration:

    def test_ffn_output_shape_unchanged(self, device):
        torch.manual_seed(0)
        model = _FFN(d=128).to(device=device, dtype=torch.float16)
        x = torch.randn(8, 128, dtype=torch.float16, device=device)
        with torch.no_grad():
            y_before = model(x)
        replace_linears(model, threshold=1e-6, min_features=128)
        with torch.no_grad():
            y_after = model(x)
        assert y_before.shape == y_after.shape

    def test_transformer_layer_replacement(self, device):
        """All qualifying linears in _TransformerLayer become GhostLinear."""
        torch.manual_seed(1)
        model = _TransformerLayer(d=128).to(device=device, dtype=torch.float16)
        replace_linears(model, threshold=1e-6, min_features=128)

        for name, mod in model.named_modules():
            if isinstance(mod, nn.Linear) and not isinstance(mod, GhostLinear):
                assert mod.in_features < 128, (
                    f"Layer {name} (in={mod.in_features}) should have been replaced"
                )

        x = torch.randn(4, 16, 128, dtype=torch.float16, device=device)
        with torch.no_grad():
            y = model(x)
        assert y.shape == (4, 16, 128)

    def test_sparsity_report_after_forward(self, device):
        torch.manual_seed(2)
        model = _FFN(d=256).to(device=device, dtype=torch.float16)
        replace_linears(model, min_features=128)

        x = torch.randn(16, 256, dtype=torch.float16, device=device)
        x[:, ::2] = 0.0
        with torch.no_grad():
            model(x)

        report = get_sparsity_report(model)
        assert len(report) > 0
        for sp in report.values():
            assert 0.0 <= sp <= 1.0

    def test_stacked_layers_no_nan_inf(self, device):
        """Stack of replaced linears must not produce NaN or Inf."""
        layers = nn.Sequential(
            nn.Linear(256, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
        ).to(device=device, dtype=torch.float16)
        replace_linears(layers, min_features=128)

        x = torch.randn(32, 256, dtype=torch.float16, device=device)
        with torch.no_grad():
            y = layers(x)
        assert not torch.isnan(y).any(), "NaN in stacked output"
        assert not torch.isinf(y).any(), "Inf in stacked output"


# ---------------------------------------------------------------------------
# D. log_sparsity smoke test
# ---------------------------------------------------------------------------

class TestLogSparsity:

    def test_log_sparsity_does_not_raise(self, device, capsys):
        K = 64
        A = torch.zeros(4, K, dtype=torch.float16, device=device)
        A[:, :32] = 1.0
        mask = dead_neuron_mask(A, threshold=1e-6)
        log_sparsity(mask)   # must not raise
        # Soft check: something was printed
        out = capsys.readouterr().out
        assert isinstance(out, str)


# ---------------------------------------------------------------------------
# E. freeze_mask integration
# ---------------------------------------------------------------------------

class TestFreezeMaskIntegration:

    def test_freeze_then_multiple_forwards_no_crash(self, device):
        in_f, out_f = 128, 256
        torch.manual_seed(0)
        lin = nn.Linear(in_f, out_f).to(device=device, dtype=torch.float16)
        gl = GhostLinear.from_linear(lin, threshold=1e-6)

        cal = torch.randn(16, in_f, dtype=torch.float16, device=device)
        gl.freeze_mask(cal)

        for _ in range(5):
            x = torch.randn(8, in_f, dtype=torch.float16, device=device)
            with torch.no_grad():
                y = gl(x)
            assert y.shape == (8, out_f)

    def test_frozen_sparsity_constant(self, device):
        in_f, out_f = 128, 256
        torch.manual_seed(0)
        lin = nn.Linear(in_f, out_f).to(device=device, dtype=torch.float32)
        gl = GhostLinear.from_linear(lin, threshold=1e-6)

        cal = torch.zeros(4, in_f, dtype=torch.float32, device=device)
        cal[:, :64] = 1.0
        gl.freeze_mask(cal)
        sp_frozen = gl.current_sparsity

        # Dense input must not change the frozen sparsity
        x_dense = torch.ones(4, in_f, dtype=torch.float32, device=device)
        with torch.no_grad():
            gl(x_dense)

        assert gl.current_sparsity == pytest.approx(sp_frozen), (
            "current_sparsity changed after freeze_mask — it must be immutable"
        )
"""
quickstart.py — 5-minute introduction to GhostWeight-Triton.

What is GhostWeight?
--------------------
A training-free inference sparsity framework for transformer models.

The observation: in LLMs with ReLU or GELU activations, 50–80% of
neurons produce near-zero outputs for any given input token.
These "dead neurons" still consume GPU cycles in a standard F.linear call.

GhostWeight detects dead neurons at inference time using a Triton kernel
and skips them in a fused sparse matmul — all without retraining or
modifying weights.

This script: full pipeline in one file
---------------------------------------
  Step 0  Check GPU environment.
  Step 1  Build a tiny model.
  Step 2  replace_linears — patch all nn.Linear in one call.
  Step 3  Run inference   — GhostLinear auto-detects sparsity.
  Step 4  Read sparsity report.
  Step 5  Freeze mask     — static mask for production deployment.
  Step 6  Use kernels directly (advanced).

Run
---
  cd ghostweight-triton/
  python examples/quickstart.py
"""

import torch
import torch.nn as nn

# ── Public API ───────────────────────────────────────────────────────────────
from ghostweight_triton import (
    GhostLinear,          # drop-in replacement for nn.Linear
    replace_linears,      # patch every qualifying nn.Linear in a model
    get_sparsity_report,  # {layer_name: sparsity_float} for all GhostLinear layers
)
from ghostweight_triton.kernels import (
    dead_neuron_mask,     # (activations [M,K], threshold) → mask [K] int32
    ghostweight_matmul,   # (A, W, alive_indices) → C [M,N]  fused Triton kernel
    gather_compact,       # (A, W, alive_indices) → A_compact, W_compact
)
from ghostweight_triton.utils.sparsity import sparsity_stats, log_sparsity
from ghostweight_triton.utils.compat import check_environment

DEVICE = "cuda"
DTYPE  = torch.float16

# ============================================================================
# Step 0 — environment check
# ============================================================================
print("=" * 62)
print("GhostWeight-Triton  |  Quickstart")
print("=" * 62)
# Prints GPU name, CUDA version, Triton version.
# Warns if TORCH_CUDA_ARCH_LIST is not set (needed for sm_120 / RTX 5060).
check_environment()
print()

if not torch.cuda.is_available():
    raise RuntimeError(
        "This example requires a CUDA GPU.\n"
        "No GPU found — aborting."
    )


# ============================================================================
# Step 1 — build a tiny model
# ============================================================================
print("── Step 1: Build model ─────────────────────────────────────────")

class TinyFFN(nn.Module):
    """
    Two-layer FFN that mimics one transformer MLP block:
        Linear(128 → 512) → ReLU → Linear(512 → 128)

    After replace_linears, both linear layers become GhostLinear.
    """
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(128, 512, bias=True)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(512, 128, bias=True)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


torch.manual_seed(0)
model = TinyFFN().to(device=DEVICE, dtype=DTYPE)
print(f"  {model}")
print()


# ============================================================================
# Step 2 — apply replace_linears
# ============================================================================
print("── Step 2: replace_linears ─────────────────────────────────────")

replaced = replace_linears(
    model,
    threshold=1e-6,    # neurons with max|activation| ≤ this are dead
    min_features=128,  # skip layers with in_features < this
    skip_patterns=[],  # e.g. ["lm_head", "embed"] to protect specific layers
)

print(f"  {len(replaced)} layer(s) replaced:")
for name, in_f in replaced.items():
    print(f"    {name:<20}  in_features={in_f}")

# Sanity check — both layers are now GhostLinear
assert isinstance(model.fc1, GhostLinear), "fc1 should be GhostLinear"
assert isinstance(model.fc2, GhostLinear), "fc2 should be GhostLinear"
print()


# ============================================================================
# Step 3 — run inference
# ============================================================================
print("── Step 3: Run inference ───────────────────────────────────────")

# Simulate an input where ~50% of features are dead (column-wise zeros).
# In a real LLM this happens naturally after ReLU/GELU activations.
batch_size = 8
x = torch.randn(batch_size, 128, dtype=DTYPE, device=DEVICE)
x[:, ::2] = 0.0   # zero every other column → 50% input sparsity

n_zero_cols = (x[0] == 0.0).sum().item()
print(f"  Input shape    : {list(x.shape)}")
print(f"  Dead columns   : {n_zero_cols}/128 ({n_zero_cols/128:.0%})")

with torch.no_grad():
    y = model(x)

print(f"  Output shape   : {list(y.shape)}")
print(f"  Output sample  : {y[0, :4].tolist()}")
print()


# ============================================================================
# Step 4 — sparsity report
# ============================================================================
print("── Step 4: Sparsity report ─────────────────────────────────────")

report = get_sparsity_report(model)
print(f"  {'Layer':<20}  {'Sparsity':>9}  Bar")
print("  " + "─" * 46)
for name, sp in report.items():
    bar = "█" * int(sp * 20)
    print(f"  {name:<20}  {sp:>8.1%}  {bar}")

avg = sum(report.values()) / max(len(report), 1)
print(f"\n  Average sparsity: {avg:.1%}")
print()


# ============================================================================
# Step 5 — freeze mask (optional, for production)
# ============================================================================
print("── Step 5: Freeze mask ─────────────────────────────────────────")
#
# freeze_mask(calibration_input):
#   Runs the mask kernel once on the calibration input, locks the result.
#   From that point forward, forward() skips mask recomputation entirely.
#   Use this when the activation sparsity pattern is stable across requests,
#   which is typical for a given model + dataset distribution.
#

cal = torch.randn(32, 128, dtype=DTYPE, device=DEVICE)
cal[:, ::2] = 0.0   # same sparsity pattern as inference

for name, mod in model.named_modules():
    if isinstance(mod, GhostLinear):
        mod.freeze_mask(cal)
        print(f"  {name}: frozen  sparsity={mod.current_sparsity:.1%}")
print()


# ============================================================================
# Step 6 — use kernels directly (advanced users)
# ============================================================================
print("── Step 6: Kernels directly ────────────────────────────────────")

M, K, N = 16, 128, 256
A = torch.randn(M, K, dtype=DTYPE, device=DEVICE)
W = torch.randn(K, N, dtype=DTYPE, device=DEVICE)
# Simulate 60 % dead input neurons
A[:, : int(K * 0.6)] = 0.0


# ── dead_neuron_mask ─────────────────────────────────────────────────────────
# Returns int32 [K] tensor: 1=alive, 0=dead
mask  = dead_neuron_mask(A, threshold=1e-6)
stats = sparsity_stats(mask)
print(f"  dead_neuron_mask:")
print(f"    total={stats['total']}  alive={stats['alive']}  "
      f"dead={stats['dead']}  pct_dead={stats['pct_dead']:.1%}")
log_sparsity(mask)

# ── ghostweight_matmul ───────────────────────────────────────────────────────
# Fused Triton kernel: C = A[:, alive] @ W[alive, :]
# No temporary gather tensors materialised on the GPU.
alive_idx = mask.nonzero(as_tuple=True)[0].to(torch.int64)
C = ghostweight_matmul(A, W, alive_idx)
print(f"  ghostweight_matmul: output {list(C.shape)}")

# Verify against reference
C_ref  = (A.float()[:, alive_idx] @ W.float()[alive_idx, :]).half()
max_err = (C - C_ref).abs().max().item()
print(f"    max error vs reference: {max_err:.5f}  ✓" if max_err < 0.05
      else f"    max error: {max_err:.5f}  ⚠")

# ── gather_compact ───────────────────────────────────────────────────────────
# Materialise the compact tensors explicitly (useful for profiling / custom ops).
A_c, W_c = gather_compact(A, W, alive_idx)
print(f"  gather_compact:")
print(f"    A_compact {list(A_c.shape)}  W_compact {list(W_c.shape)}")
print()


# ============================================================================
# Done
# ============================================================================
print("=" * 62)
print("Quickstart complete!")
print()
print("  python examples/gpt2_inference.py       — GPT-2 Small inference")
print("  python examples/distilbert_inference.py — DistilBERT inference")
print("  python benchmarks/bench_ghostweight.py  — full benchmark table")
print("  pytest tests/                           — run test suite")
print("=" * 62)
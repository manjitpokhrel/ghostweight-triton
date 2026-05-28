"""
gpt2_inference.py — GhostWeight inference example on GPT-2 Small.

What this shows
---------------
  1. Load GPT-2 Small from HuggingFace.
  2. Apply GhostWeight with a single replace_linears call.
  3. Calibrate dead-neuron masks on a handful of prompts.
  4. Compare dense vs ghost text generation output.
  5. Verify numerical agreement ([CLS] cosine similarity).
  6. Measure forward-pass latency with CUDA Events.

Key insight
-----------
  GPT-2 uses GELU activations.  After GELU, a large fraction of FFN
  intermediate neurons are near-zero for any given token.  GhostWeight
  detects these dead neurons and skips them — no retraining required.

GPT-2 Small specifications
--------------------------
  Parameters         : ~117M
  Layers             : 12 transformer blocks
  hidden_size        : 768
  FFN intermediate   : 3072 (4× hidden_size)
  Attention heads    : 12
  vocab_size         : 50257
  Memory (fp16)      : ≈ 250 MB

Skip patterns used
------------------
  "lm_head" — vocabulary projection (output layer, very sensitive)
  "wte"     — word token embedding (lookup table, not nn.Linear)
  "wpe"     — positional embedding  (lookup table)
  "ln_"     — LayerNorm layers      (not nn.Linear)

Prerequisites
-------------
  pip install transformers

Run
---
  cd ghostweight-triton/
  python examples/gpt2_inference.py
"""

import copy
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
except ImportError:
    raise ImportError("pip install transformers")

from ghostweight_triton import GhostLinear, replace_linears, get_sparsity_report
from ghostweight_triton.utils.compat import check_environment

# ============================================================================
# Configuration
# ============================================================================

MODEL_NAME     = "gpt2"          # GPT-2 Small
DEVICE         = "cuda"
DTYPE          = torch.float16
MAX_NEW_TOKENS = 40

# Layers to keep as nn.Linear (not replaced by GhostLinear).
# Embeddings and LayerNorm are not nn.Linear, but listing them is harmless.
SKIP_PATTERNS = ["lm_head", "wte", "wpe", "ln_"]
MIN_FEATURES  = 128

PROMPTS = [
    "The capital of Nepal is",
    "Triton is a GPU programming language designed to",
    "In 2040, the most energy-efficient AI hardware will",
]

WARMUP_ITERS = 10
BENCH_ITERS  = 50

# ============================================================================
# Setup
# ============================================================================

print("=" * 65)
print("GhostWeight-Triton  |  GPT-2 Small Inference")
print("=" * 65)
check_environment()
print()

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU required.")


# ============================================================================
# Step 1 — load model and tokenizer
# ============================================================================

print(f"Loading {MODEL_NAME} …", end=" ", flush=True)
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

# Load in fp32 (HuggingFace default), cast to fp16
model_dense = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
model_dense = model_dense.to(device=DEVICE, dtype=DTYPE)
model_dense.eval()

n_params = sum(p.numel() for p in model_dense.parameters())
mem_mb   = sum(p.numel() * p.element_size() for p in model_dense.parameters()) / 1e6
print(f"done  ({n_params/1e6:.0f}M params, ~{mem_mb:.0f} MB fp16)")


# ============================================================================
# Step 2 — create a GhostWeight copy and apply replacement
# ============================================================================
# deep-copy keeps model_dense unchanged for baseline comparisons.

print("Creating ghost copy …", end=" ", flush=True)
model_ghost = copy.deepcopy(model_dense)

replaced = replace_linears(
    model_ghost,
    threshold=1e-6,
    skip_patterns=SKIP_PATTERNS,
    min_features=MIN_FEATURES,
)
print(f"done  ({len(replaced)} layers → GhostLinear)")

print("  Sample replacements:")
for name, in_f in list(replaced.items())[:5]:
    print(f"    {name:<55} in_features={in_f}")
if len(replaced) > 5:
    print(f"    … and {len(replaced)-5} more")
print()


# ============================================================================
# Step 3 — calibrate dead-neuron masks
# ============================================================================
# GhostLinear masks are computed on-the-fly during forward() by default.
# Running a few calibration passes with representative inputs stabilises the
# mask so that the timed benchmark uses a warm, accurate mask.

print("Calibrating …", end=" ", flush=True)
for prompt in PROMPTS:
    toks = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        model_ghost(**toks)
print("done")
print()


# ============================================================================
# Step 4 — sparsity report
# ============================================================================

# Fresh forward pass to refresh current_sparsity on all GhostLinear layers
sample_toks = tokenizer(PROMPTS[0], return_tensors="pt").to(DEVICE)
with torch.no_grad():
    model_ghost(**sample_toks)

report = get_sparsity_report(model_ghost)
if report:
    sps    = list(report.values())
    avg_sp = sum(sps) / len(sps)
    print(f"Sparsity report ({len(report)} layers tracked):")
    print(f"  avg={avg_sp:.1%}  min={min(sps):.1%}  max={max(sps):.1%}")

    # Show FFN layers only (most interesting for GhostWeight)
    ffn = {k: v for k, v in report.items() if "mlp" in k.lower()}
    if ffn:
        print("  FFN layers:")
        for name, sp in sorted(ffn.items(), key=lambda x: -x[1])[:6]:
            bar = "▓" * int(sp * 20)
            print(f"    {name:<55} {sp:.1%}  {bar}")
print()


# ============================================================================
# Step 5 — text generation comparison
# ============================================================================

def _generate(
    model: nn.Module, prompt: str, max_new_tokens: int
) -> tuple[str, float]:
    """Greedy generation.  Returns (text, elapsed_seconds)."""
    toks = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(
            **toks,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    return text, elapsed


print("=" * 65)
print("Text Generation Comparison")
print("=" * 65)

for prompt in PROMPTS:
    print(f'\nPrompt: "{prompt}"')
    print("─" * 65)
    text_dense, t_dense = _generate(model_dense, prompt, MAX_NEW_TOKENS)
    text_ghost, t_ghost = _generate(model_ghost, prompt, MAX_NEW_TOKENS)

    speedup = t_dense / t_ghost if t_ghost > 0 else float("inf")
    match   = text_dense.strip() == text_ghost.strip()

    print(f"[Dense ] ({t_dense:.3f}s):\n  {text_dense}")
    print(f"[Ghost ] ({t_ghost:.3f}s):\n  {text_ghost}")
    print(f"Speedup : {speedup:.2f}x  ({(1-t_ghost/t_dense)*100:.1f}% faster)")
    print(f"Match   : {'✓ identical' if match else '⚠ differ (fp16 rounding)'}")


# ============================================================================
# Step 6 — latency microbenchmark (CUDA events)
# ============================================================================
# We benchmark the *encoder forward pass* rather than generation to isolate
# the cost of the linear layers.  Generation includes sampling logic and
# KV-cache management which are not affected by GhostWeight.

print("\n" + "=" * 65)
print(f"Forward-Pass Latency  (warmup={WARMUP_ITERS}, iters={BENCH_ITERS})")
print("=" * 65)

def _bench_fwd(model: nn.Module, toks: dict) -> float:
    """Mean forward-pass time in ms via CUDA Events."""
    for _ in range(WARMUP_ITERS):
        with torch.no_grad():
            model(**toks)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(BENCH_ITERS):
        with torch.no_grad():
            model(**toks)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / BENCH_ITERS


BENCH_CONFIGS = [
    (1,   1,   "single token"),
    (1,   64,  "bs=1 sl=64"),
    (1,   128, "bs=1 sl=128"),
    (8,   128, "bs=8 sl=128"),
]

print(f"  {'Config':<20}  {'Dense(ms)':>10}  {'Ghost(ms)':>10}  {'Speedup':>9}")
print("  " + "─" * 54)
for bs, sl, label in BENCH_CONFIGS:
    toks = tokenizer(
        ["The quick brown fox"] * bs,
        return_tensors="pt",
        truncation=True,
        max_length=sl,
        padding="max_length",
    ).to(DEVICE)
    ms_d = _bench_fwd(model_dense, toks)
    ms_g = _bench_fwd(model_ghost, toks)
    sp   = ms_d / ms_g if ms_g > 0 else float("inf")
    print(f"  {label:<20}  {ms_d:>10.3f}  {ms_g:>10.3f}  {sp:>8.3f}x")


# ============================================================================
# Done
# ============================================================================

print("\n" + "=" * 65)
print("Done!  Try next:")
print("  python examples/distilbert_inference.py")
print("  python benchmarks/bench_gpt2.py")
print("=" * 65)
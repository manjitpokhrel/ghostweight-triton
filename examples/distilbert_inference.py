"""
distilbert_inference.py — GhostWeight inference example on DistilBERT.

What this shows
---------------
  1. Load DistilBERT-base-uncased from HuggingFace.
  2. Apply replace_linears (skip embeddings and LayerNorm).
  3. Calibrate masks on a handful of real sentences.
  4. Compare dense vs ghost [CLS] embeddings (cosine similarity).
  5. Display the sparsity report.
  6. Latency microbenchmark with CUDA Events across batch/seq configs.
  7. Live activation demo: hook into one FFN layer and run dead_neuron_mask.

Why DistilBERT?
---------------
  DistilBERT is 40% smaller than BERT-base while retaining 97% of NLU
  performance.  Its 6-layer architecture with hidden_size=768 and FFN
  intermediate=3072 is identical in shape to one half of BERT-base,
  making it a common production encoder for classification and retrieval.

  After GELU activations in the FFN, many neurons are effectively zero
  for a given input — GhostWeight exploits this automatically.

DistilBERT specifications
--------------------------
  Parameters       : ~66M
  Layers           : 6 transformer blocks
  hidden_size      : 768
  FFN intermediate : 3072 (4× hidden_size)
  Attention heads  : 12
  vocab_size       : 30522
  Memory (fp16)    : ≈ 130 MB encoder, ≈ 260 MB with embeddings

Skip patterns used
------------------
  "embedding" — word + position embeddings (not nn.Linear)
  "LayerNorm" — normalisation layers       (not nn.Linear)

Prerequisites
-------------
  pip install transformers

Run
---
  cd ghostweight-triton/
  python examples/distilbert_inference.py
"""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from transformers import DistilBertModel, DistilBertTokenizer
except ImportError:
    raise ImportError("pip install transformers")

from ghostweight_triton import GhostLinear, replace_linears, get_sparsity_report
from ghostweight_triton.kernels import dead_neuron_mask
from ghostweight_triton.utils.sparsity import sparsity_stats, log_sparsity
from ghostweight_triton.utils.compat import check_environment

# ============================================================================
# Configuration
# ============================================================================

MODEL_NAME    = "distilbert-base-uncased"
DEVICE        = "cuda"
DTYPE         = torch.float16
SKIP_PATTERNS = ["embedding", "LayerNorm"]
MIN_FEATURES  = 128
WARMUP        = 10
BENCH_ITERS   = 100

SENTENCES = [
    "Kathmandu is the capital city of Nepal.",
    "GhostWeight is a training-free inference sparsity framework.",
    "Triton enables writing high-performance GPU kernels in Python.",
    "The RTX 5060 has sm_120 compute capability and 8 GB of VRAM.",
    "DistilBERT retains 97% of BERT accuracy with half the layers.",
]

# ============================================================================
# Setup
# ============================================================================

print("=" * 65)
print("GhostWeight-Triton  |  DistilBERT Inference")
print("=" * 65)
check_environment()
print()

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU required.")


# ============================================================================
# Step 1 — load model and tokenizer
# ============================================================================

print(f"Loading {MODEL_NAME} …", end=" ", flush=True)
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)

model_dense = DistilBertModel.from_pretrained(MODEL_NAME)
model_dense = model_dense.to(device=DEVICE, dtype=DTYPE)
model_dense.eval()

n_params = sum(p.numel() for p in model_dense.parameters())
mem_mb   = sum(p.numel() * p.element_size() for p in model_dense.parameters()) / 1e6
print(f"done  ({n_params/1e6:.0f}M params, ~{mem_mb:.0f} MB fp16)")


# ============================================================================
# Step 2 — create ghost copy and apply replacement
# ============================================================================

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
# Step 3 — calibrate masks
# ============================================================================
# Run forward passes on the calibration sentences so each GhostLinear
# builds an accurate dead-neuron mask.  In production, use a representative
# sample from your target data distribution (~100–1000 examples is sufficient).

print("Calibrating masks …", end=" ", flush=True)
for sentence in SENTENCES:
    toks = tokenizer(
        sentence, return_tensors="pt",
        truncation=True, max_length=128,
    ).to(DEVICE)
    with torch.no_grad():
        model_ghost(**toks)
print("done")
print()


# ============================================================================
# Step 4 — sparsity report
# ============================================================================

# One extra forward to refresh current_sparsity
sample_toks = tokenizer(SENTENCES[0], return_tensors="pt").to(DEVICE)
with torch.no_grad():
    model_ghost(**sample_toks)

report = get_sparsity_report(model_ghost)
print("Sparsity report:")
if report:
    sps    = list(report.values())
    avg_sp = sum(sps) / len(sps)
    print(f"  Layers tracked : {len(report)}")
    print(f"  avg={avg_sp:.1%}  min={min(sps):.1%}  max={max(sps):.1%}")
    print()
    print(f"  {'Layer':<55}  {'Sparsity':>9}  Bar")
    print("  " + "─" * 72)
    for name, sp in sorted(report.items()):
        bar = "▓" * int(sp * 20)
        print(f"  {name:<55}  {sp:>8.1%}  {bar}")
print()


# ============================================================================
# Step 5 — numerical comparison: [CLS] embedding dense vs ghost
# ============================================================================
# Because dead neurons contribute exactly 0 to the linear output regardless
# of which path (dense or ghost) is used, the outputs should be very close.
# We report cosine similarity and max absolute difference.

print("=" * 65)
print("Numerical Comparison: Dense vs Ghost [CLS] Embeddings")
print("=" * 65)

for sentence in SENTENCES[:3]:
    toks = tokenizer(
        sentence, return_tensors="pt",
        truncation=True, max_length=64,
    ).to(DEVICE)

    with torch.no_grad():
        cls_dense = model_dense(**toks).last_hidden_state[:, 0, :]  # [1, 768]
        cls_ghost = model_ghost(**toks).last_hidden_state[:, 0, :]

    cos   = F.cosine_similarity(cls_dense.float(), cls_ghost.float()).item()
    mdiff = (cls_dense - cls_ghost).abs().max().item()

    short = sentence[:50] + ("…" if len(sentence) > 50 else "")
    print(f'  "{short}"')
    print(f"    cosine similarity : {cos:.6f}")
    print(f"    max |diff|        : {mdiff:.6f}")
    print()


# ============================================================================
# Step 6 — latency microbenchmark
# ============================================================================

def _bench(model: nn.Module, toks: dict) -> float:
    """Mean forward-pass latency in ms via CUDA Events."""
    for _ in range(WARMUP):
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


print("=" * 65)
print(f"Latency Benchmark  (warmup={WARMUP}, iters={BENCH_ITERS})")
print("=" * 65)

BENCH_CONFIGS = [
    (1,  32,  "bs=1 sl=32"),
    (1,  128, "bs=1 sl=128"),
    (8,  128, "bs=8 sl=128"),
    (16, 128, "bs=16 sl=128"),
    (1,  512, "bs=1 sl=512"),
]

print(f"  {'Config':<22}  {'Dense(ms)':>10}  {'Ghost(ms)':>10}  {'Speedup':>9}")
print("  " + "─" * 58)

for bs, sl, label in BENCH_CONFIGS:
    toks = tokenizer(
        ["The quick brown fox"] * bs,
        return_tensors="pt",
        truncation=True, max_length=sl, padding="max_length",
    ).to(DEVICE)

    ms_d = _bench(model_dense, toks)
    ms_g = _bench(model_ghost, toks)
    sp   = ms_d / ms_g if ms_g > 0 else float("inf")
    print(f"  {label:<22}  {ms_d:>10.3f}  {ms_g:>10.3f}  {sp:>8.3f}x")


# ============================================================================
# Step 7 — live activation sparsity demo (forward hook)
# ============================================================================
# We attach a hook to the first FFN lin1 layer to capture its output
# activations, then run dead_neuron_mask on them to show real sparsity numbers.

print("\n" + "=" * 65)
print("Live Activation Sparsity Demo")
print("  (hook on transformer.layer.0.ffn.lin1 output)")
print("=" * 65)

_captured: dict = {}

def _hook(name: str):
    def _fn(module, inp, out):
        if name not in _captured:
            _captured[name] = out.detach().float()
    return _fn

hook_handle = None
try:
    target = model_ghost.transformer.layer[0].ffn.lin1
    hook_handle = target.register_forward_hook(_hook("ffn0_lin1"))
except AttributeError:
    pass   # layer structure may differ across transformers versions

if hook_handle is not None:
    toks = tokenizer(SENTENCES[0], return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        model_ghost(**toks)
    hook_handle.remove()

    if "ffn0_lin1" in _captured:
        act = _captured["ffn0_lin1"][0]        # [seq_len, 3072]
        act_fp16 = act.half().to(DEVICE)
        mask  = dead_neuron_mask(act_fp16, threshold=1e-6)
        stats = sparsity_stats(mask)

        print(f"  Layer: transformer.layer[0].ffn.lin1")
        print(f"  Activation shape : {list(act.shape)}")
        print(f"  Dead neurons     : {stats['dead']}/{stats['total']} "
              f"({stats['pct_dead']:.1%})")
        log_sparsity(mask)
    else:
        print("  (hook fired but output not captured)")
else:
    print("  (could not attach hook — transformers version may differ)")


# ============================================================================
# Done
# ============================================================================

print("\n" + "=" * 65)
print("Done!  Try next:")
print("  python examples/gpt2_inference.py")
print("  python benchmarks/bench_distilbert.py")
print("  python benchmarks/bench_kernels.py")
print("=" * 65)
"""
bench_gpt2.py — GhostWeight inference benchmark on GPT-2 Small (real model).

Pipeline
--------
  1. Load GPT-2 Small (GPT2Model — encoder body, no LM head) in fp16.
  2. Measure baseline forward-pass latency (all nn.Linear).
  3. Apply replace_linears, skip embeddings and LayerNorm.
  4. Calibrate GhostLinear masks (5 random forward passes).
  5. Measure ghost forward-pass latency.
  6. Print per-layer sparsity report (top-10 sparsest layers).
  7. Save full results to benchmarks/results/bench_gpt2.csv.

GPT-2 Small reference numbers
------------------------------
  hidden_size       = 768
  intermediate_size = 3072
  num_layers        = 12
  num_heads         = 12
  vocab_size        = 50257
  model_size_fp16  ≈ 250 MB

Methodology
-----------
  - torch.cuda.Event timing (GPU hardware timestamps)
  - WARMUP=10  ITERS=100
  - torch.no_grad() throughout — inference only

Usage
-----
  cd ghostweight-triton/
  python benchmarks/bench_gpt2.py

System target
-------------
  AMD Ryzen 7 7700, RTX 5060 8GB (sm_120), CUDA 12.8
  export TORCH_CUDA_ARCH_LIST="12.0"
"""

import csv
from pathlib import Path

import torch

try:
    from transformers import GPT2Model, GPT2Config
except ImportError:
    raise ImportError(
        "transformers is required.\n"
        "Install: pip install transformers"
    )

from ghostweight_triton import replace_linears, get_sparsity_report

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

WARMUP = 10
ITERS  = 100
DTYPE  = torch.float16

# (batch_size, seq_len) pairs to benchmark
SEQ_CONFIGS = [
    (1,   1),     # single-token decode — most critical for autoregressive LLMs
    (1,   64),    # short sequence
    (1,   128),   # medium sequence
    (8,   128),   # small batch
    (1,   512),   # long context
]

# Layer name patterns to exclude from replacement.
# lm_head, wte, wpe are not nn.Linear in GPT2Model, but listing them is safe.
# ln_* = LayerNorm, attn.c_proj = attention output projection (sensitive).
SKIP_PATTERNS = ["lm_head", "wte", "wpe", "ln_"]

MIN_FEATURES = 128

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH = RESULTS_DIR / "bench_gpt2.csv"


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

def _cuda_time_ms(fn, warmup: int = WARMUP, iters: int = ITERS) -> float:
    """
    GPU-accurate timing via CUDA Events.
    CUDA kernels are asynchronous: torch.cuda.Event timestamps are inserted
    into the CUDA command stream and measured by the GPU hardware.
    """
    for _ in range(warmup):
        with torch.no_grad():
            fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        with torch.no_grad():
            fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

def _load_gpt2(device: str) -> torch.nn.Module:
    """Load GPT-2 Small encoder body in fp16."""
    print("  Loading GPT-2 Small …", end=" ", flush=True)
    config = GPT2Config()   # default = gpt2-small (12 layers, d=768)
    model  = GPT2Model(config)
    model  = model.to(device=device, dtype=DTYPE)
    model.eval()
    n = sum(p.numel() for p in model.parameters())
    mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6
    print(f"done  ({n/1e6:.1f}M params, {mb:.0f} MB)")
    return model


def _make_input(bs: int, sl: int, device: str) -> dict:
    return {"input_ids": torch.randint(0, 50257, (bs, sl), device=device)}


def _calibrate(model: torch.nn.Module, device: str, n: int = 5) -> None:
    """Run n random forward passes so GhostLinear masks stabilise."""
    for _ in range(n):
        inp = _make_input(bs=4, sl=64, device=device)
        with torch.no_grad():
            model(**inp)


# ---------------------------------------------------------------------------
# Table helpers
# ---------------------------------------------------------------------------

_HDR = (
    f"  {'BS':>4}  {'SL':>5}  {'Baseline(ms)':>14}"
    f"  {'Ghost(ms)':>12}  {'Speedup':>9}  {'Improv%':>9}"
)
_SEP = "  " + "─" * (len(_HDR) - 2)


def _print_row(bs: int, sl: int, ms_b: float, ms_g: float) -> None:
    sp = ms_b / ms_g if ms_g > 0 else float("inf")
    pc = (ms_b - ms_g) / ms_b * 100
    print(f"  {bs:>4}  {sl:>5}  {ms_b:>14.4f}"
          f"  {ms_g:>12.4f}  {sp:>8.3f}x  {pc:>8.1f}%")


def _save_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n  Results saved → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required.")

    device = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = False  # keep fp16 precision

    print("=" * 65)
    print("GhostWeight-Triton  |  GPT-2 Small Benchmark")
    print("=" * 65)
    print(f"  Device  : {torch.cuda.get_device_name(device)}")
    print(f"  CUDA    : {torch.version.cuda}")
    print(f"  DType   : {DTYPE}")
    print(f"  Warmup  : {WARMUP}  |  Measured: {ITERS} iters")

    # ── Load two independent model instances ──────────────────────────────
    print()
    model_baseline = _load_gpt2(device)
    model_ghost    = _load_gpt2(device)

    # ── Apply GhostWeight ─────────────────────────────────────────────────
    print("  Applying replace_linears …", end=" ", flush=True)
    replaced = replace_linears(
        model_ghost,
        threshold=1e-6,
        skip_patterns=SKIP_PATTERNS,
        min_features=MIN_FEATURES,
    )
    print(f"done  ({len(replaced)} layers)")

    # ── Calibrate ─────────────────────────────────────────────────────────
    print("  Calibrating masks …", end=" ", flush=True)
    _calibrate(model_ghost, device, n=5)
    print("done")

    # ── Sparsity report ───────────────────────────────────────────────────
    # One more forward to refresh current_sparsity after calibration
    with torch.no_grad():
        model_ghost(**_make_input(4, 64, device))

    report = get_sparsity_report(model_ghost)
    avg_sp = 0.0
    if report:
        sps    = list(report.values())
        avg_sp = sum(sps) / len(sps)
        print(f"\n  Sparsity report ({len(report)} layers):")
        print(f"    avg={avg_sp:.3f}  min={min(sps):.3f}  max={max(sps):.3f}")
        print("    Top-10 sparsest layers:")
        for name, sp in sorted(report.items(), key=lambda x: -x[1])[:10]:
            bar = "▓" * int(sp * 25)
            print(f"      {name:<55} {sp:.3f}  {bar}")

    # ── Benchmark ─────────────────────────────────────────────────────────
    print(f"\n  {'─'*63}")
    print("  Benchmark Results")
    print(f"  {'─'*63}")
    print(_HDR)
    print(_SEP)

    all_rows: list[dict] = []
    for bs, sl in SEQ_CONFIGS:
        inp = _make_input(bs, sl, device)

        ms_b = _cuda_time_ms(lambda: model_baseline(**inp))
        ms_g = _cuda_time_ms(lambda: model_ghost(**inp))

        sp = ms_b / ms_g if ms_g > 0 else float("inf")
        pc = (ms_b - ms_g) / ms_b * 100

        _print_row(bs, sl, ms_b, ms_g)
        all_rows.append({
            "model":           "gpt2-small",
            "batch_size":      bs,
            "seq_len":         sl,
            "ms_baseline":     f"{ms_b:.4f}",
            "ms_ghost":        f"{ms_g:.4f}",
            "speedup":         f"{sp:.3f}",
            "pct_improvement": f"{pc:.1f}",
            "n_replaced":      len(replaced),
            "avg_sparsity":    f"{avg_sp:.3f}",
        })

    sps = [float(r["speedup"]) for r in all_rows]
    print(f"\n  Speedup  min={min(sps):.3f}x  max={max(sps):.3f}x  "
          f"avg={sum(sps)/len(sps):.3f}x")

    _save_csv(all_rows, CSV_PATH)


if __name__ == "__main__":
    main()
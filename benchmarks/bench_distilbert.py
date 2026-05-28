"""
bench_distilbert.py — GhostWeight inference benchmark on DistilBERT.

Pipeline
--------
  1. Load DistilBertModel (base-uncased) in fp16.
  2. Baseline forward-pass latency.
  3. replace_linears (skip embeddings, LayerNorm).
  4. Calibrate masks with 5 random passes.
  5. Ghost forward-pass latency.
  6. Sparsity report.
  7. Save CSV to benchmarks/results/bench_distilbert.csv.

DistilBERT reference numbers
-----------------------------
  hidden_size       = 768
  intermediate_size = 3072
  num_layers        = 6
  num_heads         = 12
  vocab_size        = 30522
  model_size_fp16  ≈ 260 MB

Methodology
-----------
  - torch.cuda.Event timing
  - WARMUP=10  ITERS=100
  - torch.no_grad() throughout

Usage
-----
  cd ghostweight-triton/
  python benchmarks/bench_distilbert.py
"""

import csv
from pathlib import Path

import torch

try:
    from transformers import DistilBertModel, DistilBertConfig
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

SEQ_CONFIGS = [
    (1,   32),    # short single sample
    (1,   128),   # medium single sample
    (8,   128),   # small batch
    (16,  128),   # medium batch
    (1,   512),   # long sequence (DistilBERT supports up to 512)
]

# Skip embeddings and LayerNorm.
# DistilBERT FFN uses lin1 / lin2 (both in_features=768 or 3072) — do NOT skip.
SKIP_PATTERNS = ["embedding", "LayerNorm"]

MIN_FEATURES = 128

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH = RESULTS_DIR / "bench_distilbert.csv"


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

def _cuda_time_ms(fn, warmup: int = WARMUP, iters: int = ITERS) -> float:
    """
    GPU-accurate timing via CUDA Events.

    Why CUDA Events and not time.perf_counter:
      CUDA launches are asynchronous.  perf_counter() returns to the CPU
      before the GPU completes the work.  CUDA Events are timestamped by the
      GPU hardware, recording exact start/end of the command stream.
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

def _load_distilbert(device: str) -> torch.nn.Module:
    print("  Loading DistilBERT-base-uncased …", end=" ", flush=True)
    config = DistilBertConfig()
    model  = DistilBertModel(config)
    model  = model.to(device=device, dtype=DTYPE)
    model.eval()
    n  = sum(p.numel() for p in model.parameters())
    mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6
    print(f"done  ({n/1e6:.1f}M params, {mb:.0f} MB)")
    return model


def _make_input(bs: int, sl: int, device: str) -> dict:
    return {
        "input_ids":      torch.randint(0, 30522, (bs, sl), device=device),
        "attention_mask": torch.ones(bs, sl, dtype=torch.long, device=device),
    }


def _calibrate(model: torch.nn.Module, device: str, n: int = 5) -> None:
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
    torch.backends.cuda.matmul.allow_tf32 = False

    print("=" * 65)
    print("GhostWeight-Triton  |  DistilBERT Benchmark")
    print("=" * 65)
    print(f"  Device  : {torch.cuda.get_device_name(device)}")
    print(f"  CUDA    : {torch.version.cuda}")
    print(f"  DType   : {DTYPE}")
    print(f"  Warmup  : {WARMUP}  |  Measured: {ITERS} iters")

    # ── Load ─────────────────────────────────────────────────────────────
    print()
    model_baseline = _load_distilbert(device)
    model_ghost    = _load_distilbert(device)

    # ── Replace ───────────────────────────────────────────────────────────
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
            "model":           "distilbert-base-uncased",
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
"""
bench_kernels.py — microbenchmarks for individual GhostWeight Triton kernels.

Kernels measured
----------------
  dead_neuron_mask   — activation sparsity detection
  gather_compact     — sparse gather of activations and weights
  ghostweight_matmul — fused sparse matmul, compared against torch.matmul

Methodology
-----------
  - torch.cuda.Event timing, inserted directly into the CUDA stream.
    This is more accurate than time.perf_counter for GPU work because
    CUDA operations are asynchronous: wall-clock timing races the CPU
    ahead of the actual GPU completion.
  - 10 warmup iterations  (excludes JIT / first-launch overhead)
  - 100 measured iterations
  - mean time = total_elapsed / 100

Output
------
  Formatted tables printed to stdout.
  CSV saved to benchmarks/results/bench_kernels.csv.

Usage
-----
  cd ghostweight-triton/
  python benchmarks/bench_kernels.py

System target
-------------
  AMD Ryzen 7 7700, RTX 5060 8GB (sm_120), CUDA 12.8
  export TORCH_CUDA_ARCH_LIST="12.0"   # required for Triton on sm_120
"""

import csv
from pathlib import Path

import torch

from ghostweight_triton.kernels import dead_neuron_mask, gather_compact, ghostweight_matmul

# ---------------------------------------------------------------------------
# Global config
# ---------------------------------------------------------------------------

WARMUP = 10
ITERS  = 100
DTYPE  = torch.float16

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH = RESULTS_DIR / "bench_kernels.csv"

# (M, K, N) shapes — chosen to cover GPT-2 and DistilBERT projections
SHAPES = [
    (1,    128,   256),
    (1,    768,  3072),
    (1,    768,  2304),   # GPT-2 attention Wqkv
    (16,   768,  3072),
    (128,  768,  3072),
    (128, 3072,   768),
]

SPARSITY_LEVELS = [0.0, 0.3, 0.5, 0.7, 0.9]


# ---------------------------------------------------------------------------
# Timing primitive
# ---------------------------------------------------------------------------

def _cuda_time_ms(fn, warmup: int = WARMUP, iters: int = ITERS) -> float:
    """
    Measure the average GPU time of `fn()` in milliseconds.

    We use torch.cuda.Event rather than time.perf_counter because:
      * CUDA kernels are launched asynchronously on the CUDA stream.
      * perf_counter measures CPU wall-clock time and returns before
        the GPU has finished, giving falsely low numbers unless
        torch.cuda.synchronize() is called — which itself adds overhead.
      * CUDA Events are hardware timestamps inserted directly into the
        command stream, giving sub-microsecond accurate GPU measurements.
    """
    # --- warmup ---
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()   # flush warmup work

    # --- measure ---
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        fn()
    end.record()

    torch.cuda.synchronize()   # wait for all measured iterations
    return start.elapsed_time(end) / iters   # ms per call


# ---------------------------------------------------------------------------
# dead_neuron_mask benchmark
# ---------------------------------------------------------------------------

def _bench_mask(device: str) -> list[dict]:
    rows: list[dict] = []
    print("\n╔═══ dead_neuron_mask ══════════════════════════════════════╗")
    print(f"  {'M':>6}  {'K':>6}  {'ms/iter':>10}  {'eff GB/s':>10}")
    print("  " + "─" * 42)

    for M, K, _N in SHAPES:
        A = torch.randn(M, K, dtype=DTYPE, device=device)
        # Bytes read: the input activation tensor
        nbytes = A.numel() * A.element_size()

        ms  = _cuda_time_ms(lambda: dead_neuron_mask(A, threshold=1e-6))
        gbs = (nbytes / 1e9) / (ms / 1e3)

        print(f"  {M:>6}  {K:>6}  {ms:>10.4f}  {gbs:>10.2f}")
        rows.append({
            "kernel": "dead_neuron_mask",
            "M": M, "K": K, "N": _N,
            "sparsity": "—",
            "ms_kernel": f"{ms:.4f}",
            "ms_ref": "—",
            "speedup": "—",
            "eff_gbs": f"{gbs:.2f}",
        })
    print("╚═══════════════════════════════════════════════════════════╝")
    return rows


# ---------------------------------------------------------------------------
# gather_compact benchmark
# ---------------------------------------------------------------------------

def _bench_gather(device: str) -> list[dict]:
    rows: list[dict] = []
    print("\n╔═══ gather_compact ════════════════════════════════════════╗")
    print(f"  {'M':>6}  {'K':>6}  {'N':>6}  {'sparsity':>9}"
          f"  {'ms/iter':>10}  {'eff GB/s':>10}")
    print("  " + "─" * 55)

    for M, K, N in SHAPES:
        for sparsity in [0.3, 0.5, 0.7]:
            A = torch.randn(M, K, dtype=DTYPE, device=device)
            W = torch.randn(K, N, dtype=DTYPE, device=device)
            n_alive  = max(1, int(K * (1.0 - sparsity)))
            alive_idx = torch.randperm(K, device=device)[:n_alive].to(torch.int64)

            # Effective bytes: A[:, alive] + W[alive, :]
            nbytes = (M * n_alive + n_alive * N) * A.element_size()

            ms  = _cuda_time_ms(lambda: gather_compact(A, W, alive_idx))
            gbs = (nbytes / 1e9) / (ms / 1e3)

            print(f"  {M:>6}  {K:>6}  {N:>6}  {sparsity:>9.1f}"
                  f"  {ms:>10.4f}  {gbs:>10.2f}")
            rows.append({
                "kernel": "gather_compact",
                "M": M, "K": K, "N": N,
                "sparsity": sparsity,
                "ms_kernel": f"{ms:.4f}",
                "ms_ref": "—",
                "speedup": "—",
                "eff_gbs": f"{gbs:.2f}",
            })
    print("╚═══════════════════════════════════════════════════════════╝")
    return rows


# ---------------------------------------------------------------------------
# ghostweight_matmul benchmark
# ---------------------------------------------------------------------------

def _bench_matmul(device: str) -> list[dict]:
    rows: list[dict] = []
    print("\n╔═══ ghostweight_matmul vs torch.matmul ════════════════════╗")
    print(f"  {'M':>6}  {'K':>6}  {'N':>6}  {'sparsity':>9}"
          f"  {'ghost ms':>10}  {'dense ms':>10}  {'speedup':>9}")
    print("  " + "─" * 65)

    for M, K, N in SHAPES:
        for sparsity in SPARSITY_LEVELS:
            A = torch.randn(M, K, dtype=DTYPE, device=device)
            W = torch.randn(K, N, dtype=DTYPE, device=device)
            n_alive  = max(1, int(K * (1.0 - sparsity)))
            alive_idx = torch.randperm(K, device=device)[:n_alive].to(torch.int64)

            ms_dense = _cuda_time_ms(lambda: torch.matmul(A, W))
            ms_ghost = _cuda_time_ms(lambda: ghostweight_matmul(A, W, alive_idx))
            speedup  = ms_dense / ms_ghost if ms_ghost > 0.0 else float("inf")

            print(f"  {M:>6}  {K:>6}  {N:>6}  {sparsity:>9.1f}"
                  f"  {ms_ghost:>10.4f}  {ms_dense:>10.4f}  {speedup:>8.3f}x")
            rows.append({
                "kernel": "ghostweight_matmul",
                "M": M, "K": K, "N": N,
                "sparsity": sparsity,
                "ms_kernel": f"{ms_ghost:.4f}",
                "ms_ref": f"{ms_dense:.4f}",
                "speedup": f"{speedup:.3f}",
                "eff_gbs": "—",
            })
    print("╚═══════════════════════════════════════════════════════════╝")
    return rows


# ---------------------------------------------------------------------------
# CSV writer
# ---------------------------------------------------------------------------

def _save_csv(all_rows: list[dict], path: Path) -> None:
    if not all_rows:
        return
    # Gather all unique field names in insertion order
    fieldnames = list(dict.fromkeys(k for row in all_rows for k in row))
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\n  Results saved → {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for benchmarking.")

    device = "cuda"
    print("=" * 62)
    print("GhostWeight-Triton  |  Kernel Microbenchmarks")
    print("=" * 62)
    print(f"  Device  : {torch.cuda.get_device_name(device)}")
    print(f"  CUDA    : {torch.version.cuda}")
    print(f"  DType   : {DTYPE}")
    print(f"  Warmup  : {WARMUP}  |  Measured: {ITERS} iters")

    all_rows: list[dict] = []
    all_rows += _bench_mask(device)
    all_rows += _bench_gather(device)
    all_rows += _bench_matmul(device)

    _save_csv(all_rows, CSV_PATH)


if __name__ == "__main__":
    main()
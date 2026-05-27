"""
Sparsity statistics and display utilities.
"""

import torch


def sparsity_stats(mask: torch.Tensor) -> dict:
    """
    Compute statistics from a dead-neuron mask.

    Parameters
    ----------
    mask : torch.Tensor
        [K] int32 mask from dead_neuron_mask(). 1=alive, 0=dead.

    Returns
    -------
    dict with keys: total, alive, dead, pct_dead, pct_alive
    """
    assert mask.ndim == 1, f"Expected 1D mask, got shape {mask.shape}"
    total = mask.numel()
    alive = int(mask.sum().item())
    dead = total - alive
    return {
        "total":     total,
        "alive":     alive,
        "dead":      dead,
        "pct_dead":  dead / total if total > 0 else 0.0,
        "pct_alive": alive / total if total > 0 else 0.0,
    }


def log_sparsity(
    report: dict[str, float],
    title: str = "GhostWeight Sparsity Report",
) -> None:
    """
    Pretty-print a sparsity report from get_sparsity_report().

    Parameters
    ----------
    report : dict[str, float]
        {layer_name: sparsity_fraction}
    title : str
        Header string.
    """
    if not report:
        print(f"[{title}] No GhostLinear layers found.")
        return

    max_name_len = max(len(n) for n in report)
    col_width = max(max_name_len, 20)
    bar_width = 40

    sep = "─" * (col_width + bar_width + 20)
    print(f"\n{sep}")
    print(f"  {title}")
    print(sep)
    print(f"  {'Layer':<{col_width}}  {'Sparsity':>8}  {'Dead neurons'}")
    print(f"  {'─' * col_width}  {'─' * 8}  {'─' * bar_width}")

    total_sparsity = 0.0
    for name, sparsity in report.items():
        filled = int(sparsity * bar_width)
        bar = "█" * filled + "░" * (bar_width - filled)
        print(f"  {name:<{col_width}}  {sparsity:>7.1%}  {bar}")
        total_sparsity += sparsity

    avg = total_sparsity / len(report)
    print(sep)
    print(f"  {'Average':<{col_width}}  {avg:>7.1%}")
    print(f"{sep}\n")
"""
Environment compatibility checks for GhostWeight-Triton.
"""

import os
import warnings


def check_environment(verbose: bool = True) -> dict:
    """
    Check system environment for sm_120 (Blackwell) compatibility.

    Parameters
    ----------
    verbose : bool
        If True, print formatted report.

    Returns
    -------
    dict with keys:
        torch_version, triton_version, cuda_version, gpu_name,
        compute_cap, arch_list_set, sm120_ready, warnings
    """
    info: dict = {"warnings": []}

    # Torch
    try:
        import torch
        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            cc = torch.cuda.get_device_capability(0)
            info["compute_cap"] = f"{cc[0]}.{cc[1]}"
            info["cuda_version"] = torch.version.cuda or "unknown"
        else:
            info["gpu_name"] = "N/A"
            info["compute_cap"] = "N/A"
            info["cuda_version"] = "N/A"
            info["warnings"].append("CUDA not available — kernels will not run.")
    except ImportError:
        info["torch_version"] = "NOT INSTALLED"
        info["warnings"].append("PyTorch is not installed.")

    # Triton
    try:
        import triton
        info["triton_version"] = triton.__version__
    except ImportError:
        info["triton_version"] = "NOT INSTALLED"
        info["warnings"].append(
            "Triton is not installed. Install with: pip install triton"
        )

    # TORCH_CUDA_ARCH_LIST
    arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST", "")
    info["arch_list_set"] = bool(arch_list)
    if not arch_list:
        msg = (
            "TORCH_CUDA_ARCH_LIST is not set. "
            "For RTX 5060 (sm_120), run: export TORCH_CUDA_ARCH_LIST='12.0'"
        )
        info["warnings"].append(msg)
        warnings.warn(msg, UserWarning, stacklevel=2)

    # sm_120 readiness
    cc = info.get("compute_cap", "0.0")
    info["sm120_ready"] = cc == "12.0"
    if info.get("cuda_available") and not info["sm120_ready"]:
        info["warnings"].append(
            f"GPU compute capability is {cc}, not 12.0. "
            "Kernels will still run but are tuned for sm_120."
        )

    if verbose:
        _print_report(info)

    return info


def _print_report(info: dict) -> None:
    ok = "✓"
    warn = "⚠"
    fail = "✗"

    def status(condition: bool) -> str:
        return ok if condition else fail

    print("\n┌─────────────────────────────────────────┐")
    print("│   GhostWeight-Triton Environment Check  │")
    print("└─────────────────────────────────────────┘")
    print(f"  {status(info.get('torch_version') != 'NOT INSTALLED')} "
          f"PyTorch       {info.get('torch_version', 'N/A')}")
    print(f"  {status(info.get('triton_version') != 'NOT INSTALLED')} "
          f"Triton        {info.get('triton_version', 'N/A')}")
    print(f"  {status(info.get('cuda_available', False))} "
          f"CUDA          {info.get('cuda_version', 'N/A')}")
    print(f"  {status(info.get('cuda_available', False))} "
          f"GPU           {info.get('gpu_name', 'N/A')}")
    print(f"  {status(info.get('sm120_ready', False))} "
          f"Compute Cap   {info.get('compute_cap', 'N/A')} "
          f"{'(sm_120 ✓)' if info.get('sm120_ready') else ''}")
    print(f"  {status(info.get('arch_list_set', False))} "
          f"ARCH_LIST     "
          f"{'set' if info.get('arch_list_set') else 'NOT SET → export TORCH_CUDA_ARCH_LIST=12.0'}")

    if info["warnings"]:
        print(f"\n  {warn} Warnings:")
        for w in info["warnings"]:
            print(f"    • {w}")
    else:
        print(f"\n  {ok} All checks passed.")
    print()
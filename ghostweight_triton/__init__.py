"""
GhostWeight-Triton
==================
Training-free inference sparsity via dead neuron masking.

Author : Manjit Pokhrel, Nepal
License: MIT
"""

from ghostweight_triton.kernels import (
    dead_neuron_mask,
    ghostweight_matmul,
    gather_compact,
)
from ghostweight_triton.module import GhostLinear
from ghostweight_triton.replace import replace_linears, get_sparsity_report

__version__ = "0.1.0"
__author__ = "Manjit Pokhrel"

__all__ = [
    "dead_neuron_mask",
    "ghostweight_matmul",
    "gather_compact",
    "GhostLinear",
    "replace_linears",
    "get_sparsity_report",
]
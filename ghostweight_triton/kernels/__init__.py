"""
Triton kernel wrappers.
Import these — not the raw kernel functions — in user-facing code.
"""

from ghostweight_triton.kernels.mask import dead_neuron_mask
from ghostweight_triton.kernels.gather import gather_compact
from ghostweight_triton.kernels.matmul import ghostweight_matmul

__all__ = [
    "dead_neuron_mask",
    "gather_compact",
    "ghostweight_matmul",
]
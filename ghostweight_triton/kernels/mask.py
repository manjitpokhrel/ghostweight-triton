"""
Dead neuron detection kernel.

A neuron (column j of the activation matrix) is "dead" if:
    max over all rows i of |A[i, j]| <= threshold

Kernel strategy:
  - One Triton program per column (neuron)
  - Each program iterates over rows in BLOCK_M chunks
  - Computes running max of absolute values
  - Writes 1 (alive) or 0 (dead)

This is a column-wise reduction over [M, K].
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _dead_neuron_mask_kernel(
    # Pointers
    A_ptr,
    mask_ptr,
    # Dimensions
    M,
    K,
    # Strides (elements, not bytes)
    stride_am,
    stride_ak,
    # Threshold
    threshold,
    # Compile-time tile size
    BLOCK_M: tl.constexpr,
):
    """
    One program instance = one column (neuron).
    Iterates over all M rows, accumulates running max of |activation|.
    """
    col = tl.program_id(0)

    if col >= K:
        return

    running_max = tl.zeros([BLOCK_M], dtype=tl.float32)

    for m_start in range(0, M, BLOCK_M):
        m_offs = m_start + tl.arange(0, BLOCK_M)
        m_mask = m_offs < M

        ptrs = A_ptr + m_offs * stride_am + col * stride_ak
        vals = tl.load(ptrs, mask=m_mask, other=0.0).to(tl.float32)

        running_max = tl.maximum(running_max, tl.abs(vals))

    col_max = tl.max(running_max, axis=0)
    is_alive = (col_max > threshold).to(tl.int32)
    tl.store(mask_ptr + col, is_alive)


def dead_neuron_mask(
    activations: torch.Tensor,
    threshold: float = 1e-6,
    BLOCK_M: int = 512,
) -> torch.Tensor:
    """
    Detect dead neurons in an activation matrix.

    Parameters
    ----------
    activations : torch.Tensor
        Shape [M, K]. float16 or float32. Must be on CUDA.
    threshold : float
        Neurons with max |activation| <= threshold are dead.
    BLOCK_M : int
        Row tile size for reduction. Power of 2.

    Returns
    -------
    torch.Tensor
        Shape [K], dtype int32. 1 = alive, 0 = dead.
    """
    assert activations.ndim == 2, (
        f"dead_neuron_mask expects [M, K], got shape {activations.shape}"
    )
    assert activations.is_cuda, "activations must be on CUDA"

    activations = activations.contiguous()
    M, K = activations.shape

    mask = torch.empty(K, dtype=torch.int32, device=activations.device)

    _dead_neuron_mask_kernel[(K,)](
        activations,
        mask,
        M, K,
        activations.stride(0),
        activations.stride(1),
        threshold,
        BLOCK_M=BLOCK_M,
    )

    return mask
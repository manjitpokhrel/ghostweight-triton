"""
Gather kernels for compacting activations and weights.

Two-step (non-fused) path:
    A_compact, W_compact = gather_compact(A, W, alive_indices)
    C = A_compact @ W_compact

The fused path (ghostweight_matmul) skips materialization.
These exist for:
    1. Correctness reference
    2. Cases where repeated matmuls share same alive_indices
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _gather_columns_kernel(
    A_ptr,
    idx_ptr,
    out_ptr,
    M, K, K_alive,
    stride_am, stride_ak,
    stride_om, stride_ok,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Gather columns from A: out[:, j] = A[:, idx[j]]"""
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    k_offs = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)

    m_mask = m_offs < M
    k_mask = k_offs < K_alive

    orig_cols = tl.load(idx_ptr + k_offs, mask=k_mask, other=0)

    a_ptrs = (
        A_ptr
        + m_offs[:, None] * stride_am
        + orig_cols[None, :] * stride_ak
    )
    mask_2d = m_mask[:, None] & k_mask[None, :]
    vals = tl.load(a_ptrs, mask=mask_2d, other=0.0)

    out_ptrs = (
        out_ptr
        + m_offs[:, None] * stride_om
        + k_offs[None, :] * stride_ok
    )
    tl.store(out_ptrs, vals, mask=mask_2d)


@triton.jit
def _gather_rows_kernel(
    W_ptr,
    idx_ptr,
    out_ptr,
    K, N, K_alive,
    stride_wk, stride_wn,
    stride_ok, stride_on,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Gather rows from W: out[j, :] = W[idx[j], :]"""
    pid_k = tl.program_id(0)
    pid_n = tl.program_id(1)

    k_offs = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    k_mask = k_offs < K_alive
    n_mask = n_offs < N

    orig_rows = tl.load(idx_ptr + k_offs, mask=k_mask, other=0)

    w_ptrs = (
        W_ptr
        + orig_rows[:, None] * stride_wk
        + n_offs[None, :] * stride_wn
    )
    mask_2d = k_mask[:, None] & n_mask[None, :]
    vals = tl.load(w_ptrs, mask=mask_2d, other=0.0)

    out_ptrs = (
        out_ptr
        + k_offs[:, None] * stride_ok
        + n_offs[None, :] * stride_on
    )
    tl.store(out_ptrs, vals, mask=mask_2d)


def gather_compact(
    activations: torch.Tensor,
    weights: torch.Tensor,
    alive_indices: torch.Tensor,
    BLOCK_M: int = 64,
    BLOCK_K: int = 64,
    BLOCK_N: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compact activations and weights to alive neurons only.

    Parameters
    ----------
    activations   : [M, K]
    weights       : [K, N]
    alive_indices : [K_alive], int64, sorted

    Returns
    -------
    A_compact : [M, K_alive]
    W_compact : [K_alive, N]
    """
    assert activations.ndim == 2 and weights.ndim == 2
    assert activations.is_cuda and weights.is_cuda

    M, K = activations.shape
    K_w, N = weights.shape
    assert K == K_w, f"K mismatch: A K={K}, W K={K_w}"

    activations = activations.contiguous()
    weights = weights.contiguous()
    alive_indices = alive_indices.contiguous().to(torch.int64)
    K_alive = alive_indices.shape[0]

    A_compact = torch.empty(
        (M, K_alive), dtype=activations.dtype, device=activations.device
    )
    W_compact = torch.empty(
        (K_alive, N), dtype=weights.dtype, device=weights.device
    )

    # Gather columns of A
    grid_a = (triton.cdiv(M, BLOCK_M), triton.cdiv(K_alive, BLOCK_K))
    _gather_columns_kernel[grid_a](
        activations, alive_indices, A_compact,
        M, K, K_alive,
        activations.stride(0), activations.stride(1),
        A_compact.stride(0), A_compact.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K,
    )

    # Gather rows of W
    grid_w = (triton.cdiv(K_alive, BLOCK_K), triton.cdiv(N, BLOCK_N))
    _gather_rows_kernel[grid_w](
        weights, alive_indices, W_compact,
        K, N, K_alive,
        weights.stride(0), weights.stride(1),
        W_compact.stride(0), W_compact.stride(1),
        BLOCK_K=BLOCK_K, BLOCK_N=BLOCK_N,
    )

    return A_compact, W_compact
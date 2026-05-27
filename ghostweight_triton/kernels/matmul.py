"""
Fused GhostWeight sparse matmul.

C [M, N] = A[:, alive_indices] @ W[alive_indices, :]

The kernel iterates over alive_indices in BLOCK_K chunks.
For each chunk:
  1. Load actual indices
  2. Gather A columns and W rows (in-register, no intermediate tensor)
  3. Accumulate via tl.dot

One kernel launch replaces: gather_columns + gather_rows + dense matmul.

Grid:
  axis 0 → tiles over M
  axis 1 → tiles over N

Constraints (tl.dot):
  BLOCK_M, BLOCK_N, BLOCK_K >= 16 and powers of 2.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _ghostweight_matmul_kernel(
    A_ptr,
    W_ptr,
    C_ptr,
    idx_ptr,
    M, N, K, K_alive,
    stride_am, stride_ak,
    stride_wk, stride_wn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for k_start in range(0, K_alive, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offs < K_alive

        alive = tl.load(idx_ptr + k_offs, mask=k_mask, other=0)

        # A[m_offs, alive] → [BLOCK_M, BLOCK_K]
        a_ptrs = (
            A_ptr
            + m_offs[:, None] * stride_am
            + alive[None, :] * stride_ak
        )
        a_mask = (m_offs[:, None] < M) & k_mask[None, :]
        a_tile = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # W[alive, n_offs] → [BLOCK_K, BLOCK_N]
        w_ptrs = (
            W_ptr
            + alive[:, None] * stride_wk
            + n_offs[None, :] * stride_wn
        )
        w_mask = k_mask[:, None] & (n_offs[None, :] < N)
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)

        acc += tl.dot(a_tile, w_tile)

    c_ptrs = (
        C_ptr
        + m_offs[:, None] * stride_cm
        + n_offs[None, :] * stride_cn
    )
    c_mask = (m_offs[:, None] < M) & (n_offs[None, :] < N)
    tl.store(c_ptrs, acc.to(C_ptr.dtype.element_ty), mask=c_mask)


def ghostweight_matmul(
    activations: torch.Tensor,
    weights: torch.Tensor,
    alive_indices: torch.Tensor,
    BLOCK_M: int = 64,
    BLOCK_N: int = 64,
    BLOCK_K: int = 32,
) -> torch.Tensor:
    """
    Fused sparse matmul skipping dead neurons.

    Computes C = A[:, alive_indices] @ W[alive_indices, :]
    without materializing gathered sub-matrices.

    Parameters
    ----------
    activations   : [M, K]
    weights       : [K, N]
    alive_indices : [K_alive], int64, sorted
    BLOCK_M, BLOCK_N, BLOCK_K : int
        Tile sizes, >= 16, powers of 2.

    Returns
    -------
    torch.Tensor [M, N]
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

    C = torch.empty((M, N), dtype=activations.dtype, device=activations.device)

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    _ghostweight_matmul_kernel[grid](
        activations, weights, C, alive_indices,
        M, N, K, K_alive,
        activations.stride(0), activations.stride(1),
        weights.stride(0), weights.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )

    return C
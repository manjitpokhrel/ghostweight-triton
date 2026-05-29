"""
GhostLinear: drop-in nn.Linear replacement with dead-neuron sparsity.
"""

import torch
import torch.nn as nn

from ghostweight_triton.kernels.mask import dead_neuron_mask
from ghostweight_triton.kernels.matmul import ghostweight_matmul


class GhostLinear(nn.Module):
    """
    Drop-in nn.Linear replacement with training-free inference sparsity.

    At inference time:
      1. Detects dead neurons (|activation| <= threshold across batch)
      2. If sparsity >= min_sparsity: fused Triton sparse matmul
      3. Otherwise: dense fallback (overhead > benefit for low sparsity)

    Two masking modes:
      Dynamic (default): recomputed every forward pass.
      Frozen:  computed once via freeze_mask(), reused every call.

    Parameters
    ----------
    in_features, out_features, bias : same as nn.Linear
    threshold : float
        Dead-neuron detection threshold.
    min_sparsity_to_activate : float
        Minimum dead fraction before sparse path is used.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        threshold: float = 1e-6,
        min_sparsity_to_activate: float = 0.1,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.threshold = threshold
        self.min_sparsity_to_activate = min_sparsity_to_activate

        factory = {"device": device, "dtype": dtype}
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, **factory)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory))
        else:
            self.register_parameter("bias", None)

        self.register_buffer("_frozen_indices", None)
        self._mask_frozen: bool = False
        self._last_sparsity: float = 0.0

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        threshold: float = 1e-6,
        min_sparsity_to_activate: float = 0.1,
    ) -> "GhostLinear":
        """
        Convert existing nn.Linear to GhostLinear.
        Shares weight/bias tensors — no memory copy.
        """
        ghost = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
            threshold=threshold,
            min_sparsity_to_activate=min_sparsity_to_activate,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
        )
        ghost.weight = linear.weight
        if linear.bias is not None:
            ghost.bias = linear.bias
        return ghost

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : [..., in_features] → [..., out_features]
        Supports arbitrary batch dimensions.
        """
        orig_shape = x.shape
        x_2d = x.reshape(-1, self.in_features)

        # Weight: [out_features, in_features] → [in_features, out_features]
        W = self.weight.t().contiguous()

        # Determine alive indices
        if self._mask_frozen and self._frozen_indices is not None:
            alive = self._frozen_indices
        else:
            mask = dead_neuron_mask(x_2d, threshold=self.threshold)
            alive = torch.nonzero(mask, as_tuple=False).squeeze(-1).to(torch.int64)

        K_alive = alive.shape[0]
        sparsity = 1.0 - K_alive / self.in_features
        self._last_sparsity = float(sparsity)

        # Choose compute path
        if sparsity < self.min_sparsity_to_activate or K_alive == 0:
            out = x_2d @ W
        else:
            out = ghostweight_matmul(x_2d, W, alive)

        if self.bias is not None:
            out = out + self.bias

        return out.reshape(*orig_shape[:-1], self.out_features)

    def freeze_mask(self, calibration_input: torch.Tensor) -> None:
        """
        Freeze dead-neuron mask from calibration data.
        All future forward passes reuse this mask.
        """
        with torch.no_grad():
            x_2d = calibration_input.reshape(-1, self.in_features)
            mask = dead_neuron_mask(x_2d, threshold=self.threshold)
            self._frozen_indices = (
                torch.nonzero(mask, as_tuple=False).squeeze(-1).to(torch.int64)
            )
            self._mask_frozen = True

            k_alive = int(self._frozen_indices.numel())
            self._last_sparsity = float(1.0 - k_alive / self.in_features)

    def unfreeze_mask(self) -> None:
        """Switch back to dynamic per-forward masking."""
        self._mask_frozen = False
        self._frozen_indices = None

    @property
    def current_sparsity(self) -> float:
        """Sparsity from current frozen mask or most recent forward pass."""
        return self._last_sparsity

    @property
    def mask_frozen(self) -> bool:
        return self._mask_frozen

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"bias={self.bias is not None}, "
            f"threshold={self.threshold}, "
            f"frozen={self._mask_frozen}, "
            f"sparsity={self._last_sparsity:.1%}"
        )
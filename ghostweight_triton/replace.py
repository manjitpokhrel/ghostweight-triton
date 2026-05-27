"""
Utility to patch all nn.Linear layers in a model with GhostLinear.
"""

import torch.nn as nn
from .module import GhostLinear


def replace_linears(
    model: nn.Module,
    threshold: float = 1e-6,
    skip_patterns: list[str] | None = None,
    min_features: int = 128,
) -> dict[str, float]:
    """
    Recursively replace nn.Linear modules with GhostLinear.
    
    Args:
        model:         the model to patch (modified in-place)
        threshold:     dead neuron threshold
        skip_patterns: list of substrings — if a module name contains any
                      of these, skip replacement (e.g., ["lm_head", "embed"])
        min_features:  don't replace layers smaller than this (overhead > gain)
    
    Returns:
        dict mapping replaced module names to their in_features
    """
    skip_patterns = skip_patterns or []
    replaced = {}

    for name, module in model.named_modules():
        # Find all nn.Linear children (not nested deeper)
        children_to_replace = {}
        for child_name, child in module.named_children():
            if not isinstance(child, nn.Linear):
                continue
            if isinstance(child, GhostLinear):
                continue  # already replaced

            full_name = f"{name}.{child_name}" if name else child_name

            # Skip patterns
            if any(pat in full_name for pat in skip_patterns):
                continue

            # Skip small layers
            if child.in_features < min_features:
                continue

            children_to_replace[child_name] = child
            replaced[full_name] = child.in_features

        # Perform replacement
        for child_name, child in children_to_replace.items():
            ghost = GhostLinear.from_linear(child, threshold=threshold)
            setattr(module, child_name, ghost)

    return replaced


def get_sparsity_report(model: nn.Module) -> dict[str, float]:
    """Get current sparsity for all GhostLinear layers."""
    report = {}
    for name, module in model.named_modules():
        if isinstance(module, GhostLinear):
            report[name] = module.current_sparsity
    return report
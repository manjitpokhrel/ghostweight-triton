"""
test_replace.py — correctness tests for replace_linears and get_sparsity_report.

Verified properties
-------------------
replace_linears
  1.  All qualifying nn.Linear layers become GhostLinear.
  2.  Layers whose name matches a skip_pattern are NOT replaced.
  3.  Multiple skip patterns work simultaneously.
  4.  Layers with in_features < min_features are NOT replaced.
  5.  Return value is a dict[str, numeric].
  6.  Nested sub-modules are traversed recursively.
  7.  Model produces output of the same shape after replacement.
  8.  threshold is forwarded to every GhostLinear created.

get_sparsity_report
  9.  Returns an empty dict for a plain nn.Module with no GhostLinear.
  10. Keys match the names of GhostLinear layers in the model.
  11. Values are floats in [0, 1].
  12. Average sparsity is higher for zero input than for dense input.
"""

import pytest
import torch
import torch.nn as nn

from ghostweight_triton import GhostLinear, replace_linears, get_sparsity_report


# ---------------------------------------------------------------------------
# Nested model for recursion / skip-pattern tests
# ---------------------------------------------------------------------------

class _NestedModel(nn.Module):
    """
    Sub-module tree:
        encoder.fc1   Linear(128 → 256)
        encoder.fc2   Linear(256 → 128)
        head.linear   Linear(128 → 10)
        lm_head       Linear(10  → 5)   ← small, useful for skip testing
    """

    class _Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(128, 256)
            self.fc2 = nn.Linear(256, 128)

        def forward(self, x):
            return self.fc2(torch.relu(self.fc1(x)))

    class _Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(128, 10)

        def forward(self, x):
            return self.linear(x)

    def __init__(self):
        super().__init__()
        self.encoder = self._Encoder()
        self.head    = self._Head()
        self.lm_head = nn.Linear(10, 5)

    def forward(self, x):
        return self.lm_head(self.head(self.encoder(x)))


def _move(model: nn.Module, device: str, dtype: torch.dtype) -> nn.Module:
    return model.to(device=device, dtype=dtype)


# ---------------------------------------------------------------------------
# 1 : All qualifying layers replaced
# ---------------------------------------------------------------------------

class TestReplaceLinears:

    def test_qualifying_layers_become_ghost(self, device, toy_model):
        """
        _ToyMLP: fc1(128→512), fc2(512→128).
        With min_features=128 both have in_features >= 128 → replaced.
        """
        replace_linears(toy_model, min_features=128)
        assert isinstance(toy_model.fc1, GhostLinear), "fc1 must be GhostLinear"
        assert isinstance(toy_model.fc2, GhostLinear), "fc2 must be GhostLinear"

    # 3 : min_features skip -----------------------------------------------

    def test_layer_below_min_features_not_replaced(self, device, toy_model):
        """
        min_features=512: fc1 has in_features=128 < 512 → NOT replaced.
                          fc2 has in_features=512 >= 512 → replaced.
        """
        replace_linears(toy_model, min_features=512)
        # fc1 must remain a plain nn.Linear (not GhostLinear)
        assert not isinstance(toy_model.fc1, GhostLinear), \
            "fc1 (in=128) must not be replaced when min_features=512"
        assert isinstance(toy_model.fc2, GhostLinear), \
            "fc2 (in=512) must be replaced when min_features=512"

    # 2 : Single skip pattern --------------------------------------------

    def test_single_skip_pattern(self, device):
        model = _move(_NestedModel(), device, torch.float16)
        replace_linears(model, skip_patterns=["lm_head"], min_features=1)
        assert not isinstance(model.lm_head, GhostLinear), \
            "lm_head must not be replaced (matches skip_pattern)"
        assert isinstance(model.encoder.fc1, GhostLinear), \
            "encoder.fc1 must be replaced"

    # 4 : Multiple skip patterns -----------------------------------------

    def test_multiple_skip_patterns(self, device):
        model = _move(_NestedModel(), device, torch.float16)
        replace_linears(model, skip_patterns=["lm_head", "head"], min_features=1)
        assert not isinstance(model.lm_head,    GhostLinear)
        assert not isinstance(model.head.linear, GhostLinear)
        assert isinstance(model.encoder.fc1, GhostLinear)
        assert isinstance(model.encoder.fc2, GhostLinear)

    # 5 : Return value ---------------------------------------------------

    def test_return_value_is_dict(self, device, toy_model):
        result = replace_linears(toy_model, min_features=128)
        assert isinstance(result, dict)

    def test_return_dict_keys_are_strings(self, device, toy_model):
        result = replace_linears(toy_model, min_features=128)
        for k in result:
            assert isinstance(k, str), f"Key {k!r} is not a string"

    def test_return_dict_values_are_numeric(self, device, toy_model):
        result = replace_linears(toy_model, min_features=128)
        for k, v in result.items():
            assert isinstance(v, (int, float)), \
                f"Value for {k!r} is not numeric: {v!r}"

    # 6 : Recursive traversal --------------------------------------------

    def test_recursive_nested_model(self, device):
        model = _move(_NestedModel(), device, torch.float16)
        replace_linears(model, min_features=1, skip_patterns=None)
        assert isinstance(model.encoder.fc1, GhostLinear)
        assert isinstance(model.encoder.fc2, GhostLinear)
        assert isinstance(model.head.linear, GhostLinear)
        assert isinstance(model.lm_head,     GhostLinear)

    # 7 : Model still runs -----------------------------------------------

    def test_model_runs_after_replacement(self, device, toy_model):
        replace_linears(toy_model, min_features=128)
        x = torch.randn(4, 128, dtype=torch.float16, device=device)
        with torch.no_grad():
            y = toy_model(x)
        assert y.shape == (4, 10)

    def test_output_shape_unchanged(self, device):
        torch.manual_seed(0)
        model = _move(_NestedModel(), device, torch.float16)
        x = torch.randn(4, 128, dtype=torch.float16, device=device)
        with torch.no_grad():
            y_before = model(x)
        replace_linears(model, min_features=1)
        with torch.no_grad():
            y_after = model(x)
        assert y_before.shape == y_after.shape

    # 8 : Threshold forwarded --------------------------------------------

    def test_threshold_forwarded_to_ghost_linear(self, device, toy_model):
        threshold = 0.05
        replace_linears(toy_model, threshold=threshold, min_features=128)
        for name, mod in toy_model.named_modules():
            if isinstance(mod, GhostLinear):
                assert mod.threshold == pytest.approx(threshold), \
                    f"Threshold not forwarded to {name}"


# ---------------------------------------------------------------------------
# 9–12 : get_sparsity_report
# ---------------------------------------------------------------------------

class TestGetSparsityReport:

    def test_empty_for_plain_model(self, device):
        """Plain nn.Module with no GhostLinear → empty dict."""
        model = nn.Sequential(nn.ReLU(), nn.Dropout())
        report = get_sparsity_report(model)
        assert report == {}

    def test_keys_match_ghost_layer_names(self, device, toy_model):
        replace_linears(toy_model, min_features=128)
        x = torch.randn(4, 128, dtype=torch.float16, device=device)
        with torch.no_grad():
            toy_model(x)
        report = get_sparsity_report(toy_model)
        ghost_names = {
            name
            for name, mod in toy_model.named_modules()
            if isinstance(mod, GhostLinear)
        }
        assert set(report.keys()) == ghost_names

    def test_values_are_floats(self, device, toy_model):
        replace_linears(toy_model, min_features=128)
        x = torch.randn(4, 128, dtype=torch.float16, device=device)
        with torch.no_grad():
            toy_model(x)
        report = get_sparsity_report(toy_model)
        for k, v in report.items():
            assert isinstance(v, float), f"Sparsity for {k} must be float, got {type(v)}"

    def test_values_in_range(self, device, toy_model):
        replace_linears(toy_model, min_features=128)
        x = torch.randn(4, 128, dtype=torch.float16, device=device)
        with torch.no_grad():
            toy_model(x)
        report = get_sparsity_report(toy_model)
        for k, v in report.items():
            assert 0.0 <= v <= 1.0, f"Sparsity for {k} out of [0,1]: {v}"

    def test_zero_input_higher_avg_sparsity(self, device, toy_model):
        """
        Zero input → more dead neurons → higher average sparsity
        than dense (ones) input.
        """
        replace_linears(toy_model, min_features=128)

        x_dense = torch.ones(4, 128, dtype=torch.float16, device=device)
        with torch.no_grad():
            toy_model(x_dense)
        avg_dense = sum(get_sparsity_report(toy_model).values()) / \
                    max(len(get_sparsity_report(toy_model)), 1)

        x_zero = torch.zeros(4, 128, dtype=torch.float16, device=device)
        with torch.no_grad():
            toy_model(x_zero)
        avg_zero = sum(get_sparsity_report(toy_model).values()) / \
                   max(len(get_sparsity_report(toy_model)), 1)

        assert avg_zero >= avg_dense, (
            f"Zero input should give higher avg sparsity than ones input. "
            f"dense_avg={avg_dense:.3f}, zero_avg={avg_zero:.3f}"
        )
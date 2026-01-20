import pytest
import torch
import torch.nn as nn

# Import the function under test from wherever it lives
# Adjust this import path to match your project layout.
from guaranteed_fair_ensemble.subheads import concat_subheads, extract_subhead


class _DummyFairEnsemble:
    """
    Mimics the real FairEnsemble interface used by extract_subhead:
      - .classifiers is the FULL classifier module (not a ModuleList)
      - .num_classifiers is the number of ensemble members
    """

    def __init__(self, num_classifiers: int = 21, out_per: int = 1) -> None:
        self.num_classifiers = num_classifiers
        total_out = num_classifiers * out_per

        # MobileNetV3-like classifier:
        # extracted features are 576-d, classifier expands to 1024 then outputs total_out
        self.classifiers = nn.Sequential(
            nn.Linear(576, 1024),
            nn.Hardswish(),
            nn.Dropout(p=0.2),
            nn.Linear(1024, total_out),
        )


def test_extract_subhead_fails_on_mobilenet_feature_dim_mismatch() -> None:
    model = _DummyFairEnsemble(num_classifiers=21)

    # MobilenetV3 extracted feature size is typically 576
    features = torch.randn(8, 576)

    subhead, _ = extract_subhead(model, member_idx=0)

    # This should fail because subhead expects 1024 input features, not 576.
    output = subhead(features)
    assert output.shape[0] == features.shape[0]


def test_concat_subheads_linear_only():
    # Create three simple last-linears (in_features=128, out_features=1)
    subheads = [nn.Linear(128, 1, bias=True) for _ in range(3)]

    # Fill with deterministic weights for reproducibility (optional)
    for i, layer in enumerate(subheads):
        with torch.no_grad():
            layer.weight.fill_(0.1 * (i + 1))
            if layer.bias is not None:
                layer.bias.fill_(0.01 * (i + 1))

    combined = concat_subheads(subheads)

    # For linear-only case we should get an nn.Linear
    assert isinstance(combined, nn.Linear)
    # out_features should equal number of subheads and be accessible
    assert getattr(combined, "out_features", None) == len(subheads)

    # Forward pass shape check
    x = torch.randn(5, 128)
    out = combined(x)
    assert out.shape == (5, len(subheads))


def test_concat_subheads_sequential_prefix_and_out_features():
    # Build three subheads with a shared prefix:
    # prefix: Linear(64->32) + ReLU, final: Linear(32->1)
    def make_subhead():
        return nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    subheads = [make_subhead() for _ in range(4)]

    # Initialize final linears differently to ensure concatenation is happening
    for idx, sh in enumerate(subheads):
        final = [m for m in sh if isinstance(m, nn.Linear)][-1]
        with torch.no_grad():
            final.weight.fill_(0.05 * (idx + 1))
            if final.bias is not None:
                final.bias.fill_(0.005 * (idx + 1))

    combined = concat_subheads(subheads)

    # For sequential case we expect an nn.Sequential (prefix + combined_last)
    assert isinstance(combined, nn.Sequential)

    # The combined Sequential must expose out_features (compatibility shim)
    assert hasattr(combined, "out_features")
    assert combined.out_features == len(subheads)

    # Forward pass: input should be of shape (batch, 64)
    x = torch.randn(7, 64)
    out = combined(x)
    assert out.shape == (7, len(subheads))

    # And the prefix should be preserved (first layer is Linear with in_features=64)
    first_layer = next(iter(combined))
    assert isinstance(first_layer, nn.Linear)
    assert first_layer.in_features == 64

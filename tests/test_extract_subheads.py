import pytest
import torch
import torch.nn as nn

# Import the function under test from wherever it lives
# Adjust this import path to match your project layout.
from guaranteed_fair_ensemble.subheads import extract_subhead


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

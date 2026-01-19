import pytest
import torch
from torch import nn

from guaranteed_fair_ensemble.backbone import get_backbone


def _get_advertised_in_features_and_remove_classifier(model: nn.Module):
    """
    Emulate OneHeadFairretModel.__init__:
    - read the final Linear's in_features (advertised)
    - replace the final fc/classifier with nn.Identity() so backbone(x) returns features
    Returns: advertised_in_features (int)
    """
    # ResNet-style
    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        in_f = model.fc.in_features
        model.fc = nn.Identity()
        return in_f

    # classifier-style (MobileNet/EfficientNet)
    if hasattr(model, "classifier"):
        cls = model.classifier
        # If it's a Sequential, search in reverse for the last Linear and replace it
        if isinstance(cls, nn.Sequential):
            for i in range(len(cls) - 1, -1, -1):
                if isinstance(cls[i], nn.Linear):
                    in_f = cls[i].in_features
                    cls[i] = nn.Identity()
                    # assign back in case classifier is a view into model
                    model.classifier = cls
                    return in_f
        # If classifier itself is a Linear
        if isinstance(cls, nn.Linear):
            in_f = cls.in_features
            model.classifier = nn.Identity()
            return in_f

    raise RuntimeError("Could not locate final nn.Linear to inspect in backbone.")


def test_mobilenet_advertised_in_features_matches_actual_features_after_removal():
    # Create the mobilenet the same way the factory does
    model = get_backbone("mobilenetv3", num_heads=1, freeze=True)

    # Inspect advertised in_features and remove the final classification layer
    advertised_in = _get_advertised_in_features_and_remove_classifier(model)

    # Now forward to obtain actual features
    model.eval()
    with torch.no_grad():
        # standard input for torchvision models
        x = torch.randn(2, 3, 224, 224)
        feats = model(x)

    assert feats.ndim == 2, (
        f"Expected backbone output to be 2D (batch, features), got {feats.shape}"
    )
    actual_feat_dim = feats.shape[1]

    assert actual_feat_dim == advertised_in, (
        f"mobilenet advertised in_features={advertised_in} but produced actual features={actual_feat_dim}"
    )

import torch
import torch.nn as nn

"""Single-head classifier compatible with FairRet regularisation."""

__all__ = ["OneHeadFairretModel"]


class OneHeadFairretModel(nn.Module):
    """Backbone followed by a *single* classifier head.

    Parameters
    ----------
    backbone:
        Feature extractor (e.g. a torchvision ResNet).
    num_classes:
        Number of output classes (default 1 - binary classification).
    """

    def __init__(self, backbone: nn.Module, num_classes: int = 1) -> None:
        super().__init__()
        self.backbone = backbone
        self.num_classes = num_classes

        # Resolve feature dimension depending on backbone style
        if hasattr(backbone, "fc"):
            in_features = backbone.fc.in_features
            backbone.fc = nn.Identity()
        elif hasattr(backbone, "classifier"):
            cls_layer = backbone.classifier
            if isinstance(cls_layer, nn.Sequential):
                in_features = cls_layer[-1].in_features
            else:
                in_features = cls_layer.in_features
            backbone.classifier = nn.Identity()
        else:
            raise ValueError(
                "Unsupported backbone architecture - cannot locate final fully connected layer."
            )

        self.classifier = nn.Linear(in_features, num_classes)

    # -----------------------------------------------------------------
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.extract_features(x)
        return self.classifier(feats)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        return torch.sigmoid(logits)

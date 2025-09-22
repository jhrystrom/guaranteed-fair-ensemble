import torch
import torch.nn as nn


class DomainDiscriminativeModel(nn.Module):
    """A backbone + *K* domain-specific linear heads.

    Parameters
    ----------
    backbone:
        A CNN feature extractor (e.g. :class:`torchvision.models.ResNet`).
    num_domains:
        *K* - number of sensitive domains (default 3: Fitzpatrick 1-4, 5, 6).
    num_classes:
        Number of classes *per domain* (default 1 for binary tasks).
    """

    def __init__(
        self,
        backbone: nn.Module,
        num_domains: int = 3,
        num_classes: int = 1,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.num_domains = num_domains
        self.num_classes = num_classes

        # Determine backbone feature dimension
        if hasattr(backbone, "fc"):
            # ResNet-style network
            feat_dim = backbone.fc.in_features  # type: ignore[attr-defined]
            backbone.fc = nn.Identity()  # type: ignore[attr-defined]
        elif hasattr(backbone, "classifier"):
            cls_layer = backbone.classifier  # type: ignore[attr-defined]
            if isinstance(cls_layer, nn.Sequential):
                feat_dim = cls_layer[-1].in_features  # type: ignore[index]
            else:
                feat_dim = cls_layer.in_features  # type: ignore[attr-defined]
            backbone.classifier = nn.Identity()  # type: ignore[attr-defined]
        else:
            raise ValueError(
                "Unsupported backbone architecture - cannot locate final FC layer."
            )

        # Domain-specific classifier (single linear layer)
        if not hasattr(self, "classifier"):
            self.classifier = nn.Linear(feat_dim, num_domains * num_classes)
        else:
            self.classifier[-1] = nn.Linear(feat_dim, num_domains * num_classes)

    # forward / predict
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return backbone features."""

        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Raw logits for *all* domains (not passed through ``sigmoid``)."""

        feats = self.extract_features(x)
        return self.classifier(feats)

    # Inference helper - probability-sum across domains (no alternatives)
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return final class-probabilities via domain-sum rule."""

        logits = self.forward(x)
        return self.inference_sum_prob(logits, self.num_domains, self.num_classes)

    # Static so it can be reused in other contexts (e.g. Lightning module)
    @staticmethod
    def inference_sum_prob(
        logits: torch.Tensor,
        num_domains: int,
        num_classes: int = 1,
    ) -> torch.Tensor:
        """Sum per-domain probabilities.

        Shape assumptions
        -----------------
        ``logits`` shape :: (batch, num_domains * num_classes)
        """

        # Our shape is (B, K, C)
        logits_reshaped = logits.view(-1, num_domains, num_classes)
        probs = torch.sigmoid(logits_reshaped)
        # sum over K ->  (B, C)
        return probs.sum(dim=1)

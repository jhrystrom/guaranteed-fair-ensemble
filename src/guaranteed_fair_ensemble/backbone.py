from pathlib import Path

import torchvision.models as models
from lightning import LightningModule
from torch import nn

from guaranteed_fair_ensemble.data_models import ModelInfo

WEIGHTS_DICT = {
    "mobilenetv3": models.MobileNet_V3_Small_Weights.IMAGENET1K_V1,
    "efficientnet": models.EfficientNet_V2_M_Weights.IMAGENET1K_V1,
    "efficientnet_s": models.EfficientNet_V2_S_Weights.IMAGENET1K_V1,
    "resnet18": models.ResNet18_Weights.DEFAULT,
    "resnet50": models.ResNet50_Weights.DEFAULT,
}


def get_backbone(name: str, num_heads: int = 4, freeze: bool = True) -> nn.Module:
    """
    Get a pre-trained backbone model with custom output heads

    Args:
        name: Name of the backbone ('mobilenetv3', 'resnet18', 'resnet50')
        num_heads: Number of output heads
        freeze: Whether to freeze the backbone weights

    Returns:
        The backbone model with custom classification head
    """
    if name == "mobilenetv3":
        model = models.mobilenet_v3_small(weights=WEIGHTS_DICT["mobilenetv3"])
        if freeze:
            for param in model.parameters():
                param.requires_grad = False
        model.classifier[3] = nn.Linear(1024, num_heads)

    elif name == "efficientnet":
        model = models.efficientnet_v2_m(weights=WEIGHTS_DICT["efficientnet"])
        if freeze:
            for param in model.parameters():
                param.requires_grad = False
        model.classifier[-1] = nn.Linear(model.classifier[1].in_features, num_heads)

    elif name == "efficientnet_s":
        model = models.efficientnet_v2_s(weights=WEIGHTS_DICT["efficientnet_s"])
        if freeze:
            for param in model.parameters():
                param.requires_grad = False
        model.classifier[-1] = nn.Linear(model.classifier[1].in_features, num_heads)

    elif name == "resnet18":
        model = models.resnet18(weights=WEIGHTS_DICT["resnet18"])
        if freeze:
            for param in model.parameters():
                param.requires_grad = False
        model.fc = nn.Linear(model.fc.in_features, num_heads)

    elif name == "resnet50":
        model = models.resnet50(weights=WEIGHTS_DICT["resnet50"])
        if freeze:
            for param in model.parameters():
                param.requires_grad = False
        model.fc = nn.Linear(model.fc.in_features, num_heads)

    else:
        raise ValueError(f"Unknown backbone: {name}")

    return model


def get_model_for_method(method: str, backbone_name: str, **kwargs):
    """
    Factory function to create a model appropriate for the given training method

    Args:
        method: Training method name
        backbone_name: Name of the backbone to use
        **kwargs: Additional arguments for specific model types

    Returns:
        Model instance appropriate for the specified method
    """
    if method in {"standard", "ensemble"}:
        return get_backbone(backbone_name, **kwargs)

    if method == "domain_independent":
        from guaranteed_fair_ensemble.models.domain_independent import (
            DomainIndependentModel,
        )

        num_heads = kwargs.get("num_heads")
        if num_heads is None:
            raise ValueError(
                "num_heads must be specified for domain independent method"
            )
        # For domain independent method, we need to create a backbone first
        # then wrap it in the DomainIndependentModel
        backbone = get_backbone(backbone_name, num_heads=num_heads)

        # Fixed number of domains based on custom_one_hot grouping (3 groups: 1-4, 5, and 6)
        num_domains = num_heads - 1
        num_classes = 1  # Default to binary classification

        # Create and return the domain independent model
        return DomainIndependentModel(
            backbone=backbone, num_domains=num_domains, num_classes=num_classes
        )
    if method == "domain_discriminative":
        from guaranteed_fair_ensemble.models.domain_discriminative import (
            DomainDiscriminativeModel,
        )

        num_heads = kwargs.get("num_heads", 4)
        # For domain discriminative method, we need to create a backbone first
        # then wrap it in the DomainDiscriminativeModel
        backbone = get_backbone(backbone_name, num_heads=num_heads)

        num_domains = num_heads - 1
        num_classes = 1
        # Create and return the domain discriminative model
        return DomainDiscriminativeModel(
            backbone=backbone, num_domains=num_domains, num_classes=num_classes
        )
    if method in {"fairret", "erm", "rebalance"}:
        from guaranteed_fair_ensemble.models.fairret_one_head import OneHeadFairretModel

        backbone = get_backbone(backbone_name, num_heads=1)

        return OneHeadFairretModel(
            backbone=backbone,
            num_classes=1,  # Binary classification
        )

    raise ValueError(f"Unknown training method: {method}")


def initialize_model_checkpoint(
    model_info: ModelInfo, checkpoint_path: Path, num_heads: int = 1
) -> LightningModule:
    model = get_model_for_method(
        method=model_info.method, backbone_name=model_info.backbone, num_heads=num_heads
    )
    num_domains = num_heads - 1
    if model_info.method == "domain_independent":
        from guaranteed_fair_ensemble.models.domain_independent_lit import (
            DomainIndependentLitModule,
        )

        return DomainIndependentLitModule.load_from_checkpoint(
            checkpoint_path, model=model
        )
    if model_info.method == "domain_discriminative":
        from guaranteed_fair_ensemble.models.domain_discriminative_lit import (
            DomainDiscriminativeLitModule,
        )

        return DomainDiscriminativeLitModule.load_from_checkpoint(
            checkpoint_path, model=model, num_domains=num_domains
        )
    if model_info.method in {"fairret", "erm"}:
        from guaranteed_fair_ensemble.models.fairret_lit import OneHeadFairretLit

        return OneHeadFairretLit.load_from_checkpoint(
            checkpoint_path, model=model, num_domains=num_domains
        )
    if model_info.method == "ensemble":
        import torch

        from guaranteed_fair_ensemble.models.guaranteed_fair_ensemble_lit import (
            load_guaranteed_fair_ensemble_from_checkpoint,
        )

        model = get_model_for_method(
            method=model_info.method,
            backbone_name=model_info.backbone,
            num_heads=num_heads * 21,
        )

        return load_guaranteed_fair_ensemble_from_checkpoint(
            checkpoint_path,
            backbone=model,
            num_heads_per_member=num_heads,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

    raise ValueError(f"Unknown training method: {model_info.method}")


if __name__ == "__main__":
    mobilenet = get_backbone("mobilenetv3", num_heads=4, freeze=True)
    efficientnet = get_backbone("efficientnet", num_heads=4, freeze=True)

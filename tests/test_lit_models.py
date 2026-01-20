import pytest
import torch

from guaranteed_fair_ensemble.backbone import get_backbone
from guaranteed_fair_ensemble.models.domain_discriminative import (
    DomainDiscriminativeModel,
)
from guaranteed_fair_ensemble.models.domain_independent import DomainIndependentModel
from guaranteed_fair_ensemble.models.domain_independent_lit import (
    DomainIndependentLitModule,
)
from guaranteed_fair_ensemble.models.fairensemble_lit import FairEnsemble
from guaranteed_fair_ensemble.models.fairret_one_head import OneHeadFairretModel


# Each runner just needs to execute a path that would crash if wiring is wrong.
def run_fairensemble_forward(
    backbone_name: str, num_heads: int, x: torch.Tensor
) -> torch.Tensor:
    backbone = get_backbone(backbone_name, num_heads=num_heads, freeze=True)
    model = FairEnsemble(model=backbone)
    return model(x)  # or model.forward(x)


def run_domain_independent(
    backbone_name: str, num_heads: int, x: torch.Tensor
) -> torch.Tensor:
    backbone = get_backbone(backbone_name, num_heads=num_heads, freeze=True)
    model = DomainIndependentModel(
        backbone=backbone, num_classes=1, num_domains=num_heads
    )
    return model(x)  # or model.forward(x)


def run_domain_discriminative(
    backbone_name: str, num_heads: int, x: torch.Tensor
) -> torch.Tensor:
    backbone = get_backbone(backbone_name, num_heads=num_heads, freeze=True)
    model = DomainDiscriminativeModel(
        backbone=backbone, num_classes=1, num_domains=num_heads
    )
    return model(x)  # or model.forward(x)


def run_one_head_fairret(
    backbone_name: str, num_heads: int, x: torch.Tensor
) -> torch.Tensor:
    backbone = get_backbone(backbone_name, num_heads=num_heads, freeze=True)
    model = OneHeadFairretModel(backbone=backbone)
    return model(x)


def run_domain_independent_lit(
    backbone_name: str, num_heads: int, x: torch.Tensor
) -> torch.Tensor:
    backbone = get_backbone(backbone_name, num_heads=num_heads, freeze=True)
    base_model = DomainIndependentModel(
        backbone=backbone, num_classes=1, num_domains=num_heads
    )
    lit_model = DomainIndependentLitModule(model=base_model)
    return lit_model(x)


CASES = [
    pytest.param(
        "mobilenetv3_large",
        run_fairensemble_forward,
        8,
        id="mobilenetv3_large-fairensemble-forward",
    ),
    pytest.param(
        "efficientnet_s",
        run_fairensemble_forward,
        8,
        id="efficientnet_s-fairensemble-forward",
    ),
    pytest.param(
        "mobilenetv3",
        run_fairensemble_forward,
        8,
        id="mobilenetv3-fairensemble-forward",
    ),
    pytest.param(
        "mobilenetv3", run_domain_independent, 3, id="mobilenetv3-domain-independent"
    ),
    pytest.param(
        "mobilenetv3",
        run_domain_discriminative,
        3,
        id="mobilenetv3-domain-discriminative",
    ),
    pytest.param(
        "mobilenetv3", run_one_head_fairret, 1, id="mobilenetv3-onehead-fairret"
    ),
    pytest.param(
        "mobilenetv3",
        run_domain_independent_lit,
        3,
        id="mobilenetv3-domain-independent-lit",
    ),
]


@pytest.mark.parametrize(("backbone_name", "runner", "num_heads"), CASES)
def test_backbones_do_not_crash(backbone_name, runner, num_heads) -> None:
    x = torch.randn(2, 3, 224, 224)

    # The test passes if this line doesn't raise.
    y = runner(backbone_name, num_heads, x)

    # Optional sanity checks (still dim-agnostic):
    assert isinstance(y, torch.Tensor)
    assert y.shape[0] == x.shape[0]

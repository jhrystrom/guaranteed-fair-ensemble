import torch

from guaranteed_fair_ensemble.backbone import get_backbone
from guaranteed_fair_ensemble.models.fairensemble_lit import FairEnsemble
from guaranteed_fair_ensemble.models.fairret_one_head import OneHeadFairretModel


def test_efficientnet_s() -> None:
    backbone = get_backbone("efficientnet_s", num_heads=8, freeze=True)
    model = FairEnsemble(model=backbone)
    test_input = torch.randn(2, 3, 224, 224)
    output = model.forward(test_input)
    assert output.shape == (2, 8)


def test_mobilenetv3() -> None:
    backbone = get_backbone("mobilenetv3", num_heads=8, freeze=True)
    model = FairEnsemble(model=backbone)
    test_input = torch.randn(2, 3, 224, 224)
    features = model.extract_features(test_input)
    assert features.shape[0] == 2
    output = model.classifiers(features)
    assert output.shape == (2, 8)


def test_one_head_fairret() -> None:
    backbone = get_backbone("mobilenetv3", num_heads=1, freeze=True)
    model = OneHeadFairretModel(backbone=backbone)
    test_input = torch.randn(2, 3, 224, 224)
    output = model(test_input)
    assert output.shape == (2, 1)

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import torch
from numpy.typing import NDArray

FairnessMetric = Literal["min_recall", "equal_opportunity"]
SplitType = Literal["val", "test"]
FairMethod = Literal["multi", "joint", "oxonfair"]


@dataclass
class ConstraintResults:
    constraint_value: float
    min_recall: float
    equal_opportunity: float
    accuracy: float
    constraint_type: str = "min_recall"


@dataclass
class MinimalData:
    features: torch.Tensor
    groups: NDArray[np.int8]
    labels: NDArray[np.int8]


@dataclass
class DatasetInfo:
    name: str = "fitzpatrick17k"
    val_size: float = 0.2
    test_size: float = 0.2
    fairness_metric: FairnessMetric = "min_recall"

    @property
    def to_cmd(self) -> dict[str, str]:
        return {
            "dataset": self.name,
            "val_size": self.val_size,
            "test_size": self.test_size,
        }


@dataclass
class ModelInfo:
    method: str = "erm"
    backbone: str = "mobilenetv3"
    scaling_factor: float = 0.5
    rebalanced: bool = True
    ensemble_members: int = 21
    learning_rate: float = 1e-3
    max_epochs: int = 50

    @property
    def backbone_suffix(self):
        return f"_{self.backbone}" if self.backbone != "mobilenetv3" else ""


@dataclass
class WandbInfo:
    project: str = "fairness-image-classification"
    offline: bool = False


@dataclass
class TrainingInfo:
    dataset: DatasetInfo = field(default_factory=DatasetInfo)
    model: ModelInfo = field(default_factory=ModelInfo)
    wandb: WandbInfo = field(default_factory=WandbInfo)
    batch_size: int = 256
    iterations: int = 3
    seed: int = 4

    @property
    def dataset_name(self):
        return self.dataset.name

    @property
    def val_size(self):
        return self.dataset.val_size

    @property
    def test_size(self):
        return self.dataset.test_size

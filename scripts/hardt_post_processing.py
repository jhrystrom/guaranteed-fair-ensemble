import argparse
import functools
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from guaranteed_fair_ensemble.transforms import get_transforms
import lightning
import numpy as np
import oxonfair
import pandas as pd
import polars as pl
import torch
import torch.nn as nn
from imblearn.under_sampling import RandomUnderSampler
from loguru import logger
from numpy.typing import NDArray
from oxonfair import group_metrics as gm
from tqdm import tqdm

import guaranteed_fair_ensemble.backbone
import guaranteed_fair_ensemble.datasets
import guaranteed_fair_ensemble.names
import guaranteed_fair_ensemble.preprocess
from guaranteed_fair_ensemble.config import get_dataset_info
from guaranteed_fair_ensemble.constants import DATASET_HPARAMS, DEFAULT_SEED
from guaranteed_fair_ensemble.data.base import DatasetSpec
from guaranteed_fair_ensemble.data.registry import get_dataset
from guaranteed_fair_ensemble.data_models import (
    ConstraintResults,
    DatasetInfo,
    FairnessMetric,
    MinimalData,
    ModelInfo,
    TrainingInfo,
)
from guaranteed_fair_ensemble.directories import DATA_DIR, OUTPUT_DIR
from guaranteed_fair_ensemble.models.fairensemble_lit import (
    FairEnsemble,
    predict_from_features,
)
from guaranteed_fair_ensemble.predict import predict_across_thresholds, vote_majority


@dataclass
class Result:
    accuracy: float
    equal_opportunity: float
    min_recall: float

def get_sizes(dataset_name: str) -> tuple[float, float]:
    for params in DATASET_HPARAMS:
        if params.name != dataset_name:
            continue
        return params.val_size, params.test_size
    raise ValueError(f"Dataset '{dataset_name}' not found in DATASET_HPARAMS.")


def get_data_loaders(
    training_info: TrainingInfo,
    iteration: int = 0,
    batch_size: int = 256,
    img_dict: dict[str, Any] | None = None,
) -> tuple[
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader | None,
    torch.utils.data.DataLoader,
    dict[str, Any],
]:
    spec = get_dataset(name=training_info.dataset.name)
    val_size, test_size = get_sizes(training_info.dataset.name)
    full_data = spec.load_and_clean_data(data_dir=DATA_DIR)

    # Train/Test split for this iteration -------------------------------------------------------------
    train_df, test_df = guaranteed_fair_ensemble.datasets.split_data(
        df=full_data,
        cfg=spec.cfg,
        test_size=test_size,
        random_seed=DEFAULT_SEED + iteration,
    )

    train_split, val_split = guaranteed_fair_ensemble.datasets.split_data(
        df=train_df,
        cfg=spec.cfg,
        random_seed=DEFAULT_SEED + iteration,
        test_size=val_size,
    )

    if img_dict is None:
        img_dict = guaranteed_fair_ensemble.preprocess.get_image_dict(spec)
    else:
        logger.debug("Reusing existing image dictionary")

    train_tfms = get_transforms(
        is_train=True, backbone_name=training_info.model.backbone
    )
    eval_tfms = get_transforms(
        is_train=False, backbone_name=training_info.model.backbone
    )

    train_ds = guaranteed_fair_ensemble.datasets.create_dataset(
        df=train_split,
        img_dict=img_dict,
        cfg=spec.cfg,
        transform_ops=train_tfms,
        rebalance=True,
    )

    val_ds = guaranteed_fair_ensemble.datasets.create_dataset(
        df=val_split,
        img_dict=img_dict,
        cfg=spec.cfg,
        transform_ops=eval_tfms,
        rebalance=True,
    )

    train_loader, val_loader = guaranteed_fair_ensemble.datasets.create_data_loaders(
        train_dataset=train_ds,
        val_dataset=val_ds,
        batch_size=batch_size,
        rebalance=True,
    )

    # ── Test
    test_ds = guaranteed_fair_ensemble.datasets.create_dataset(
        df=test_df,
        img_dict=img_dict,
        cfg=spec.cfg,
        transform_ops=eval_tfms,
        rebalance=False,
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, img_dict
    




def main(
    dataset: str = "ham10000",
    min_iteration: int = 0,
    max_iterations: int = 3,
    overwrite: bool = False,
) -> None:
    # Get dataset and so on
    dataset_info = get_dataset_info(dataset)
    logger.debug(f"Dataset info: {dataset_info}")
    model_info = ModelInfo(method="erm_ensemble", backbone="efficientnet_s")
    training_info = TrainingInfo(dataset=dataset_info, model=model_info)
    spec = get_dataset(dataset)
    full_df = spec.load_and_clean_data(DATA_DIR)
    iterations = list(range(min_iteration, max_iterations))
    logger.info(f"Running iterations: {iterations}")
    img_dict = None
    for iteration in tqdm(iterations, desc="Train/test iterations"):
        iter_seed = DEFAULT_SEED + iteration
        train_loader, val_loader, test_loader, img_dict = get_data_loaders(
            training_info=training_info,
            iteration=iteration,
            batch_size=256,
            img_dict=img_dict,
        )
        # load model
        model_path = guaranteed_fair_ensemble.names.get_model_path(info=training_info, iteration=iteration)
        model = initialize_model(training_info=training_info, model_path=model_path)











def parse_result(raw_result: pd.DataFrame) -> Result:
    updated_scores = raw_result["upated"].to_dict()
    return Result(accuracy=updated_scores["Accuracy"],
                  equal_opportunity=updated_scores["Equal Opportunity"],
                  min_recall=updated_scores["Min Recall"])


def predict_average(model, images):
    scores = model(images)
    sigmoids = torch.sigmoid(scores)
    mean_probs = sigmoids.mean(dim=1).reshape(-1, 1)
    return mean_probs.cpu().detach().numpy()

def initialize_model(
    training_info: TrainingInfo,
    model_path: Path,
) -> lightning.LightningModule:
    spec = get_dataset(name=training_info.dataset.name)
    num_heads = (
        spec.cfg.num_protected_classes + 1 if spec.cfg.num_protected_classes > 2 else 2
    ) if training_info.model.method != "erm_ensemble" else training_info.model.ensemble_members
    logger.info(
        f"Initializing model for method '{training_info.model.method}' with {num_heads} heads"
    )
    model_info = training_info.model
    lit_model = guaranteed_fair_ensemble.backbone.initialize_model_checkpoint(
        model_info=model_info, checkpoint_path=model_path, num_heads=num_heads
    )
    return lit_model




dataset_params = next(hparam for hparam in DATASET_HPARAMS if hparam.name == "fitzpatrick17k")




training_info = TrainingInfo(model=baseline_info, dataset=dataset_params)




model(fake_images)

N_DATAPOINTS = 100
N_GROUPS = 3
# Binary dummy target
fake_images = torch.rand((N_DATAPOINTS, 3, 224, 224)).cuda()
target = torch.randint(0, 2, (N_DATAPOINTS,))
groups = torch.randint(0, N_GROUPS, (N_DATAPOINTS,))
val_predictions = predict_average(model, fake_images)


fpred = oxonfair.DeepFairPredictor(
    target, val_predictions, groups, use_actual_groups=True
)

fpred.fit(gm.accuracy, gm.equal_opportunity, value=0.005)

fpred.evaluate_fairness(metrics={"Accuracy": gm.accuracy, "Equal Opportunity": gm.equal_opportunity})





test_size = 50
test_images = torch.rand((test_size, 3, 224, 224)).cuda()
test_target = torch.randint(0, 2, (test_size,))
test_groups = torch.randint(0, N_GROUPS, (test_size,))


test_predictions= oxonfair.DeepDataDict(
 test_target,
 predict_average(model, test_images),
 test_groups,
)

raw = fpred.evaluate(data=test_predictions, metrics={"Accuracy": gm.accuracy, "Equal Opportunity": gm.equal_opportunity, "Min Recall": gm.recall.min})


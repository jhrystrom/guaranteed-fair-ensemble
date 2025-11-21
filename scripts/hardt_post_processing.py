from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import oxonfair
import pandas as pd
import polars as pl
import torch
from loguru import logger
from numpy.typing import NDArray
from oxonfair import group_metrics as gm

import guaranteed_fair_ensemble.backbone
import guaranteed_fair_ensemble.datasets
import guaranteed_fair_ensemble.names
import guaranteed_fair_ensemble.preprocess
from guaranteed_fair_ensemble.config import get_dataset_info
from guaranteed_fair_ensemble.constants import DATASET_HPARAMS, DEFAULT_SEED
from guaranteed_fair_ensemble.data.base import DatasetSpec
from guaranteed_fair_ensemble.data.registry import get_dataset
from guaranteed_fair_ensemble.data_models import (
    MinimalData,
    ModelInfo,
    TrainingInfo,
)
from guaranteed_fair_ensemble.directories import DATA_DIR, OUTPUT_DIR
from guaranteed_fair_ensemble.lit_model import LitMultiHead
from guaranteed_fair_ensemble.metrics import get_fairness_metric


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


def construct_minimal_data(
    spec: DatasetSpec, all_features: torch.Tensor, dataframe: pd.DataFrame
) -> MinimalData:
    _groups = dataframe[spec.cfg.protected_col].to_numpy()
    _labels = dataframe[spec.cfg.target_col].to_numpy().astype(np.int8)
    _features = all_features[dataframe.index.to_numpy()]
    return MinimalData(features=_features, groups=_groups, labels=_labels)


def analyse_dataset(
    dataset: str = "ham10000",
    metric: str = "equal_opportunity",
    min_iteration: int = 0,
    max_iterations: int = 3,
    overwrite: bool = False,
) -> pl.DataFrame:
    # Get dataset and so on

    dataset_info = get_dataset_info(dataset)
    logger.debug(f"Dataset info: {dataset_info}")
    model_info = ModelInfo(method="erm_ensemble", backbone="efficientnet_s")
    training_info = TrainingInfo(dataset=dataset_info, model=model_info)
    spec = get_dataset(dataset)
    full_df = spec.load_and_clean_data(DATA_DIR)
    all_features = guaranteed_fair_ensemble.preprocess.get_features(
        dataset_name=dataset, backbone_name=model_info.backbone
    )
    iterations = list(range(min_iteration, max_iterations))
    logger.info(f"Running iterations: {iterations}")
    result_complete = []
    for iteration in iterations:
        file_name = guaranteed_fair_ensemble.names.create_baseline_save_path(
            iteration=iteration, model_info=model_info, dataset_name=dataset
        )
        if file_name.exists() and not overwrite:
            logger.info(
                f"File {file_name} exists and overwrite is False. Skipping iteration {iteration}."
            )
            existing_df = pl.read_csv(file_name)
            result_complete.append(existing_df)
            continue
        iter_seed = DEFAULT_SEED + iteration
        model_path = guaranteed_fair_ensemble.names.get_model_path(
            info=training_info, iteration=iteration
        )
        model: LitMultiHead = (
            initialize_model(training_info=training_info, model_path=model_path)
            .eval()
            .to("cpu")
        )
        train_df, test_df = guaranteed_fair_ensemble.datasets.split_data(
            df=full_df,
            cfg=spec.cfg,
            random_seed=iter_seed,
            test_size=dataset_info.test_size,
        )
        _, val_split = guaranteed_fair_ensemble.datasets.split_data(
            df=train_df,
            cfg=spec.cfg,
            random_seed=iter_seed,
            test_size=dataset_info.val_size,
        )

        validation_data = construct_minimal_data(
            spec=spec, all_features=all_features, dataframe=val_split
        )

        validation_preds = predict_average(
            classifier=model.classifier, features=validation_data.features
        )

        fpred = oxonfair.DeepFairPredictor(
            target=validation_data.labels,
            score=validation_preds,
            groups=validation_data.groups,
            use_actual_groups=True,
        )
        fpred.fit(gm.accuracy, get_fairness_metric(metric), value=0.005)

        test_data = construct_minimal_data(
            spec=spec, all_features=all_features, dataframe=test_df
        )
        test_predictions = predict_average(
            classifier=model.classifier, features=test_data.features
        )

        test_data_dict = oxonfair.DeepDataDict(
            test_data.labels,
            test_predictions,
            test_data.groups,
        )

        probas = fpred.predict_proba(test_data_dict)

        raw_result = fpred.evaluate(
            data=test_data_dict,
            metrics={
                "Accuracy": gm.accuracy,
                "Equal Opportunity": gm.equal_opportunity,
                "Min Recall": gm.recall.min,
            },
        )

        result = parse_result(raw_result)
        logger.debug(f"Iteration {iteration} result: {result}")
        results_df = pl.DataFrame([result]).with_columns(
            pl.lit(iteration).alias("iteration"),
            pl.lit(dataset).alias("dataset"),
            pl.lit(metric).alias("metric"),
            pl.lit(f"{model_info.method}_hardt").alias("method"),
        )
        results_df.write_csv(file_name)
        result_complete.append(results_df)
    return pl.concat(result_complete)


def analyse_all_datasets(overwrite: bool = False) -> None:
    for fairness_metric in ["equal_opportunity"]:
        for dataset_param in DATASET_HPARAMS:
            dataset_name = dataset_param.name
            logger.info(f"Analysing dataset: {dataset_name}")
            dataset_result = analyse_dataset(
                dataset=dataset_name,
                min_iteration=0,
                max_iterations=3,
                overwrite=overwrite,
                metric=fairness_metric,
            )
            logger.debug(f"Dataset {dataset_name} result:\n{dataset_result}")


def parse_result(raw_result: pd.DataFrame) -> Result:
    updated_scores = raw_result["updated"].to_dict()
    return Result(
        accuracy=updated_scores["Accuracy"],
        equal_opportunity=updated_scores["Equal Opportunity"],
        min_recall=updated_scores["Minimal Group Recall"],
    )


def predict_average(classifier, features):
    scores = classifier(features)
    sigmoids = torch.sigmoid(scores)
    mean_probs = sigmoids.mean(dim=1).reshape(-1, 1)
    return mean_probs.cpu().detach().numpy()


def initialize_model(
    training_info: TrainingInfo,
    model_path: Path,
) -> LitMultiHead:
    spec = get_dataset(name=training_info.dataset.name)
    num_heads = (
        (
            spec.cfg.num_protected_classes + 1
            if spec.cfg.num_protected_classes > 2
            else 2
        )
        if training_info.model.method != "erm_ensemble"
        else training_info.model.ensemble_members
    )
    logger.info(
        f"Initializing model for metho '{training_info.model.method}' with {num_heads} heads"
    )
    model_info = training_info.model
    lit_model = guaranteed_fair_ensemble.backbone.initialize_model_checkpoint(
        model_info=model_info, checkpoint_path=model_path, num_heads=num_heads
    )
    return lit_model


if __name__ == "__main__":
    analyse_all_datasets(overwrite=False)

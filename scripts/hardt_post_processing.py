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
import guaranteed_fair_ensemble.metrics
import guaranteed_fair_ensemble.names
import guaranteed_fair_ensemble.preprocess
from guaranteed_fair_ensemble.config import get_dataset_info
from guaranteed_fair_ensemble.constants import DATASET_HPARAMS, DEFAULT_SEED
from guaranteed_fair_ensemble.data.base import DatasetSpec
from guaranteed_fair_ensemble.data.registry import get_dataset
from guaranteed_fair_ensemble.data_models import (
    FairnessMetric,
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


def prediction_over_threshold(
    val_predictions: pl.DataFrame, prediction_thresholds: list[float]
) -> pl.DataFrame:
    _observed_rates = []
    pred_label = val_predictions["prediction"].to_numpy()
    groups = val_predictions["protected_attr"].to_numpy()
    gt_label = val_predictions["true_label"].to_numpy()
    for pred_threshold in prediction_thresholds:
        metrics = guaranteed_fair_ensemble.metrics.evaluate_threshold(
            test_groups=groups,
            predictions=pred_label,
            test_labels=gt_label,
            threshold=pred_threshold,
        )
        _observed_rates.append(metrics)
    observed_rate_df = pl.DataFrame(_observed_rates)
    return observed_rate_df


def find_optimal_threshold(
    observed_rate_df: pl.DataFrame, fairness_metric: FairnessMetric
) -> pl.DataFrame:
    fairness_constraints = (
        np.linspace(0.5, 1.0, num=11)
        if fairness_metric == "min_recall"
        else np.linspace(0.01, 0.15, num=21)
    )
    _fairness_thresholds: list[pl.DataFrame] = []
    for fairness_constraint in fairness_constraints:
        fairness_filter = (
            pl.col(fairness_metric) >= fairness_constraint
            if fairness_metric == "min_recall"
            else pl.col(fairness_metric) <= fairness_constraint
        )
        _fairness_thresholds.append(
            observed_rate_df.filter(fairness_filter)
            .top_k(k=1, by="accuracy")
            .with_columns(pl.lit(fairness_constraint).alias("fairness_thresh"))
        )
    fairness_thresholds = pl.concat(_fairness_thresholds).rename(
        {"threshold": "prediction_threshold"}
    )
    return fairness_thresholds


def evaluate_row_metric(test_predictions: pl.DataFrame, row: dict[str, float]):
    row_metrics = guaranteed_fair_ensemble.metrics.evaluate_threshold(
        test_groups=test_predictions["protected_attr"].to_numpy(),
        predictions=test_predictions["prediction"].to_numpy(),
        test_labels=test_predictions["true_label"].to_numpy(),
        threshold=row["prediction_threshold"],
    )
    row_metrics["fairness_thresh"] = row["fairness_thresh"]
    return row_metrics


def compute_thresholded_predictions(
    test_predictions: pl.DataFrame, fairness_thresholds: pl.DataFrame
) -> pl.DataFrame:
    _threshold_predictions = []
    for row in fairness_thresholds.iter_rows(named=True):
        _threshold_predictions.append(
            evaluate_row_metric(pl.DataFrame(test_predictions), row)
        )
    return (
        pl.DataFrame(_threshold_predictions)
        .drop("threshold")
        .rename({"fairness_thresh": "threshold"})
    )


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
    new_model_info = ModelInfo(method="hpp_ensemble", backbone="efficientnet_s")
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
            iteration=iteration, model_info=new_model_info, dataset_name=dataset
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
        constraint_value = 0.005 if metric == "equal_opportunity" else 0.80
        logger.debug(
            f"Fitting fairness predictor with constraint {metric}={constraint_value}"
        )
        fpred.fit(gm.accuracy, get_fairness_metric(metric), value=constraint_value)

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

        if metric == "min_recall":
            val_probas = fpred.predict_proba(
                oxonfair.DeepDataDict(
                    validation_data.labels,
                    validation_preds,
                    validation_data.groups,
                ),
                force_normalization=True,
            )[:, 1]
            prediction_thresholds = np.linspace(0.0, 1.0, num=101)
            val_pred_df = pl.DataFrame(
                {
                    "prediction": val_probas,
                    "protected_attr": validation_data.groups,
                    "true_label": validation_data.labels,
                }
            )
            observed_rate_df = prediction_over_threshold(
                val_predictions=val_pred_df,
                prediction_thresholds=prediction_thresholds.tolist(),
            )
            fairness_thresholds = find_optimal_threshold(
                observed_rate_df=observed_rate_df,
                fairness_metric="min_recall",
            )
            test_probas = fpred.predict_proba(
                test_data_dict,
                force_normalization=True,
            )[:, 1]
            test_pred_df = pl.DataFrame(
                {
                    "prediction": test_probas,
                    "protected_attr": test_data.groups,
                    "true_label": test_data.labels,
                }
            )
            final_fairness_thresholds = compute_thresholded_predictions(
                test_predictions=test_pred_df, fairness_thresholds=fairness_thresholds
            )

            final_fairness_thresholds.write_csv(file_name)
            result_complete.append(final_fairness_thresholds)

        else:
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
                pl.lit(0.5).alias("threshold"),
            )
            results_df.write_csv(file_name)
            result_complete.append(results_df)
    return pl.concat(result_complete)


def analyse_all_datasets(overwrite: bool = False) -> None:
    for dataset_param in DATASET_HPARAMS:
        dataset_name = dataset_param.name
        fairness_metric = dataset_param.fairness_metric
        logger.info(f"Analysing dataset: {dataset_name} (metric: {fairness_metric})")
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
    analyse_all_datasets(overwrite=True)

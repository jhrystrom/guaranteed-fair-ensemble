import argparse
import functools
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import oxonfair
import polars as pl
import torch
import torch.nn as nn
from imblearn.under_sampling import RandomUnderSampler
from loguru import logger
from numpy.typing import NDArray
from oxonfair import group_metrics as gm
from tqdm import tqdm

import guaranteed_fair_ensemble.datasets
import guaranteed_fair_ensemble.names
import guaranteed_fair_ensemble.preprocess
from guaranteed_fair_ensemble.config import get_dataset_info
from guaranteed_fair_ensemble.constants import DEFAULT_SEED
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

MAX_MEMBERS = 21
ENSEMBLE_SIZES = list(range(3, MAX_MEMBERS + 1, 2))
ITERATIONS = 3
FAIRNESS_METRIC: FairnessMetric = "min_recall"


@dataclass
class ValidationPredictions:
    classifier_idx: int
    labels: NDArray[np.int8]
    groups: NDArray[np.int8]
    predictions: NDArray[np.float32]


@dataclass
class FairnessConstraint:
    name: FairnessMetric
    thresholds: NDArray[np.float32]
    constraint: gm.GroupMetric


def get_fairness_constraint(fairness_metric: FairnessMetric) -> FairnessConstraint:
    if fairness_metric == "min_recall":
        return FairnessConstraint(
            name="min_recall",
            thresholds=np.linspace(0.0, stop=1, num=21),
            constraint=gm.recall.min,
        )
    if fairness_metric == "equal_opportunity":
        return FairnessConstraint(
            name="equal_opportunity",
            thresholds=np.linspace(0.0, stop=0.2, num=11),
            constraint=gm.equal_opportunity,
        )
    raise ValueError(f"Unsupported fairness metric: {fairness_metric}")


def undersample(
    predictions, labels, groups
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    under_sampler = RandomUnderSampler(random_state=DEFAULT_SEED)
    under_validation_predictions, under_validation_groups = under_sampler.fit_resample(
        X=predictions, y=groups
    )
    under_validation_labels = labels[under_sampler.sample_indices_]
    return (
        under_validation_predictions,
        under_validation_groups,
        under_validation_labels,
    )


def get_multi_heads_with_constraints(
    model, val_results, fairness_constraint: FairnessConstraint
):
    multi_merged_heads: dict[float, list[nn.Module]] = defaultdict(list)
    for member_results in tqdm(val_results):
        fpred = oxonfair.DeepFairPredictor(
            target=member_results.labels.astype(np.bool_),
            score=member_results.predictions,
            groups=member_results.groups,
        )
        for constraint in tqdm(fairness_constraint.thresholds):
            fpred = fpred.fit(
                objective=gm.accuracy,
                constraint=fairness_constraint.constraint,
                value=constraint,
                recompute=False,
            )
            # extract the head
            subhead = extract_subhead(model, member_idx=member_results.classifier_idx)
            merged_head = fpred.merge_heads_pytorch(subhead.to("cpu"))
            multi_merged_heads[constraint].append(merged_head)
    combined_heads = {
        constraint: concat_linears(heads)
        for constraint, heads in multi_merged_heads.items()
    }
    return combined_heads


def fit_joint_ensemble_constraints(
    model: FairEnsemble,
    fairness_constraint: FairnessConstraint,
    validation_results: list[ValidationPredictions],
):
    ensemble_size = len(validation_results)
    all_validation_predictions = np.vstack(
        [vpred.predictions for vpred in validation_results]
    )
    all_validation_labels = np.hstack([vpred.labels for vpred in validation_results])
    all_validation_groups = np.hstack([vpred.groups for vpred in validation_results])
    # We need to downsample
    under_validation_predictions, under_validation_groups, under_validation_labels = (
        undersample(
            all_validation_predictions, all_validation_labels, all_validation_groups
        )
    )
    joint_fpred = oxonfair.DeepFairPredictor(
        target=under_validation_labels,
        score=under_validation_predictions,
        groups=under_validation_groups,
    )
    joint_heads: dict[float, list[nn.Module]] = defaultdict(list)
    for constraint in fairness_constraint.thresholds:
        joint_fpred.fit(
            objective=gm.accuracy,
            constraint=fairness_constraint.constraint,
            value=constraint,
            recompute=False,
        )
        # Apply to all ensemble members
        for classifier_idx in range(ensemble_size):
            subhead = extract_subhead(model, member_idx=classifier_idx)
            merged_head = joint_fpred.merge_heads_pytorch(subhead.to("cpu"))
            joint_heads[constraint].append(merged_head)
    joint_merged_heads = {
        constraint: concat_linears(heads) for constraint, heads in joint_heads.items()
    }
    return joint_merged_heads


def evaluate_thresholded_predictions(
    threshold_predictions: torch.Tensor,
    constraints: np.ndarray,
    dataset: MinimalData,
) -> pl.DataFrame:
    joint_majority_vote = vote_majority(threshold_predictions)
    joint_metrics = evaluate_metrics_fairness(
        constraints,
        joint_majority_vote,
        test_labels=dataset.labels,
        test_groups=dataset.groups,
    )
    return pl.DataFrame(joint_metrics)


def concat_linears(layers: list[nn.Linear]) -> nn.Linear:
    """
    Concatenate multiple nn.Linear layers (same in_features, out_features=1)
    into a single nn.Linear with out_features = len(layers).
    """
    if not layers:
        raise ValueError("layers must be non-empty")

    in_features = layers[0].in_features
    use_bias = layers[0].bias is not None

    # stack weights and biases
    weight = torch.cat([layer.weight.data for layer in layers], dim=0)
    bias = torch.cat([layer.bias.data for layer in layers], dim=0) if use_bias else None

    combined = nn.Linear(in_features, len(layers), bias=use_bias)
    combined.weight.data = weight
    if use_bias:
        combined.bias.data = bias  # type: ignore

    return combined


def calculate_metrics(
    test_labels: np.ndarray, predictions: np.ndarray, test_groups: np.ndarray
):
    MIN_RECALL = gm.recall.min
    EQUAL_OPPORTUNITY = gm.equal_opportunity
    ACCURACY = gm.accuracy.overall
    return {
        "min_recall": MIN_RECALL(test_labels, predictions, test_groups),
        "equal_opportunity": EQUAL_OPPORTUNITY(test_labels, predictions, test_groups),
        "accuracy": ACCURACY(test_labels, predictions, test_groups),
    }


def extract_subhead(fair_ensemble: FairEnsemble, member_idx: int):
    classifier = fair_ensemble.classifiers[-1]
    in_features = classifier.in_features
    out_features = int(classifier.out_features) // fair_ensemble.num_classifiers
    subhead = nn.Linear(in_features, out_features)
    # Copy the weights and bias from the original classifier's final layer
    # But only for this fold's heads
    start_idx = member_idx * out_features
    stop_idx = start_idx + out_features
    # Copy weights for just this fold's heads
    subhead.weight.data = classifier.weight.data[start_idx:stop_idx, :]
    if classifier.bias is not None:
        subhead.bias.data = classifier.bias.data[start_idx:stop_idx]
    return subhead


def evaluate_metrics_fairness(
    thresholds: NDArray[np.float32], threshold_predictions, test_labels, test_groups
):
    metrics_per_constraint: list[ConstraintResults] = []
    if thresholds.size != threshold_predictions.shape[1]:
        raise ValueError(
            "Mismatch between number of constraints and majority vote shape"
        )
    for constraint_idx, constraint in tqdm(enumerate(thresholds)):
        metrics = calculate_metrics(
            test_labels,
            threshold_predictions[:, constraint_idx].astype(np.int8),
            test_groups,
        )
        constraint_result = ConstraintResults(
            constraint_value=constraint,
            min_recall=metrics["min_recall"],
            equal_opportunity=metrics["equal_opportunity"],
            accuracy=metrics["accuracy"],
            constraint_type="min_recall",
        )
        metrics_per_constraint.append(constraint_result)
    return metrics_per_constraint


def get_validation_predictions(
    model: FairEnsemble, validation_data: MinimalData, train_mask_matrix: torch.Tensor
) -> list[ValidationPredictions]:
    # Create validation predictions
    val_mask = ~train_mask_matrix
    validation_labels = validation_data.labels
    validation_groups = validation_data.groups
    all_validation_features = validation_data.features
    val_results: list[ValidationPredictions] = []
    for classifier_idx in range(model.num_classifiers):
        classifier_mask = val_mask[:, classifier_idx]
        validation_features = all_validation_features[classifier_mask]
        start_idx, stop_idx = model._get_classifier_group(classifier_idx)
        with torch.no_grad():
            raw_val_predictions = model.classifiers(
                validation_features.to(model.device)
            )
            head_predictions: torch.Tensor = raw_val_predictions[:, start_idx:stop_idx]
        # Groups need to be single column
        classifier_mask_np = classifier_mask.numpy()
        val_results.append(
            ValidationPredictions(
                classifier_idx=classifier_idx,
                labels=validation_labels[classifier_mask_np],
                groups=validation_groups[classifier_mask_np],
                predictions=head_predictions.cpu().numpy(),
            )
        )
    return val_results


def fit_joint_model_with_constraints(
    iteration,
    dataset_info: DatasetInfo,
    device,
    model,
    validation_data: MinimalData,
    test_data: MinimalData,
    val_results,
    fairness_constraint: FairnessConstraint,
    overwrite: bool = False,
):
    get_file_path = functools.partial(
        guaranteed_fair_ensemble.names.get_fairensemble_file_path,
        dataset_name=dataset_info.name,
        fairness_metric=dataset_info.fairness_metric,
        method="joint",
        iteration=iteration,
    )
    validation_output = get_file_path(split="val")
    test_output_path = get_file_path(split="test")
    if validation_output.exists() and not overwrite:
        logger.info(
            f"Skipping iteration {iteration} as output exists and overwrite is False"
        )
        return
    joint_validation_results = []
    joint_test_results = []
    for ensemble_size in tqdm(ENSEMBLE_SIZES, desc="Joint ensemble sizes"):
        chosen_results = val_results[:ensemble_size]
        joint_heads = fit_joint_ensemble_constraints(
            model, fairness_constraint, chosen_results
        )
        joint_val_predictions = predict_across_thresholds(
            merged_heads=joint_heads,
            features=validation_data.features,
            fairness_thresholds=fairness_constraint.thresholds,
            device=device,
        )
        joint_test_predictions = predict_across_thresholds(
            merged_heads=joint_heads,
            features=test_data.features.to(device),
            fairness_thresholds=fairness_constraint.thresholds,
            device=device,
        )
        _joint_validation_results = evaluate_thresholded_predictions(
            threshold_predictions=joint_val_predictions,
            constraints=fairness_constraint.thresholds,
            dataset=validation_data,
        ).with_columns(
            pl.lit("joint").alias("method"),
            pl.lit(ensemble_size).alias("ensemble_size"),
        )
        joint_validation_results.append(_joint_validation_results)
        _joint_test_results = evaluate_thresholded_predictions(
            threshold_predictions=joint_test_predictions,
            constraints=fairness_constraint.thresholds,
            dataset=test_data,
        ).with_columns(
            pl.lit("joint").alias("method"),
            pl.lit(ensemble_size).alias("ensemble_size"),
        )
        joint_test_results.append(_joint_test_results)
    pl.concat(joint_validation_results).write_csv(validation_output)
    pl.concat(joint_test_results).write_csv(test_output_path)


def run_for_iteration(
    training_info: TrainingInfo,
    spec: DatasetSpec,
    full_df,
    all_features,
    iteration: int = 0,
    overwrite: bool = False,
) -> None:
    dataset_info = training_info.dataset
    seed = DEFAULT_SEED + iteration
    train_df, test_df = guaranteed_fair_ensemble.datasets.split_data(
        full_df, cfg=spec.cfg, test_size=dataset_info.test_size, random_seed=seed
    )
    train_mask_matrix = guaranteed_fair_ensemble.datasets.get_training_mask(
        train_df=train_df, training_info=training_info, seed=seed, spec=spec
    )

    device = "cpu"
    model = guaranteed_fair_ensemble.preprocess.get_models_multihead(
        training_info=training_info, spec=spec, iteration=iteration
    )
    model = model.eval().to(device)

    validation_data = guaranteed_fair_ensemble.datasets.construct_minimal_data(
        spec, all_features, train_df
    )
    test_data = guaranteed_fair_ensemble.datasets.construct_minimal_data(
        spec=spec, all_features=all_features, dataframe=test_df
    )

    logger.info("Getting validation predictions")
    val_results = get_validation_predictions(
        model=model,
        validation_data=validation_data,
        train_mask_matrix=train_mask_matrix,
    )

    predictions = predict_from_features(model=model, features=test_data.features)
    predicted_labels = ((predictions > 0.5).numpy()).mean(axis=1) > 0.5
    raw_ensemble_iteration_score = calculate_metrics(
        test_labels=test_data.labels,
        predictions=predicted_labels,
        test_groups=test_data.groups,
    )

    pl.DataFrame(raw_ensemble_iteration_score).with_columns(
        pl.lit(dataset_info.name).alias("dataset"),
        pl.lit(iteration).alias("iteration"),
        pl.lit("ensemble").alias("method"),
    ).write_csv(
        OUTPUT_DIR
        / f"{dataset_info.name}-iteration{iteration}-raw-ensemble-performance.csv"
    )

    fairness_constraint = get_fairness_constraint(dataset_info.fairness_metric)

    fit_predict_multi_threshold(
        iteration,
        dataset_info,
        device,
        model,
        validation_data,
        test_data,
        val_results,
        fairness_constraint=fairness_constraint,
        overwrite=overwrite,
        backbone=training_info.model.backbone,
    )

    # fit_joint_model_with_constraints(
    #     iteration,
    #     dataset_info,
    #     device,
    #     model,
    #     validation_data,
    #     test_data,
    #     val_results,
    #     fairness_constraint=fairness_constraint,
    #     overwrite=overwrite,
    # )


def fit_predict_multi_threshold(
    iteration,
    dataset_info: DatasetInfo,
    device,
    model,
    validation_data,
    test_data,
    val_results,
    fairness_constraint: FairnessConstraint,
    overwrite: bool = False,
    backbone: str = "efficientnet_s",
):
    get_file_path = functools.partial(
        guaranteed_fair_ensemble.names.get_fairensemble_file_path,
        dataset_name=dataset_info.name,
        fairness_metric=dataset_info.fairness_metric,
        iteration=iteration,
        backbone=backbone,
    )
    constraint_thresholds = fairness_constraint.thresholds
    test_file_path = get_file_path(split="test", method="multi")
    val_output_file_path = get_file_path(split="val", method="multi")
    oxonfair_val_path = get_file_path(split="val", method="oxonfair")
    oxonfair_test_path = get_file_path(split="test", method="oxonfair")
    if (
        test_file_path.exists()
        and val_output_file_path.exists()
        and oxonfair_val_path.exists()
        and oxonfair_test_path.exists()
        and not overwrite
    ):
        logger.info(
            f"Skipping iteration {iteration} as output exists and overwrite is False"
        )
        return
    combined_heads = get_multi_heads_with_constraints(
        model, val_results, fairness_constraint=fairness_constraint
    )
    # Save combined heads
    combined_heads_path = (
        DATA_DIR / f"{dataset_info.name}-iteration{iteration}-multi-fitted-heads.pth"
    )
    torch.save(combined_heads, combined_heads_path)
    multi_val_predictions = predict_across_thresholds(
        merged_heads=combined_heads,
        features=validation_data.features,
        fairness_thresholds=constraint_thresholds,
        device=device,
    )
    multi_test_predictions = predict_across_thresholds(
        merged_heads=combined_heads,
        features=test_data.features.to(device),
        fairness_thresholds=constraint_thresholds,
        device=device,
    )
    multi_metrics_min_recall_test = []
    multi_metrics_min_recall_val = []

    for ensemble_size in ENSEMBLE_SIZES:
        ensemble_val_predictions = multi_val_predictions[:, :ensemble_size, :]
        ensemble_test_predictions = multi_test_predictions[:, :ensemble_size, :]
        multi_metrics_min_recall_test.append(
            evaluate_thresholded_predictions(
                threshold_predictions=ensemble_test_predictions,
                constraints=constraint_thresholds,
                dataset=test_data,
            ).with_columns(pl.lit(ensemble_size).alias("ensemble_size"))
        )
        multi_metrics_min_recall_val.append(
            evaluate_thresholded_predictions(
                threshold_predictions=ensemble_val_predictions,
                constraints=constraint_thresholds,
                dataset=validation_data,
            ).with_columns(pl.lit(ensemble_size).alias("ensemble_size"))
        )
    multi_metrics_min_recall_test = pl.concat(multi_metrics_min_recall_test)
    multi_metrics_min_recall_val = pl.concat(multi_metrics_min_recall_val)
    multi_metrics_min_recall_test.write_csv(test_file_path)
    multi_metrics_min_recall_val.write_csv(val_output_file_path)

    # Now on to single member predictions
    oxonfair_val_metrics = evaluate_thresholded_predictions(
        threshold_predictions=multi_val_predictions[:, :1, :],
        constraints=constraint_thresholds,
        dataset=validation_data,
    ).with_columns(
        pl.lit("oxonfair").alias("method"),
    )
    oxonfair_test_metrics = evaluate_thresholded_predictions(
        threshold_predictions=multi_test_predictions[:, :1, :],
        constraints=constraint_thresholds,
        dataset=test_data,
    ).with_columns(
        pl.lit("oxonfair").alias("method"),
    )
    oxonfair_val_metrics.write_csv(oxonfair_val_path)
    oxonfair_test_metrics.write_csv(oxonfair_test_path)


def main(
    dataset: str = "ham10000",
    min_iteration: int = 0,
    max_iterations: int = 3,
    overwrite: bool = False,
    backbone: str = "efficientnet_s",
) -> None:
    # Get dataset and so on
    dataset_info = get_dataset_info(dataset)
    logger.debug(f"Dataset info: {dataset_info}")
    model_info = ModelInfo(method="ensemble", backbone=backbone)
    training_info = TrainingInfo(dataset=dataset_info, model=model_info)
    spec = get_dataset(dataset)
    full_df = spec.load_and_clean_data(DATA_DIR)
    all_features = guaranteed_fair_ensemble.preprocess.get_features(
        dataset_name=dataset, backbone_name=model_info.backbone
    )
    iterations = list(range(min_iteration, max_iterations))
    logger.info(f"Running iterations: {iterations}")
    for iteration in tqdm(iterations, desc="Train/test iterations"):
        run_for_iteration(
            training_info,
            spec,
            full_df,
            all_features,
            iteration=iteration,
            overwrite=overwrite,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run fit_fair_frontiers with specified iterations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="ham10000",
        help="Dataset name (must be registered in dataset registry)",
    )
    parser.add_argument(
        "--min-iteration",
        type=int,
        default=0,
        help="Minimum iteration index (inclusive)",
    )
    parser.add_argument(
        "--max-iteration",
        type=int,
        default=3,
        help="Maximum iteration index (exclusive)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Whether to overwrite existing output files",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="efficientnet_s",
        help="Backbone model name",
        choices=["efficientnet_s", "mobilenetv3"],
    )
    args = parser.parse_args()
    main(
        dataset=args.dataset,
        min_iteration=args.min_iteration,
        max_iterations=args.max_iteration,
        overwrite=args.overwrite,
        backbone=args.backbone,
    )

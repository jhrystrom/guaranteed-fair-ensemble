from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from fairlearn.metrics import MetricFrame
from oxonfair import group_metrics as gm
from sklearn.metrics import (
    accuracy_score,
    recall_score,
)

from guaranteed_fair_ensemble.torch_utils import reverse_one_hot

MIN_RECALL = gm.recall.min
EQUAL_OPPORTUNITY = gm.equal_opportunity
ACCURACY = gm.accuracy.overall


# Vectorized implementation with explicit pairwise comparisons
def disagreement_rate_vectorized(
    predictions: np.ndarray, threshold: float = 0.5
) -> float:
    """
    Calculate the disagreement rate among ensemble members using vectorized operations
    with explicit pairwise comparisons.
    Parameters:
    -----------
    predictions : np.ndarray
        An N x d array where N is the number of samples and d is the number of ensemble members.
        Each value represents a prediction score (typically between 0 and 1).

    threshold : float, default=0.5
        The threshold to convert prediction scores to binary decisions.

    Returns:
    --------
    float
        The average disagreement rate across all samples, ranging from 0 (perfect agreement)
        to 1 (complete disagreement).
    """
    # Convert predictions to binary decisions using the threshold
    binary_decisions = (predictions >= threshold).astype(int)

    # Get the number of samples and ensemble members
    _, n_members = binary_decisions.shape

    # Generate all pairs of indices (j, k) where j < k
    j_indices, k_indices = np.triu_indices(n_members, k=1)

    # For each sample and each pair, check if there's a disagreement
    # This creates a n_samples x n_pairs boolean array
    # We use broadcasting to compare all pairs for all samples at once
    disagreements = binary_decisions[:, j_indices] != binary_decisions[:, k_indices]

    # Calculate the disagreement rate for each sample
    # by taking the mean across all pairs
    sample_disagreement_rates = np.mean(disagreements, axis=1)

    # Return the average disagreement rate across all samples
    return np.mean(sample_disagreement_rates)


def calculate_threshold_metrics(
    test_labels: np.ndarray, predictions: np.ndarray, test_groups: np.ndarray
):
    metric_frame = MetricFrame(
        metrics={"recall": recall_score},
        y_true=test_labels,
        y_pred=predictions,
        sensitive_features=test_groups,
    )
    # Recall for individual groups
    group_recalls = metric_frame.by_group
    group_dict = {
        f"recall_group{index}": group_recalls.loc[index, "recall"]
        for index in range(len(group_recalls))
    }
    return {
        "min_recall": metric_frame.group_min()["recall"],
        "recall": metric_frame.overall["recall"],
        "equal_opportunity": metric_frame.difference()["recall"],
        "accuracy": accuracy_score(test_labels, predictions),
        **group_dict,
    }


def get_fairness_metric(name: str) -> Callable:
    """Map fairness metric name to oxonfair function"""
    metrics = {
        "demographic_parity": gm.demographic_parity,
        "equal_opportunity": gm.equal_opportunity,
        "equalized_odds": gm.equalized_odds,
        "predictive_parity": gm.predictive_parity,
        "treatment_equality": gm.treatment_equality,
        "min_recall": gm.recall.min,
    }
    metric = metrics.get(name)
    if metric is None:
        raise ValueError(f"Unknown fairness metric: {name}")
    return metric


def get_performance_metric(name: str) -> Callable:
    """Map performance metric name to oxonfair function"""
    metrics = {
        "accuracy": gm.accuracy,
        "balanced_accuracy": gm.balanced_accuracy,
        "precision": gm.precision,
        "recall": gm.recall,
    }
    metric = metrics.get(name)
    if metric is None:
        raise ValueError(f"Unknown performance metric: {name}")
    return metric


def collect_predictions(
    model, loader, device
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Collect model predictions on the given loader

    Args:
        model: The model to use for predictions
        loader: DataLoader with validation/test data
        device: Device to run the model on

    Returns:
        Tuple of (all_outputs, all_targets, all_groups)
    """
    model.eval().to(device)
    all_outputs = []
    all_targets = []
    all_groups = []

    with torch.no_grad():
        for x, y in loader:
            outputs = model(x.to(device)).cpu().numpy()
            all_outputs.append(outputs)
            all_targets.append(y[:, 0].numpy())  # First column is the target
            all_groups.append(
                reverse_one_hot(y[:, 1:]).numpy()
            )  # Second column is the protected attribute

    return (
        np.concatenate(all_outputs),
        np.concatenate(all_targets),
        np.concatenate(all_groups),
    )


def compute_fairness_metrics(
    outputs: np.ndarray,
    targets: np.ndarray,
    groups: np.ndarray,
    threshold: float = 0.5,
    metrics: dict[str, Callable] | None = None,
) -> dict[str, float]:
    """
    Compute various fairness metrics

    Args:
        outputs: Model outputs
        targets: Ground truth labels
        groups: Protected attribute values
        threshold: Decision threshold
        metrics: Dictionary of metric names to metric functions

    Returns:
        Dictionary of metric names to values
    """
    if metrics is None:
        metrics = {
            "demographic_parity": get_fairness_metric("demographic_parity"),
            "equal_opportunity": get_fairness_metric("equal_opportunity"),
            "equalized_odds": get_fairness_metric("equalized_odds"),
        }

    # Apply threshold to get binary predictions
    preds = (outputs > threshold).astype(int)

    # Calculate metrics
    results = {}
    for name, metric_fn in metrics.items():
        try:
            value = metric_fn(preds, targets, groups)
            results[name] = value
        except Exception as e:
            print(f"Error computing {name}: {e}")
            results[name] = float("nan")

    return results


def compute_performance_metrics(
    outputs: np.ndarray,
    targets: np.ndarray,
    threshold: float = 0.5,
    metrics: dict[str, Callable] | None = None,
) -> dict[str, float]:
    """
    Compute various performance metrics

    Args:
        outputs: Model outputs
        targets: Ground truth labels
        threshold: Decision threshold
        metrics: Dictionary of metric names to metric functions

    Returns:
        Dictionary of metric names to values
    """
    if metrics is None:
        metrics = {
            "accuracy": get_performance_metric("accuracy"),
            "balanced_accuracy": get_performance_metric("balanced_accuracy"),
            "precision": get_performance_metric("precision"),
            "recall": get_performance_metric("recall"),
        }

    # Apply threshold to get binary predictions
    preds = (outputs > threshold).astype(int)

    # Calculate metrics
    results = {}
    for name, metric_fn in metrics.items():
        try:
            value = metric_fn(preds, targets)
            results[name] = value
        except Exception as e:
            print(f"Error computing {name}: {e}")
            results[name] = float("nan")

    return results


def log_metrics(metrics: dict[str, Any], logger=None, prefix: str = ""):
    """
    Log metrics to console and logger (if provided)

    Args:
        metrics: Dictionary of metrics to log
        logger: Logger to use (e.g., WandB)
        prefix: Prefix to add to metric names
    """
    # Log to console
    for name, value in metrics.items():
        full_name = f"{prefix}_{name}" if prefix else name
        print(f"{full_name}: {value}")

    # Log to logger if provided
    if logger is not None:
        logger_metrics = {
            f"{prefix}_{k}" if prefix else k: v for k, v in metrics.items()
        }
        logger.log(logger_metrics)


def calculate_metrics(
    test_labels: np.ndarray, predictions: np.ndarray, test_groups: np.ndarray
):
    return {
        "min_recall": MIN_RECALL(test_labels, predictions, test_groups),
        "equal_opportunity": EQUAL_OPPORTUNITY(test_labels, predictions, test_groups),
        "accuracy": ACCURACY(test_labels, predictions, test_groups),
    }


def evaluate_threshold(
    test_labels: np.ndarray,
    predictions: np.ndarray,
    test_groups: np.ndarray,
    threshold: float = 0.5,
):
    # Convert probabilities to binary predictions
    if predictions.ndim > 1 and predictions.shape[1] > 1:
        # We are an ensemble
        binary_predictions = ((predictions >= threshold).mean(axis=1) > 0.5).astype(int)
    else:
        binary_predictions = (predictions >= threshold).astype(int)
    # Calculate metrics
    metrics = calculate_metrics(test_labels, binary_predictions, test_groups)
    metrics["threshold"] = threshold
    return metrics

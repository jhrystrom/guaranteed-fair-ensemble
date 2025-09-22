from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tqdm import tqdm

from guaranteed_fair_ensemble.torch_utils import reverse_one_hot


def vote_majority(predictions: torch.Tensor) -> np.ndarray:
    if predictions.dim() != 3:
        raise ValueError(
            "Predictions tensor must be 3-dimensional (test_size, ensemble_size, num_constraints)"
        )
    majority_votes = (predictions > 0.0).float().mean(axis=1) > 0.5  # type: ignore
    return majority_votes.numpy()  # Dim: (test_size, num_constraints)


def predict_across_thresholds(
    merged_heads: dict[float, torch.nn.Module],
    features: torch.Tensor,
    fairness_thresholds: list[float],
    device: str = "cpu",
):
    # Evaluate on full set for each constraint
    num_ensemble_members = merged_heads[fairness_thresholds[0]].out_features
    predictions = torch.zeros(
        (features.shape[0], num_ensemble_members, len(fairness_thresholds))
    )  # type: ignore
    for constraint_idx, constraint in enumerate(fairness_thresholds):
        classifiers = merged_heads[constraint].to(device)
        with torch.no_grad():
            val_predictions = classifiers(features)
        predictions[:, :, constraint_idx] = val_predictions
    return predictions


def predict_results(
    model, loader, device: str = "cuda", custom_predict_fn: bool = False
) -> pd.DataFrame:
    if not custom_predict_fn:
        model.eval()

    predictions = []
    targets = []
    protected_attrs = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Generating predictions"):
            x, y_raw, _ = batch
            target = y_raw[:, 0]
            sensitive_attr = reverse_one_hot(y_raw[:, 1:])
            x = x.to(device)

            # Use the custom predict function if provided
            if custom_predict_fn:
                # The model is actually a custom predict function
                probs = model(x)
            # Otherwise, use model's predict method if available
            elif hasattr(model, "predict"):
                # The model has a predict method, use it
                probs = model.to(device).predict(x)
            else:
                # Standard model with sigmoid activation
                outputs = model.to(device)(x)
                probs = torch.sigmoid(outputs)

            # Store predictions and ground truth
            predictions.append(probs.cpu())
            targets.append(target.cpu())
            protected_attrs.append(sensitive_attr.cpu())

    # Concatenate predictions and other data
    predictions = torch.cat(predictions).numpy()
    targets = torch.cat(targets).numpy()
    protected_attrs = torch.cat(protected_attrs).numpy()

    # Create predictions DataFrame
    inputs = {
        "prediction": predictions[:, 0].flatten(),
        "true_label": targets.flatten(),
        "protected_attr": protected_attrs.flatten(),
    }
    if predictions.shape[1] > 1:
        print(f"Model has {predictions.shape[1]} output heads. Saving all predictions.")
        for i in range(1, predictions.shape[1]):
            inputs[f"protected_pred_{i}"] = predictions[:, i].flatten()

    return pd.DataFrame(inputs)


def predict_and_save(
    model,
    test_loader,
    output_dir,
    file_prefix="model",
    device="cuda",
    save: bool = True,
    custom_predict_fn=False,  # New parameter to indicate a custom predict function
):
    """
    Generate predictions on test data and save to CSV

    Args:
        model: Trained model or custom predict function
        test_loader: Test data loader
        output_dir: Directory to save predictions
        file_prefix: Prefix for output file names
        device: Device to run inference on
        save: Whether to save predictions to CSV
        custom_predict_fn: Whether model is a custom predict function

    Returns:
        DataFrame with predictions
    """
    # Ensure output directory exists
    if save:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    pred_df = predict_results(
        model, loader=test_loader, device=device, custom_predict_fn=custom_predict_fn
    )

    # Save predictions to CSV
    output_path = get_pred_path(output_dir, file_prefix)
    if save:
        pred_df.to_csv(output_path, index=False)
        print(f"Predictions saved to {output_path}")

    return pred_df


def get_pred_path(output_dir: Path, file_prefix: str) -> Path:
    output_path = Path(output_dir) / f"{file_prefix}_predictions.csv"
    return output_path


def compute_and_save_metrics(
    predictions, targets, groups, output_dir, file_prefix="model"
):
    """
    Compute and save evaluation metrics

    Args:
        predictions: Model predictions
        targets: Ground truth labels
        groups: Protected attribute groups
        output_dir: Directory to save metrics
        file_prefix: Prefix for output file names

    Returns:
        Dictionary of metrics
    """
    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Binary predictions (threshold at 0.5)
    binary_preds = (predictions >= 0.5).astype(int)

    # Overall metrics
    metrics = {
        "accuracy": accuracy_score(targets, binary_preds),
        "precision": precision_score(targets, binary_preds, zero_division=0),
        "recall": recall_score(targets, binary_preds, zero_division=0),
        "f1": f1_score(targets, binary_preds, zero_division=0),
        "auc": roc_auc_score(targets, predictions)
        if len(np.unique(targets)) > 1
        else 0,
    }

    # Group-wise metrics
    unique_groups = np.unique(groups)

    # Calculate metrics per group
    for group in unique_groups:
        group_idx = groups == group
        group_targets = targets[group_idx]
        group_preds = predictions[group_idx]
        group_binary_preds = binary_preds[group_idx]

        # Skip groups with too few samples or only one class
        if len(group_targets) < 2 or len(np.unique(group_targets)) < 2:
            continue

        metrics[f"group_{group}_accuracy"] = accuracy_score(
            group_targets, group_binary_preds
        )
        metrics[f"group_{group}_precision"] = precision_score(
            group_targets, group_binary_preds, zero_division=0
        )
        metrics[f"group_{group}_recall"] = recall_score(
            group_targets, group_binary_preds, zero_division=0
        )
        metrics[f"group_{group}_f1"] = f1_score(
            group_targets, group_binary_preds, zero_division=0
        )
        metrics[f"group_{group}_auc"] = roc_auc_score(group_targets, group_preds)

    # Calculate group fairness metrics
    if len(unique_groups) > 1:
        # Calculate false positive rates (FPR) per group
        fprs = {}
        fnrs = {}

        for group in unique_groups:
            group_idx = groups == group
            group_targets = targets[group_idx]
            group_binary_preds = binary_preds[group_idx]

            # Calculate FPR: FP / (FP + TN)
            neg_idx = group_targets == 0
            if np.sum(neg_idx) > 0:
                fpr = np.sum((group_binary_preds == 1) & (group_targets == 0)) / np.sum(
                    neg_idx
                )
                fprs[f"group_{group}"] = fpr
                metrics[f"group_{group}_fpr"] = fpr

            # Calculate FNR: FN / (FN + TP)
            pos_idx = group_targets == 1
            if np.sum(pos_idx) > 0:
                fnr = np.sum((group_binary_preds == 0) & (group_targets == 1)) / np.sum(
                    pos_idx
                )
                fnrs[f"group_{group}"] = fnr
                metrics[f"group_{group}_fnr"] = fnr

        # Calculate disparities between groups
        if len(fprs) > 1:
            fpr_values = list(fprs.values())
            metrics["fpr_disparity"] = max(fpr_values) - min(fpr_values)

        if len(fnrs) > 1:
            fnr_values = list(fnrs.values())
            metrics["fnr_disparity"] = max(fnr_values) - min(fnr_values)

    # Save metrics to CSV
    metrics_df = pd.DataFrame(metrics.items(), columns=["metric", "value"])
    output_path = Path(output_dir) / f"{file_prefix}_metrics.csv"
    metrics_df.to_csv(output_path, index=False)
    print(f"Metrics saved to {output_path}")

    return metrics

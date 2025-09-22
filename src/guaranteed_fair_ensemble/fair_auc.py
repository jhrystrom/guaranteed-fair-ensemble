import numpy as np
import polars as pl

from guaranteed_fair_ensemble.data_models import FairnessMetric


def calculate(
    method_df: pl.DataFrame,
    baseline_scores: pl.DataFrame | None = None,
    metric: FairnessMetric = "min_recall",
):
    """
    Implements the mean improvement score as described in the paper.
    """

    method_maxs = _calculate_max_expressions(method_df, metric)

    # Compute all max values for the baseline in one go
    baseline_maxs = (
        _calculate_max_expressions(baseline_scores, metric)
        if baseline_scores is not None
        else [0.0] * len(method_maxs)
    )

    # Calculate improvements
    improvements = [
        method_max - baseline_max
        for method_max, baseline_max in zip(method_maxs, baseline_maxs)
    ]

    return np.mean(improvements)


def _calculate_max_expressions(method_df: pl.DataFrame, metric: FairnessMetric):
    expr_list = _generate_threshold_expressions(metric)

    # Compute all max values for method_df in one go
    method_maxs = method_df.select([expr.max() for expr in expr_list]).row(0)
    return method_maxs


def _generate_threshold_expressions(metric: FairnessMetric) -> list[pl.Expr]:
    fairness_thresholds = get_fairness_thresholds(metric)
    # Create a column for each threshold
    expr_list = []
    for threshold in fairness_thresholds:
        if metric == "min_recall":
            # For each threshold, create an expression that computes the max accuracy
            # where metric >= threshold
            expr = (
                # We select where the target metric is less than the threshold both on val AND test
                pl.when(
                    (pl.col("threshold") >= threshold) & (pl.col(metric) >= threshold)
                )
                .then(pl.col("accuracy"))
                .otherwise(0.0)
            )
        else:
            expr = (
                # We select where the target metric is less than the threshold both on val AND test
                pl.when(
                    (pl.col("threshold") <= threshold) & (pl.col(metric) <= threshold)
                )
                .then(pl.col("accuracy"))
                .otherwise(0.0)
            )
        expr_list.append(expr.alias(f"t_{threshold}"))
    return expr_list


def get_fairness_thresholds(metric: FairnessMetric) -> np.ndarray:
    return (
        np.linspace(0.5, 0.9, 100)
        if metric == "min_recall"
        else np.linspace(0.01, 0.2, 50)
    )

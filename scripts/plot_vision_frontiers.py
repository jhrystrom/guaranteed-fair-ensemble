""" """

import argparse
from functools import cache
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from dotenv import load_dotenv
from line_profiler import profile
from loguru import logger
from tqdm import tqdm

import guaranteed_fair_ensemble.colors
import guaranteed_fair_ensemble.fair_auc
import guaranteed_fair_ensemble.names
from guaranteed_fair_ensemble.config import get_dataset_info
from guaranteed_fair_ensemble.constants import (
    ALL_BASELINES,
    ALL_METHODS,
    DATASET_HPARAMS,
    FAIRRET_SCALES,
    SIMPLE_BASELINES,
)
from guaranteed_fair_ensemble.data.registry import get_dataset
from guaranteed_fair_ensemble.data_models import (
    DatasetInfo,
    FairnessMetric,
    ModelInfo,
    SplitType,
)
from guaranteed_fair_ensemble.directories import DATA_DIR, OUTPUT_DIR, PLOT_DIR
from guaranteed_fair_ensemble.metrics import evaluate_threshold

load_dotenv(override=True)


PLOT_COLUMNS = ["method", "accuracy", "min_recall", "equal_opportunity", "threshold"]
_val_rank = pl.lit(0).alias("val_rank").cast(pl.UInt32)


NICE_DATASET_NAMES = {
    "fitzpatrick17k": "Fitzpatrick17k",
    "ham10000": "HAM10000",
    "fairvlmed": "FairVLMed",
}


def get_middle_predictions(
    dataset: str, backbone: str, num_iteration: int = 3
) -> pl.DataFrame:
    all_infos = guaranteed_fair_ensemble.names.get_all_model_infos(backbone=backbone)
    middle_predictions = []
    for iteration in range(num_iteration):
        for model_info in all_infos:
            baseline_path = guaranteed_fair_ensemble.names.create_baseline_save_path(
                iteration=iteration,
                model_info=model_info,
                dataset_name=dataset,
                is_val=False,
            )
            baseline_observed_path = (
                (OUTPUT_DIR / "temp" / f"{baseline_path.stem}_observed_rates.csv")
                if model_info.method != "hpp_ensemble"
                else baseline_path
            )
            if not baseline_observed_path.exists():
                raise FileNotFoundError(f"Missing {baseline_observed_path}")
            method_name = (
                model_info.method
                if model_info.method != "fairret"
                else f"fairret-{guaranteed_fair_ensemble.names.float_to_str(model_info.scaling_factor)}"
            )
            middle_predictions.append(
                pl.read_csv(baseline_observed_path).with_columns(
                    pl.lit(method_name).alias("method"),
                    pl.lit(iteration).alias("iteration"),
                    pl.lit("Baseline").alias("method_type"),
                )
            )
    return pl.concat(middle_predictions).filter(pl.col("threshold") == 0.5)


def filter_ensemble_methods(threshold_df: pl.DataFrame) -> pl.DataFrame:
    """Filter to only keep the best (biggest) ensemble method for each type"""
    fair_filter = pl.col("method_type").is_in(["Multi"])
    top_fair = (
        threshold_df.filter(fair_filter)
        .with_columns(
            pl.col("method")
            .str.extract(r"^(\d+)")
            .alias("ensemble_size")
            .cast(pl.Int16)
        )
        .select(
            pl.all()
            .top_k_by("ensemble_size", k=1, reverse=False)
            .over("method_type", mapping_strategy="explode")
        )["method"]
    )
    logger.debug(f"Top ensemble methods: {top_fair}")
    return threshold_df.filter(~fair_filter | (pl.col("method").is_in(top_fair)))


def filter_fairret(threshold_df: pl.DataFrame) -> tuple[pl.DataFrame, pl.Expr]:
    fairret_filter = pl.col("method").str.starts_with("fairret")
    top_fairret = (
        threshold_df.filter(fairret_filter)
        .group_by("method")
        .agg(pl.col("val_rank").mean())
        .sort("val_rank")["method"]
        .first()
    )
    logger.debug(f"Top fairret method: {top_fairret}")
    fairret_removal_filter = ~fairret_filter | (pl.col("method") == top_fairret)
    return threshold_df.filter(fairret_removal_filter), fairret_removal_filter


@profile
def main(
    methods: list[str],
    datasets: list[str],
    backbone: str = "efficientnet_s",
    iterations: int = 3,
) -> None:
    sns.set_theme(style="whitegrid", font_scale=1.25)
    if "all" in methods:
        methods = ALL_METHODS
    if "all" in datasets:
        datasets = [data_info.name for data_info in DATASET_HPARAMS]
    # Load the predictions
    all_thresholds = []
    for dataset in DATASET_HPARAMS:
        if dataset.name not in datasets:
            logger.info(f"Skipping dataset: {dataset.name}")
            continue
        for iteration in range(iterations):
            temp_threshold_df = pl.concat(
                [
                    get_thresholds(
                        method_name=method,
                        dataset=dataset.name,
                        backbone=backbone,
                        iteration=iteration,
                    )
                    .select(*PLOT_COLUMNS, "val_rank")
                    .with_columns(
                        pl.lit(iteration).alias("iteration"),
                    )
                    for method in tqdm(methods)
                ],
                how="vertical_relaxed",
            ).with_columns(
                pl.lit(dataset.fairness_metric).alias("fairness_metric"),
                pl.lit(dataset.name).alias("dataset"),
            )
            all_thresholds.append(temp_threshold_df)

    method_type_identifier = (
        pl.when(pl.col("method").str.contains(r"\dmulti"))
        .then(pl.lit("Multi"))
        .otherwise(pl.lit("Baseline"))
        .alias("method_type")
    )
    new_names = {
        "domain_discriminative": "DomainDisc",
        "domain_independent": "DomainInd",
        "hpp_ensemble": "Ensemble (HPP)",
    }

    minimum_rate_filter = (
        pl.when(pl.col("fairness_metric") == "min_recall")
        .then(pl.col("threshold") > 0.5)
        .otherwise(pl.lit(True))
    )

    threshold_df: pl.DataFrame = (
        pl.concat(all_thresholds)
        .filter(minimum_rate_filter)
        .with_columns(method_type_identifier)
        .with_columns(pl.col("method").replace(new_names))
        .with_columns(pl.col("method").str.replace("_ensemble", ""))
    )

    threshold_df = filter_ensemble_methods(threshold_df)
    threshold_df, fairret_removal = filter_fairret(threshold_df)
    assert "Ensemble (HPP)" in threshold_df["method"].to_list(), (
        "hpp_ensemble missing from thresholds"
    )

    cols = [
        "method",
        "method_type",
        "accuracy",
        "iteration",
        "equal_opportunity",
        "min_recall",
        "threshold",
    ]

    _plot_threshold_dfs = []
    for (dataset, fairness_metric), subdf in threshold_df.group_by(
        "dataset", "fairness_metric"
    ):
        if fairness_metric == "min_recall":
            plot_threshold_df = (
                subdf.group_by("min_recall", "method", "threshold")
                .agg(pl.col("accuracy").first())
                .sort("method")
            )
        else:
            middle_predictions = (
                get_middle_predictions(
                    dataset=dataset, backbone=backbone, num_iteration=iterations
                )
                .select(cols)
                .filter(fairret_removal)
            )
            plot_threshold_df = (
                pl.concat(
                    [
                        subdf.select(cols)
                        .filter(
                            (pl.col("method_type") != "Baseline")
                            | (pl.col("method") == "oxonfair")
                            | (pl.col("method") == "Ensemble (HPP)")
                        )
                        .filter(pl.col("threshold") > 0.0),
                        middle_predictions,
                    ]
                )
                .group_by(fairness_metric, "method", "threshold")
                .agg(pl.col("accuracy").first())
                .sort("method")
            )
        _plot_threshold_dfs.append(
            plot_threshold_df.with_columns(
                pl.lit(fairness_metric).alias("fairness_metric"),
                pl.lit(dataset).alias("dataset"),
            ).rename({fairness_metric: "fairness_value"})
        )

    plot_threshold_df = pl.concat(_plot_threshold_dfs).with_columns(
        pl.col("dataset").replace(NICE_DATASET_NAMES)
    )
    assert "Ensemble (HPP)" in plot_threshold_df["method"].to_list(), (
        "hpp_ensemble missing from thresholds"
    )

    logger.debug(f"Plotting methods: {plot_threshold_df[['method']].unique()}")
    reversed_domain_disc = {v: k for k, v in new_names.items()}

    averaged_threshold_df = (
        plot_threshold_df.group_by("method", "threshold", "dataset", "fairness_metric")
        .mean()
        .with_columns(guaranteed_fair_ensemble.names.normalise_method_names)
        .with_columns(pl.col("method").replace(reversed_domain_disc))
    )
    logger.debug(f"{averaged_threshold_df['method'].unique().to_list()}")

    # --- Wide layout + simple font control via Seaborn ---------------------------
    N_COLS = 3  # always 3 datasets
    TARGET_WIDTH_IN = 21.0  # make it wide for the paper
    PANEL_HEIGHT_IN = 5.0  # facet height
    FONT_SCALE = 2.0  # <— tweak this single number

    # Compute relplot geometry from width target
    aspect = TARGET_WIDTH_IN / (N_COLS * PANEL_HEIGHT_IN)
    height = PANEL_HEIGHT_IN

    # One place to control fonts
    sns.set_theme(style="whitegrid", font_scale=FONT_SCALE)
    plt.rcParams["legend.markerscale"] = 1.25

    g = sns.relplot(
        averaged_threshold_df.sort("fairness_metric", "dataset", descending=True),
        x="fairness_value",
        col="dataset",
        y="accuracy",
        hue="method",
        style="method",
        s=225,
        height=height,
        aspect=aspect,
        palette=guaranteed_fair_ensemble.colors.get_method_colours(),
        facet_kws={"sharey": False, "sharex": False},
    )

    # X labels per dataset's fairness metric
    dataset_to_metric = averaged_threshold_df.group_by("dataset").agg(
        pl.col("fairness_metric").first()
    )
    dataset_to_metric = {
        row["dataset"]: (
            "Min Recall"
            if row["fairness_metric"] == "min_recall"
            else "Equal Opportunity"
        )
        for row in dataset_to_metric.iter_rows(named=True)
    }
    label_size = 25
    for ax, dataset in zip(g.axes.flat, g.col_names):
        ax.set_xlabel(dataset_to_metric[dataset], fontsize=label_size)

    g.set_ylabels("Accuracy", fontsize=label_size)
    g.set_titles("{col_name}")

    # Legend at bottom; let Seaborn handle font sizes (no manual overrides)
    sns.move_legend(
        g,
        "lower center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=max(1, int(averaged_threshold_df["method"].n_unique())),
        title=None,
        frameon=True,
    )

    g.figure.set_constrained_layout(True)

    plt.savefig(
        PLOT_DIR
        / f"{backbone}-all_datasets-all_iterations-all_threshold_evaluation.pdf",
        dpi=300,
        bbox_inches="tight",
    )
    plt.clf()


def generate_filter(average_performance: pl.DataFrame) -> pl.Expr:
    best_multi = (
        average_performance.filter(pl.col("method_type") == "Multi")
        .sort("rank")["method"]
        .first()
    )
    multi_filter = (pl.col("method_type") == "Multi") & (pl.col("method") != best_multi)
    combined_filter = ~multi_filter
    return combined_filter


def calculate_improvement_metrics(
    metric: FairnessMetric,
    threshold_df: pl.DataFrame,
    num_iterations: int = 200,
):
    results = []
    total_combinations = threshold_df.group_by("method", "iteration").len().height
    for (method, method_type, iteration), method_df in tqdm(
        threshold_df.group_by("method", "method_type", "iteration"),
        total=total_combinations,
    ):
        improvement = guaranteed_fair_ensemble.fair_auc.calculate(
            method_df=method_df, baseline_scores=None, metric=metric
        )
        improvement_df = pl.DataFrame(
            pl.Series("improvement", [improvement])
        ).with_columns(
            pl.lit(method).alias("method"),
            pl.lit(method_type).alias("method_type"),
            pl.lit(iteration).alias("iteration"),
        )
        results.append(improvement_df)
    improvement_df: pl.DataFrame = pl.concat(results)
    N_SAMPLES = 10
    bootstrap_df = pl.concat(
        [
            _single_bootstrap_improvement(
                improvement_df, n_samples=N_SAMPLES, seed=seed
            )
            for seed in tqdm(range(num_iterations))
        ]
    )
    return bootstrap_df


def _single_bootstrap_improvement(
    improvement_df: pl.DataFrame, n_samples: int, seed: int = 0
) -> pl.DataFrame:
    return (
        improvement_df.group_by("method")
        .agg(pl.all().sample(n_samples, with_replacement=True, seed=seed))
        .explode(pl.all().exclude("method"))
        .group_by("method", "method_type")
        .agg(pl.col("improvement").mean())
    )


def bootstrap_evaluate_violations(
    threshold_df: pl.DataFrame, metric: str = "min_recall", num_iterations: int = 100
) -> pl.DataFrame:
    return pl.concat(
        [
            single_evaluate_violations(
                threshold_df.sample(fraction=1.0, with_replacement=True), metric=metric
            )
            for _ in tqdm(range(num_iterations))
        ]
    )


def single_evaluate_violations(
    threshold_df: pl.DataFrame, metric: str = "min_recall"
) -> pl.DataFrame:
    method_df = threshold_df.filter(
        pl.col("method").str.contains("multi") | (pl.col("method") == "oxonfair")
    )
    comparison = (
        pl.col(metric) - pl.col("threshold")
        if metric == "min_recall"
        else pl.col("threshold") - pl.col(metric)
    )
    return (
        method_df.with_columns(
            comparison.alias("violation"),
        )
        .with_columns(
            pl.when(pl.col("violation") < 0)
            .then(pl.col("violation"))
            .otherwise(0)
            .alias("violation"),
        )
        .group_by(
            "method",
            "iteration",
        )
        .agg(pl.col("violation").mean(), pl.col("accuracy").mean())
    )


def get_test_labels(dataset: str, iteration: int = 0) -> tuple[np.ndarray, np.ndarray]:
    _method_name = "erm"
    prediction_df = pl.read_csv(
        get_prediction_path(_method_name, dataset=dataset, iteration=iteration)
    )
    return prediction_df["true_label"].to_numpy(), prediction_df[
        "protected_attr"
    ].to_numpy()


@profile
@cache
def calculate_random_thresholds(
    num_threshold: int = 100,
    seed: int = 0,
    dataset: str = "papila",
    iteration: int = 0,
    minimum_threshold: float = 0.0,
) -> pl.DataFrame:
    test_labels, test_groups = get_test_labels(dataset=dataset, iteration=iteration)
    np.random.seed(seed=seed)
    random_baselines = np.random.uniform(low=0, high=1, size=(len(test_labels)))
    thresholds = np.linspace(minimum_threshold, 1, num_threshold)
    results = []
    for threshold in thresholds:
        metrics = evaluate_threshold(
            test_labels, random_baselines, test_groups, threshold=threshold
        )
        results.append(metrics)
    return pl.DataFrame(results).with_columns(
        pl.lit("random").alias("method"),
        pl.lit(iteration).alias("iteration"),
    )


def get_single_random_baseline(
    seed: int = 0, dataset: str = "fitzpatrick17k"
) -> pl.DataFrame:
    test_labels, test_groups = get_test_labels(dataset=dataset)
    np.random.seed(seed=seed)
    metrics = evaluate_threshold(
        test_labels,
        np.random.uniform(low=0, high=1, size=(len(test_labels))),
        test_groups,
    )
    return pl.DataFrame([metrics]).with_columns(
        pl.lit("random").alias("method"),
        pl.lit(0).alias("iteration"),
    )


def get_single_trivial_baseline(
    seed: int = 0, dataset: str = "fitzpatrick17k"
) -> pl.DataFrame:
    test_labels, test_groups = get_test_labels(dataset=dataset)
    np.random.seed(seed=seed)
    # No one is sick
    trivial_scores = np.zeros(len(test_labels))
    metrics = evaluate_threshold(
        test_labels,
        trivial_scores,
        test_groups,
    )
    test_labels.mean()
    return pl.DataFrame([metrics]).with_columns(
        pl.lit("random").alias("method"),
    )


def check_fairret_path(
    path: Path, iteration: int = 0, backbone: str = "efficientnet"
) -> bool:
    backbone_string = backbone if backbone != "mobilenetv3" else ""
    string_path = str(path)
    if "scale" not in string_path:
        return False
    if backbone_string and backbone_string not in string_path:
        return False
    return not (iteration > 0 and f"iteration{iteration}" not in string_path)


def get_thresholds(
    dataset: str,
    method_name: str,
    backbone: str = "efficientnet_s",
    iteration: int = 0,
) -> pl.DataFrame:
    model_info = ModelInfo(method=method_name, backbone=backbone)
    dataset_info = get_dataset_info(dataset)
    if method_name in [*SIMPLE_BASELINES, "ensemble"]:
        return _get_baseline_threshold(
            model_info=model_info, iteration=iteration, dataset_name=dataset
        ).with_columns(pl.lit(method_name).alias("method"), _val_rank)
    if method_name == "hpp_ensemble":
        return _get_baseline_threshold(
            model_info=model_info, iteration=iteration, dataset_name=dataset
        ).with_columns(pl.lit(method_name).alias("method"), pl.lit(1).alias("val_rank"))
    if method_name == "fairret":
        combined_dfs = get_fairret_thresholds(dataset, iteration, model_info)
        return combined_dfs
    if method_name == "multiensemble":
        df = get_fairensemble_thresholds(
            data_info=dataset_info, iteration=iteration, backbone=backbone
        )
        return df
    if method_name == "oxonfair":
        file_name = guaranteed_fair_ensemble.names.get_fairensemble_file_path(
            dataset_name=dataset_info.name,
            method=method_name,
            iteration=iteration,
            split="test",
            fairness_metric=dataset_info.fairness_metric,
        )
        return (
            pl.read_csv(OUTPUT_DIR / file_name)
            .with_columns(_val_rank)
            .rename({"constraint_value": "threshold"})
        )
    raise ValueError(f"Unknown method name: {method_name}")


def get_fairensemble_thresholds(
    data_info: DatasetInfo, iteration: int = 0, backbone: str = "efficientnet_s"
) -> pl.DataFrame:
    if backbone != "efficientnet_s":
        raise ValueError("Backbone must be efficientnet_s for multiensemble")
    methods = ["multi"]
    validation_dfs = [
        pl.read_csv(
            guaranteed_fair_ensemble.names.get_fairensemble_file_path(
                dataset_name=data_info.name,
                method=method,
                iteration=iteration,
                split="val",
                fairness_metric=data_info.fairness_metric,
            )
        ).with_columns(pl.lit(method).alias("method"))
        for method in methods
    ]
    columns = validation_dfs[0].columns
    validation_df = (
        pl.concat(df.select(columns) for df in validation_dfs)
        .with_columns(
            (pl.col("ensemble_size").cast(pl.String) + pl.col("method")).alias("method")
        )
        .drop("ensemble_size")
        .rename({"constraint_value": "threshold"})
    )
    improvement = pl.concat(
        [
            val_df.with_columns(
                pl.lit(
                    guaranteed_fair_ensemble.fair_auc.calculate(
                        method_df=val_df,
                        baseline_scores=None,
                        metric=data_info.fairness_metric,
                    )
                ).alias("improvement")
            )
            for (_,), val_df in validation_df.group_by("method")
        ]
    )
    val_rank = (
        improvement.unique(subset=["method", "improvement"])
        .with_columns(
            pl.col("improvement").rank("min", descending=True).alias("val_rank")
        )
        .select("method", "val_rank", pl.col("improvement").alias("val_improvement"))
    )
    logger.debug(f"Validation rank:\n{val_rank.sort('val_rank')}")
    test_results = [
        pl.read_csv(
            guaranteed_fair_ensemble.names.get_fairensemble_file_path(
                dataset_name=data_info.name,
                method=method,
                iteration=iteration,
                split="test",
                fairness_metric=data_info.fairness_metric,
            )
        ).with_columns(pl.lit(method).alias("method"))
        for method in methods
    ]
    test_df = (
        pl.concat(df.select(columns) for df in test_results)
        .with_columns(
            (pl.col("ensemble_size").cast(pl.String) + pl.col("method")).alias("method")
        )
        .drop("ensemble_size")
    )
    return test_df.rename({"constraint_value": "threshold"}).join(
        val_rank, on="method", how="inner"
    )


def get_fairret_thresholds(dataset, iteration, model_info):
    if model_info.method != "fairret":
        raise ValueError("Model info method must be 'fairret'")
    val_results = _get_fairret_results(dataset, iteration, model_info, split="val")
    val_rank = _calculate_val_rank(val_results)
    test_results = _get_fairret_results(dataset, iteration, model_info, split="test")
    return test_results.join(val_rank, on="method")


def _calculate_val_rank(val_results: pl.DataFrame) -> pl.DataFrame:
    val_rank = (
        val_results.group_by("method")
        .agg(pl.col("accuracy").mean())
        .sort("accuracy", descending=True)
        .with_row_index("val_rank")
        .drop("accuracy")
    )
    return val_rank


def _get_fairret_results(dataset, iteration, model_info, split: SplitType = "test"):
    fairret_dfs = []
    for scale in FAIRRET_SCALES:
        model_info.scaling_factor = scale
        fairret_dfs.append(
            _get_baseline_threshold(
                model_info=model_info,
                iteration=iteration,
                dataset_name=dataset,
                split=split,
            ).with_columns(
                pl.lit(
                    f"fairret-{guaranteed_fair_ensemble.names.float_to_str(scale)}"
                ).alias("method")
            )
        )
    combined_dfs: pl.DataFrame = pl.concat(fairret_dfs)
    return combined_dfs


def _get_baseline_threshold(
    model_info: ModelInfo,
    iteration: int = 0,
    dataset_name: str = "fitzpatrick17k",
    split: SplitType = "test",
) -> pl.DataFrame:
    threshold_path = guaranteed_fair_ensemble.names.create_baseline_save_path(
        iteration=iteration, model_info=model_info, dataset_name=dataset_name
    )
    if split == "val":
        threshold_path = guaranteed_fair_ensemble.names.convert_to_val_path(
            threshold_path
        )
    if not threshold_path.exists():
        raise FileNotFoundError(f"Threshold file {threshold_path} does not exist.")
    return pl.read_csv(threshold_path)


def get_prediction_path(
    method_name: str, dataset: str = "ham10000", iteration: int = 0
) -> Path:
    prediction_path = DATA_DIR / dataset / f"{method_name}_rebalanced"
    if iteration > 0:
        prediction_path = prediction_path / f"iteration{iteration}"
    file_name = f"{method_name}_predictions.csv"
    return prediction_path / file_name


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Fitzpatrick Frontier")
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=["all"],
        help="Name of the method to evaluate",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=[data_info.name for data_info in DATASET_HPARAMS],
        help="Datasets to evaluate",
    )
    parser.add_argument(
        "--num_threshold",
        type=int,
        default=100,
        help="Number of thresholds to evaluate",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=3,
        help="Number of iterations to evaluate",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="efficientnet_s",
        choices=["efficientnet", "mobilenetv3", "efficientnet_s"],
    )
    args = parser.parse_args()

    main(
        methods=args.methods,
        datasets=args.datasets,
        backbone=args.backbone,
        iterations=args.iterations,
    )

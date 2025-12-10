""" """

import argparse
from functools import cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import scipy.stats
import seaborn as sns
from dotenv import load_dotenv
from line_profiler import profile
from loguru import logger
from matplotlib.ticker import FuncFormatter
from tqdm import tqdm

import guaranteed_fair_ensemble.colors
import guaranteed_fair_ensemble.fair_auc
import guaranteed_fair_ensemble.names
from guaranteed_fair_ensemble.config import get_dataset_info
from guaranteed_fair_ensemble.constants import (
    ALL_METHODS,
    DATASET_HPARAMS,
    FAIRRET_SCALES,
    PRETTY_METHOD_NAMES,
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


def get_positive_rate(dataset: str = "fairvlmed") -> float:
    spec = get_dataset(dataset)
    all_data = spec.load_and_clean_data(DATA_DIR)
    return all_data[spec.cfg.target_col].mean()


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2.0, n - 1)
    return m, m - h, m + h


def parse_method(path: Path) -> str:
    return path.stem.split("_")[0]


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
                OUTPUT_DIR / "temp" / f"{baseline_path.stem}_observed_rates.csv"
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
    fair_filter = pl.col("method_type").is_in(["Multi", "Joint"])
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
def plot_dataset(
    methods: list[str],
    backbone: str = "efficientnet_s",
    dataset: str = "fitzpatrick17k",
    iterations: int = 3,
    filter_best: bool = True,
) -> None:
    if "all" in methods:
        methods = ALL_METHODS
    # Load the predictions
    dataset_params = get_dataset_info(dataset)
    logger.debug(f"{dataset_params=}")
    fairness_metric = dataset_params.fairness_metric
    all_thresholds = []
    for iteration in range(iterations):
        temp_threshold_df = pl.concat(
            [
                get_thresholds(
                    method_name=method,
                    dataset=dataset,
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
        )
        all_thresholds.append(temp_threshold_df)

    metric_minimum = 0.5 if fairness_metric == "min_recall" else 0.0

    method_type_identifier = (
        pl.when(pl.col("method").str.contains(r"\dmulti"))
        .then(pl.lit("Multi"))
        .when(pl.col("method").str.contains("joint"))
        .then(pl.lit("Joint"))
        .otherwise(pl.lit("Baseline"))
        .alias("method_type")
    )

    threshold_df: pl.DataFrame = (
        pl.concat(all_thresholds)
        .filter(pl.col("threshold") > metric_minimum)
        .with_columns(pl.col("method").replace(PRETTY_METHOD_NAMES))
        .with_columns(method_type_identifier)
        .with_columns(pl.col("method").str.replace("_ensemble", ""))
    )

    threshold_df = filter_ensemble_methods(threshold_df)
    threshold_df, _ = filter_fairret(threshold_df)

    improvement_df = calculate_improvement_metrics(
        fairness_metric,
        threshold_df,
        num_iterations=1000,
    )

    average_performance = (
        improvement_df.group_by("method", "method_type")
        .agg(pl.col("improvement").mean())
        .sort("improvement", descending=True)
        .with_row_index("rank")
        .drop("improvement")
    )

    combined_filter = (
        generate_filter(average_performance) if filter_best else pl.lit(True)
    )

    erm_improvement = improvement_df.filter(pl.col("method") == "ERM")[
        "improvement"
    ].mean()
    assert erm_improvement is not None, "ERM improvement should not be None"

    sns.set_theme(style="whitegrid", font_scale=2)
    plot_df = (
        improvement_df.join(average_performance, on="method", how="inner")
        .sort("rank")
        .filter(combined_filter)
        .with_columns(
            pl.when(pl.col("method_type") == "Baseline")
            .then(pl.lit("Baseline"))
            .otherwise(pl.lit("OxEnsemble  (ours)"))
            .alias("MethodType"),
            (pl.col("improvement") - erm_improvement).alias("relative_improvement"),
        )
    )

    logger.debug(
        f"Plotting mean: {plot_df.group_by('method').agg(pl.col('improvement').mean())}"
    )

    # Make figure wider
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(
        plot_df.with_columns(guaranteed_fair_ensemble.names.normalise_method_names),
        x="method",
        y="relative_improvement",
        hue="MethodType",
        errorbar=("ci", 99),
        palette=guaranteed_fair_ensemble.colors.get_method_type_colours(),
    )
    # Rotate x labels 45 degrees
    # Annotate means using seaborn's containers (matplotlib backend)
    # Annotate outside bars, offset above error bar
    for container in ax.containers:
        for bar in container:
            height = bar.get_height()  # + erm_improvement
            if np.isnan(height):
                continue
            # small offset relative to axis scale
            offset = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.01

            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + offset,
                f"{height + erm_improvement:.3f}",
                ha="center",
                va="bottom",
                fontsize=20,
                fontweight="bold",
                color="black",  # always black since it's outside
            )
    sns.move_legend(
        ax,
        "lower center",
        bbox_to_anchor=(0.5, 1),
        ncol=3,
        title=None,
        frameon=False,
    )
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda y, pos: f"{y + erm_improvement:.2f}")  # noqa: ARG005
    )
    ax.margins(y=0.1)  # adds 10% extra space above/below bars
    ax.set_ylabel(f"FairAUC ({fairness_metric.replace('_', ' ').title()})")
    # Re-enable + style the tick marks
    ax.tick_params(
        axis="x",
        which="both",
        bottom=True,  # show bottom ticks
        top=False,  # hide top ticks
        length=6,  # visible tick
        width=1.2,
        direction="out",
    )

    # Rotate & right-align label text
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.tick_params(axis="x", length=8, width=1.5, direction="out")  # add ticks

    plt.tight_layout()
    plt.savefig(
        PLOT_DIR
        / f"{backbone}-{dataset}-all_threshold_evaluation_{fairness_metric}_improvement.pdf",
    )
    plt.clf()

    markdown_improvements = (
        improvement_df.group_by("method")
        .agg(
            pl.col("improvement").mean().alias("mean_improvement"),
            (pl.col("improvement").std() / pl.len().sqrt()).alias("std_error"),
        )
        .with_columns(
            (pl.col("mean_improvement") + pl.col("std_error")).alias("upper_bound"),
            (pl.col("mean_improvement") - pl.col("std_error")).alias("lower_bound"),
        )
        .join(
            average_performance,
            on="method",
            how="inner",
        )
        .sort("rank")
        .with_columns(
            # floats to 3 decimal places
            pl.selectors.float().round(3).cast(pl.String)
        )
        .to_pandas()
        .to_markdown()
    )
    (OUTPUT_DIR / f"{backbone}-{dataset}_improvement_table.md").write_text(
        markdown_improvements
    )

    violations = bootstrap_evaluate_violations(
        threshold_df=threshold_df, metric=fairness_metric
    )
    mean_violation = (
        violations.group_by("method")
        .agg(
            pl.col("violation").mean(),
            pl.col("accuracy").mean(),
        )
        .sort("violation", descending=True)
        .with_row_index("violation_rank")
        .sort("accuracy", descending=True)
        .with_row_index("accuracy_rank")
    )

    violations_ranked = violations.join(
        mean_violation.drop("violation", "accuracy"), on="method"
    )
    violations_ranked.write_csv(
        OUTPUT_DIR / f"{backbone}-{dataset}_{fairness_metric}_multihead_violations.csv"
    )


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
        pl.col("method").str.contains("joint|multi") | (pl.col("method") == "oxonfair")
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


def main(
    methods: list[str], datasets: list[str], backbone: str, iterations: int
) -> None:
    for dataset in datasets:
        logger.info(f"Processing dataset: {dataset}")
        plot_dataset(
            methods=methods,
            dataset=dataset,
            backbone=backbone,
            iterations=iterations,
        )


if __name__ == "__main__":
    eligible_datasets = [
        ds.name for ds in DATASET_HPARAMS if ds.fairness_metric == "min_recall"
    ]
    parser = argparse.ArgumentParser(description="Evaluate Fitzpatrick Frontier")
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=["all"],
        choices=["all", *ALL_METHODS],
        help="Name of the method to evaluate",
    )
    parser.add_argument(
        "--num_threshold",
        type=int,
        default=100,
        help="Number of thresholds to evaluate",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=eligible_datasets,
        choices=eligible_datasets,
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

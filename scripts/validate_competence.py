import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
import torch
from loguru import logger

import guaranteed_fair_ensemble.datasets
import guaranteed_fair_ensemble.predict
import guaranteed_fair_ensemble.preprocess
from guaranteed_fair_ensemble.config import get_dataset_info
from guaranteed_fair_ensemble.constants import DEFAULT_SEED
from guaranteed_fair_ensemble.data.registry import get_dataset
from guaranteed_fair_ensemble.data_models import ModelInfo, SplitType, TrainingInfo
from guaranteed_fair_ensemble.directories import DATA_DIR, PLOT_DIR


def validate_competence(error_rates: np.ndarray, ts: np.ndarray | None = None) -> float:
    """
    Returns the worst interval
    """
    if ts is None:
        ts = np.linspace(0, 0.5, num=200)
    worst_difference = 1.0
    for t in ts:
        good_mass = (error_rates >= t) & (error_rates < 0.5)
        bad_mass = (error_rates >= 0.5) & (error_rates <= 1 - t)
        difference = np.mean(good_mass) - np.mean(bad_mass)
        if difference < worst_difference:
            worst_difference = difference
    return float(worst_difference)


def calculate_threshold_violations(
    example: pl.DataFrame, num_trials: int = 200
) -> pl.DataFrame:
    if num_trials < 1:
        raise ValueError("num_trials must be at least 1")
    threshold_violations = []
    group = "all"
    for (threshold, group), data in example.group_by("threshold", "groups"):
        truth_np = data["truth"].to_numpy()
        folds_np = data.drop("truth", "groups", "threshold").to_numpy()
        errors = folds_np != truth_np[:, None]
        sample_rho = np.mean(errors, axis=1)
        ts = np.linspace(0, 0.5, num=num_trials)
        competences = []
        for t in ts:
            good_mass = np.mean((sample_rho >= t) & (sample_rho < 0.5))
            bad_mass = np.mean((sample_rho >= 0.5) & (sample_rho <= 1 - t))
            competence_mass = {
                "t": t,
                "good": good_mass,
                "bad": bad_mass,
            }
            competences.append(competence_mass)
        threshold_violations.append(
            pl.DataFrame(competences).with_columns(
                minimum_rate=pl.lit(threshold), group=pl.lit(group)
            )
        )
    plot_data = pl.concat(threshold_violations)
    return plot_data


def get_long_predictions(test_data, fairness_thresholds, predictions):
    long_predictions = pl.DataFrame(
        [pl.Series("groups", test_data.groups), pl.Series("truth", test_data.labels)]
    )
    _all_long_predictions = []
    for constraint_idx, constraint in enumerate(fairness_thresholds):
        fold_preds = predictions[:, :, constraint_idx].numpy() > 0
        prediction_polars = pl.DataFrame(
            {f"fold{i}": fold_preds[:, i] for i in range(fold_preds.shape[1])}
        )
        _long_pred = long_predictions.with_columns(prediction_polars).with_columns(
            pl.lit(constraint).alias("threshold")
        )
        _all_long_predictions.append(_long_pred)
    all_long_predictions = pl.concat(_all_long_predictions)
    return all_long_predictions


def get_plot_data(dataset_info, spec, full_df, all_features, iteration: int = 0):
    all_long_predictions = generate_long_predictions(
        dataset_info, spec, full_df, all_features, iteration
    )
    plot_data = (
        calculate_threshold_violations(
            all_long_predictions.filter(pl.col("truth") == 1)
        )
        # .filter(is_close_expr)
        .with_columns(
            competence_diff=pl.col("good") - pl.col("bad"),
        )
    )
    return plot_data


def generate_long_predictions(
    dataset_info,
    spec,
    full_df,
    all_features,
    iteration,
    split: SplitType = "test",
    trained_on: SplitType = "val",
):
    if split not in ["val", "test"]:
        raise ValueError("split must be 'val' or 'test'")
    seed = DEFAULT_SEED + iteration
    train_data, test_df = guaranteed_fair_ensemble.datasets.split_data(
        full_df, cfg=spec.cfg, test_size=dataset_info.test_size, random_seed=seed
    )
    target_data = train_data if split == "val" else test_df
    pred_data = guaranteed_fair_ensemble.datasets.construct_minimal_data(
        spec=spec, all_features=all_features, dataframe=target_data
    )
    fit_suffix = "" if trained_on == "val" else "-test"
    heads_path = (
        DATA_DIR
        / f"{dataset_info.name}-iteration{iteration}-multi-fitted-heads{fit_suffix}.pth"
    )
    heads = torch.load(heads_path, weights_only=False)
    fairness_thresholds = np.array(list(heads.keys()))
    # Predictions shape: (N, ensemble_size, constraints)
    predictions = guaranteed_fair_ensemble.predict.predict_across_thresholds(
        merged_heads=heads,
        features=pred_data.features,
        fairness_thresholds=fairness_thresholds,
    )
    all_long_predictions = get_long_predictions(
        pred_data, fairness_thresholds, predictions
    )

    return all_long_predictions


def add_binary_threshold(
    df: pl.DataFrame, minimum_threshold: float = 0.7
) -> pl.DataFrame:
    return df.with_columns(
        (pl.col("minimum_rate") > minimum_threshold).alias("above_threshold")
    )


def _calculate_error_by_group(iteration, long_positives):
    recall_competence_plot_data = (
        long_positives.with_row_index("case_id")
        .unpivot(
            on=pl.selectors.starts_with("fold"),
            index=["case_id", "groups", "truth", "threshold"],
        )
        .group_by("case_id", "groups", "threshold")
        .agg((1 - pl.col("value").mean()).alias("error_rate"))
        .with_columns(pl.lit(iteration).alias("iteration"))
    )
    return recall_competence_plot_data


def get_low_scoring(long_predictions):
    low_scoring = (
        long_predictions.filter(pl.col("truth") == 1)
        .group_by("threshold", "groups")
        .mean()
        .drop("truth")
        .unpivot(
            index=["threshold", "groups"],
        )
        .group_by("threshold", "groups")
        .agg(pl.col("value").mean())
        .rename({"value": "mean_recall"})
    )
    return low_scoring


def competence_on_dataset(
    combined_recall_competence_val, recall_threshold: float = 0.5
) -> pl.DataFrame:
    comptence_thresholds = np.linspace(0.0, 0.5, num=200)
    grouping_keys = ["minimum_rate", "dataset", "groups"]
    _data = []
    for (
        threshold,
        dataset,
        group,
    ), group_data in combined_recall_competence_val.filter(
        pl.col("mean_recall") < recall_threshold
    ).group_by(grouping_keys):
        worst_difference = validate_competence(
            group_data["error_rate"].to_numpy(), ts=comptence_thresholds
        )
        data_attributes = f"Dataset: {dataset}, Minimum Rate: {threshold:.2f}, Group: {group}, Valid Competence: {worst_difference} (Mean Recall: {group_data['mean_recall'].first()})"
        if worst_difference >= 0:
            logger.warning(f"All Trials Valid! {data_attributes}")
        _data.append(
            group_data.select(grouping_keys).with_columns(
                pl.lit(worst_difference).alias("is_competent")
            )
        )
    return pl.concat(_data)


if __name__ == "__main__":
    DATASETS = ["ham10000", "fitzpatrick17k"]

    _combined_plot_data = []
    _combined_recall_competence_plot_data_test = []
    _combined_recall_competence_plot_data_val = []
    _combined_low_scorers = []
    for dataset in DATASETS:
        dataset_info = get_dataset_info(dataset)
        logger.debug(f"Dataset info: {dataset_info}")
        model_info = ModelInfo(method="ensemble", backbone="efficientnet_s")
        training_info = TrainingInfo(dataset=dataset_info, model=model_info)
        spec = get_dataset(dataset)
        full_df = spec.load_and_clean_data(DATA_DIR)
        all_features = guaranteed_fair_ensemble.preprocess.get_features(
            dataset_name=dataset, backbone_name=model_info.backbone
        )
        dataset_info = training_info.dataset
        ITERATIONS = 3
        _all_plot_data = []
        _all_recall_competence_plot_data = []
        _recall_competence_val = []
        _combined_recall_competence_val = []
        for iteration in range(ITERATIONS):
            long_predictions_test_on_test: pl.DataFrame = generate_long_predictions(
                dataset_info,
                spec,
                full_df,
                all_features,
                iteration,
                split="test",
                trained_on="test",
            )
            long_positives_test_on_test = long_predictions_test_on_test.filter(
                pl.col("truth") == 1
            )
            long_predictions_test_on_val: pl.DataFrame = generate_long_predictions(
                dataset_info,
                spec,
                full_df,
                all_features,
                iteration,
                split="test",
                trained_on="val",
            )
            long_positives_test_on_val = long_predictions_test_on_val.filter(
                pl.col("truth") == 1
            )
            recall_competence_plot_data = _calculate_error_by_group(
                iteration, long_positives_test_on_test
            )
            recall_competence_val = _calculate_error_by_group(
                iteration, long_positives_test_on_val
            )
            low_scoring_val = get_low_scoring(
                long_predictions_test_on_val
            ).with_columns(
                pl.lit(iteration).alias("iteration"),
                pl.lit(dataset).alias("dataset"),
                pl.lit("val").alias("split"),
            )
            low_scoring_test = get_low_scoring(
                long_predictions_test_on_test
            ).with_columns(
                pl.lit(iteration).alias("iteration"),
                pl.lit(dataset).alias("dataset"),
                pl.lit("test").alias("split"),
            )
            _combined_low_scorers.append(low_scoring_test)
            _combined_low_scorers.append(low_scoring_val)
            _recall_competence_val.append(recall_competence_val)
            _all_recall_competence_plot_data.append(recall_competence_plot_data)
            test_plot_data = get_plot_data(
                dataset_info=dataset_info,
                spec=spec,
                full_df=full_df,
                all_features=all_features,
                iteration=iteration,
            ).with_columns(pl.lit(iteration).alias("iteration"))
            _all_plot_data.append(test_plot_data)
        all_plot_data = (
            pl.concat(_all_plot_data).group_by("t", "group", "minimum_rate").mean()
        ).with_columns(pl.lit(dataset).alias("dataset"))
        _combined_plot_data.append(
            all_plot_data.with_columns(pl.lit(dataset).alias("dataset"))
        )
        _combined_recall_competence_plot_data_test.append(
            pl.concat(_all_recall_competence_plot_data).with_columns(
                pl.lit(dataset).alias("dataset")
            )
        )
        logger.debug(f"Recall competence val data: {dataset}")
        _combined_recall_competence_plot_data_val.append(
            pl.concat(_recall_competence_val).with_columns(
                pl.lit(dataset).alias("dataset")
            )
        )

    combined_plot_data = pl.concat(_combined_plot_data)
    low_scorers_combined = (
        pl.concat(_combined_low_scorers)
        .group_by("threshold", "groups", "dataset", "split")
        .mean()
        .drop("iteration")
        .rename({"threshold": "minimum_rate"})
    )

    only_divisible = pl.col("minimum_rate") % 0.05 == 0
    combined_recall_competence_test = (
        pl.concat(_combined_recall_competence_plot_data_test)
        .rename({"threshold": "minimum_rate"})
        .filter(only_divisible)
        .join(
            low_scorers_combined.filter(pl.col("split") == "test"),
            on=["minimum_rate", "groups", "dataset"],
            how="inner",
        )
    )
    test_competence = competence_on_dataset(
        combined_recall_competence_test, recall_threshold=1
    ).unique()

    combined_recall_competence_val = (
        pl.concat(_combined_recall_competence_plot_data_val)
        .rename({"threshold": "minimum_rate"})
        .filter(pl.col("minimum_rate").is_between(0.0, 1.0))
        .join(
            low_scorers_combined.filter(pl.col("split") == "val"),
            on=["minimum_rate", "groups", "dataset"],
            how="inner",
        )
    )
    val_competence = competence_on_dataset(
        combined_recall_competence_val, recall_threshold=2.0
    ).unique()

    val_plot_data = combined_recall_competence_val.join(
        val_competence, on=["minimum_rate", "dataset", "groups"]
    )

    test_plot_data = combined_recall_competence_test.join(
        test_competence, on=["minimum_rate", "dataset", "groups"]
    ).filter(pl.col("minimum_rate").is_between(0.0, 1.5))

    combined_plot_data = pl.concat([test_plot_data, val_plot_data]).with_columns(
        pl.col("split")
        .str.replace("val", "Val → Test")
        .str.replace("test", "Test → Test")
    )

    sns.set_theme(style="whitegrid", font_scale=2.3)
    simpler_plot = sns.relplot(
        combined_plot_data,
        kind="line",
        col="split",
        x="mean_recall",
        palette="colorblind",
        y="is_competent",
        # Increase line width
        linewidth=4,
        style="groups",
        hue="dataset",
        # make it very wide
        aspect=2,
    )
    simpler_plot.set_ylabels(r"$C_\rho^\text{min}$")
    simpler_plot.set_xlabels("")
    simpler_plot.figure.supxlabel("Observed Minimum Recall")
    # Move legend to the bottom
    sns.move_legend(
        simpler_plot,
        "lower center",
        bbox_to_anchor=(0.45, -0.15),
        ncol=8,
        frameon=True,
    )
    for ax in simpler_plot.axes.flat:
        ax.axvline(0.5, color="black", linestyle="--")
    for ax in simpler_plot.axes.flat:
        size = ax.yaxis.label.get_size() * 1.4
        ax.set_ylabel(ax.get_ylabel(), fontsize=size)

    simpler_plot.set_titles(col_template="{col_name}")
    plt.savefig(PLOT_DIR / "recall_competence_simple_plot.png", bbox_inches="tight")
    plt.savefig(PLOT_DIR / "recall_competence_simple_plot.pdf", bbox_inches="tight")
    plt.clf()
    sns.set_theme(style="whitegrid", font_scale=1.2)

    # val_plot = sns.catplot(
    #     data=combined_recall_competence_val,
    #     x="minimum_rate",
    #     kind="violin",
    #     row="dataset",
    #     y="error_rate",
    #     height=6,
    #     aspect=3,
    #     hue="groups",
    #     palette="Dark2",
    # )
    # val_plot.figure.supxlabel("Minimum Recall")
    # val_plot.figure.supylabel("Error Rate for Positive Class")
    # val_plot.set_axis_labels("", "")
    # labels = [
    #     f"{v:.2f}"
    #     for v in sorted(combined_recall_competence_val["minimum_rate"].unique())
    # ]
    # val_plot.set_titles(row_template="{row_name}")
    # for ax in val_plot.axes.flat:
    #     ax.set_xticks(range(len(labels)))
    #     ax.set_xticklabels(labels, rotation=0)
    # # Move legend outside plot to the right
    # plt.savefig(PLOT_DIR / "recall_competence_val_violin_plot.png", bbox_inches="tight")
    # plt.clf()

    # graph = sns.relplot(
    #     data=add_binary_threshold(combined_plot_data),
    #     x="t",
    #     y="competence_diff",
    #     col="group",
    #     row="dataset",
    #     hue="above_threshold",
    #     # palette=sns.color_palette("coolwarm", as_cmap=True),
    # )
    # # Add hline at 0 for each subplot
    # for ax in plt.gcf().axes:
    #     ax.axhline(0, color="black", linestyle="--")
    # graph.figure.supxlabel("T")
    # graph.figure.supylabel("Competence Mass Difference (Good - Bad)")
    # # Set axis labels to None
    # graph.set_xlabels("")
    # graph.set_ylabels("")
    # plt.savefig(PLOT_DIR / "competence_mass_plot_by_group.png", bbox_inches="tight")
    # plt.clf()

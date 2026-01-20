from pathlib import Path
from typing import Any

import lightning
import numpy as np
import polars as pl
import torch
from joblib import Memory
from loguru import logger
from tqdm import tqdm

import guaranteed_fair_ensemble.backbone
import guaranteed_fair_ensemble.metrics
import guaranteed_fair_ensemble.names
from guaranteed_fair_ensemble.constants import (
    DATASET_HPARAMS,
    DEFAULT_SEED,
)
from guaranteed_fair_ensemble.data.registry import get_dataset
from guaranteed_fair_ensemble.data_models import (
    DatasetInfo,
    FairnessMetric,
    ModelInfo,
    TrainingInfo,
)
from guaranteed_fair_ensemble.datasets import (
    create_data_loaders,
    create_dataset,
    split_data,
)
from guaranteed_fair_ensemble.directories import DATA_DIR, OUTPUT_DIR
from guaranteed_fair_ensemble.models.fairensemble_lit import FairEnsemble
from guaranteed_fair_ensemble.predict import predict_results
from guaranteed_fair_ensemble.torch_utils import reverse_one_hot
from guaranteed_fair_ensemble.transforms import get_transforms

CACHE_DIR = DATA_DIR / ".joblib_cache"


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


def get_sizes(dataset_name: str) -> tuple[float, float]:
    for params in DATASET_HPARAMS:
        if params.name != dataset_name:
            continue
        return params.val_size, params.test_size
    raise ValueError(f"Dataset '{dataset_name}' not found in DATASET_HPARAMS.")


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
    train_df, test_df = split_data(
        df=full_data,
        cfg=spec.cfg,
        test_size=test_size,
        random_seed=DEFAULT_SEED + iteration,
    )

    train_split, val_split = split_data(
        df=train_df,
        cfg=spec.cfg,
        random_seed=DEFAULT_SEED + iteration,
        test_size=val_size,
    )

    if img_dict is None:
        img_dict = get_image_dict(spec)
    else:
        logger.debug("Reusing existing image dictionary")

    train_tfms = get_transforms(
        is_train=True, backbone_name=training_info.model.backbone
    )
    eval_tfms = get_transforms(
        is_train=False, backbone_name=training_info.model.backbone
    )

    train_ds = create_dataset(
        df=train_split,
        img_dict=img_dict,
        cfg=spec.cfg,
        transform_ops=train_tfms,
        rebalance=True,
    )

    val_ds = create_dataset(
        df=val_split,
        img_dict=img_dict,
        cfg=spec.cfg,
        transform_ops=eval_tfms,
        rebalance=True,
    )

    train_loader, val_loader = create_data_loaders(
        train_dataset=train_ds,
        val_dataset=val_ds,
        batch_size=batch_size,
        rebalance=True,
    )

    # ── Test
    test_ds = create_dataset(
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


def get_image_dict(spec):
    img_dir = DATA_DIR / spec.cfg.img_relpath
    # Load images into memory ------------------------------------------------------
    image_paths = list(img_dir.glob("*.jpg"))
    img_dict = spec.load_images(image_paths)
    return img_dict


def get_training_info(dataset_params: DatasetInfo, model_info: ModelInfo):
    training_info = TrainingInfo(
        dataset=dataset_params,
        model=model_info,
    )
    return training_info


def initialize_model(
    training_info: TrainingInfo,
    model_path: Path,
) -> lightning.LightningModule:
    spec = get_dataset(name=training_info.dataset.name)
    num_heads = (
        spec.cfg.num_protected_classes + 1 if spec.cfg.num_protected_classes > 2 else 2
    )
    logger.info(
        f"Initializing model for method '{training_info.model.method}' with {num_heads} heads"
    )
    model_info = training_info.model
    lit_model = guaranteed_fair_ensemble.backbone.initialize_model_checkpoint(
        model_info=model_info, checkpoint_path=model_path, num_heads=num_heads
    )
    return lit_model


def ensemble_predict(
    model: FairEnsemble, data_loader: torch.utils.data.DataLoader
) -> pl.DataFrame:
    model.eval()
    predictions = []
    ys = []
    for batch in tqdm(data_loader):
        image, y_raw, _ = batch
        probabilities = model.ensemble_predict(image.to(model.device))
        predictions.append(probabilities)
        ys.append(y_raw)
    probs = torch.cat(predictions).cpu().numpy()
    ys_vec = torch.cat(ys)
    group = reverse_one_hot(ys_vec[:, 1:]).cpu().numpy()
    labels = ys_vec[:, 0].cpu().numpy()
    return pl.DataFrame(
        {"prediction": probs, "protected_attr": group, "true_label": labels}
    )


def calculate_optimal_thresholds(
    val_loader: Any,
    test_loader: Any,
    model: lightning.LightningModule,
    val_save_path: Path | None = None,
    test_save_path: Path | None = None,
    fairness_metric: FairnessMetric = "min_recall",
):
    val_predictions = get_model_predictions(val_loader, model)
    test_predictions = get_model_predictions(test_loader, model)
    prediction_thresholds = np.linspace(0.0, 1.0, num=101)
    observed_rate_df = prediction_over_threshold(val_predictions, prediction_thresholds)
    if test_save_path is not None:
        observed_test_rate_df = prediction_over_threshold(
            val_predictions=test_predictions,
            prediction_thresholds=prediction_thresholds,
        )
        observed_test_rate_df.write_csv(
            OUTPUT_DIR / "temp" / f"{test_save_path.stem}_observed_rates.csv"
        )
    fairness_thresholds = find_optimal_threshold(
        observed_rate_df, fairness_metric=fairness_metric
    )
    if val_save_path is not None:
        logger.info(f"Saving validation observed rates to {val_save_path}")
        fairness_thresholds.write_csv(val_save_path)
    optimal_thresholds = compute_thresholded_predictions(
        test_predictions, fairness_thresholds
    )
    return optimal_thresholds


def get_model_predictions(data_loader, model):
    val_predictions = (
        pl.DataFrame(predict_results(model=model, loader=data_loader, device="cuda"))
        if not isinstance(model, FairEnsemble)
        else ensemble_predict(model=model, data_loader=data_loader)
    )
    return val_predictions


def save_path_outdated(model_path: Path, save_path: Path) -> bool:
    if not save_path.exists():
        logger.warning(f"Save path {save_path} does not exist")
        return True
    model_updated_at = model_path.stat().st_mtime
    save_path_updated = save_path.stat().st_mtime if save_path.exists() else 0
    return model_updated_at > save_path_updated


def main(batch_size, backbone, overwrite, num_iterations):
    baseline_model_infos = guaranteed_fair_ensemble.names.get_all_model_infos(
        backbone=backbone
    )
    for dataset_params in tqdm(DATASET_HPARAMS, desc="Datasets"):
        img_dict = None
        for baseline_info in tqdm(baseline_model_infos, desc="Baselines", leave=False):
            if baseline_info.method != "hpp_ensemble":
                logger.info(
                    f"Constructing validation frontier for {dataset_params.name} - {baseline_info.method}"
                )
                continue
            training_info = get_training_info(dataset_params, model_info=baseline_info)
            for iteration in tqdm(
                range(num_iterations), desc="Iterations", leave=False
            ):
                save_path = guaranteed_fair_ensemble.names.create_baseline_save_path(
                    iteration,
                    model_info=baseline_info,
                    dataset_name=dataset_params.name,
                )
                val_save_path = guaranteed_fair_ensemble.names.convert_to_val_path(
                    save_path
                )
                logger.debug(
                    f"Save path: {save_path.name}, val path: {val_save_path.name}"
                )
                model_path = guaranteed_fair_ensemble.names.get_model_path(
                    info=training_info, iteration=iteration
                )
                if (
                    not overwrite
                    and not save_path_outdated(model_path, save_path)
                    and not save_path_outdated(model_path, val_save_path)
                ):
                    logger.info(
                        f"Skipping existing results for {training_info.dataset.name} - {training_info.model.method} - iteration {iteration}"
                    )
                    continue

                model = initialize_model(
                    training_info=training_info, model_path=model_path
                )

                _, val_loader, test_loader, img_dict = get_data_loaders(
                    training_info=training_info,
                    iteration=iteration,
                    batch_size=batch_size,
                    img_dict=img_dict,
                )
                optimal_thresholds = calculate_optimal_thresholds(
                    val_loader=val_loader,
                    test_loader=test_loader,
                    model=model,
                    val_save_path=val_save_path,
                    test_save_path=save_path,
                    fairness_metric=dataset_params.fairness_metric,
                )
                optimal_thresholds.write_csv(save_path)
                # Clean up memory from model
                logger.info("Cleaning up model from memory")
                del model
                del val_loader
                del test_loader
                torch.cuda.empty_cache()


if __name__ == "__main__":
    BATCH_SIZE = 1024
    BACKBONE = "mobilenetv3"
    OVERWRITE = False
    NUM_ITERATIONS = 3

    main(
        batch_size=BATCH_SIZE,
        backbone=BACKBONE,
        overwrite=OVERWRITE,
        num_iterations=NUM_ITERATIONS,
    )

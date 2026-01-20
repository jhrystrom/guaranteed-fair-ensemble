import re
from pathlib import Path

import polars as pl
from loguru import logger

from guaranteed_fair_ensemble.config import args_to_info, get_dataset_info
from guaranteed_fair_ensemble.constants import FAIRRET_SCALES, SIMPLE_BASELINES
from guaranteed_fair_ensemble.data_models import (
    FairMethod,
    FairnessMetric,
    ModelInfo,
    SplitType,
    TrainingInfo,
)
from guaranteed_fair_ensemble.directories import CHECKPOINT_DIR, OUTPUT_DIR


def create_baseline_save_path(
    iteration: int,
    model_info: ModelInfo,
    dataset_name: str,
    is_val: bool = False,
) -> Path:
    fairness_metric = get_dataset_info(dataset_name).fairness_metric
    iteration_string = f"-iteration{iteration}" if iteration > 0 else ""
    model_method = model_info.method
    if model_method == "fairret":
        model_method += f"-{float_to_str(model_info.scaling_factor)}"
    save_name = f"{dataset_name}-{fairness_metric}{iteration_string}_{model_method}{model_info.backbone_suffix}_threshold_evaluation.csv"
    if is_val:
        save_name = save_name.replace(".csv", "_val.csv")
    logger.debug(f"Save name: {save_name}")
    save_path = OUTPUT_DIR / save_name
    return save_path


def get_all_model_infos(backbone: str) -> list[ModelInfo]:
    model_infos = []
    for baseline in SIMPLE_BASELINES:
        model_infos.append(ModelInfo(method=baseline, backbone=backbone))
    for scaling_factor in FAIRRET_SCALES:
        model_infos.append(
            ModelInfo(
                method="fairret", backbone=backbone, scaling_factor=scaling_factor
            )
        )
    model_infos.append(ModelInfo(method="ensemble", backbone=backbone))
    return model_infos


def convert_to_val_path(save_path: Path) -> Path:
    return save_path.with_name(save_path.stem + "_val" + save_path.suffix)


def float_to_str(value: float) -> str:
    return str(value).replace(".", "_")


def str_to_float(value: str) -> float:
    return float(value.replace("_", "."))


def extract_validation(path: Path) -> float:
    """
    extract the validation size from the path name (format in path is 'val0_x' -> 0.x)
    """
    match = re.search(r"val0_(\d+)", str(path))
    if match:
        return float(match.group(1)) / 10
    raise ValueError(f"Validation size not found in path: {path}")


def extract_iteration(path: Path) -> int:
    """
    extract the iteration number from the path name (format in path is 'iter_x' -> x)
    """
    match = re.search(r"iteration(\d+)", str(path))
    if match:
        return int(match.group(1))
    return 0


def extract_fold(path: Path) -> int:
    """
    extract the fold number from the path name (format in path is 'fold_x' -> x)
    """
    match = re.search(r"fold(\d+)", str(path))
    if match:
        return int(match.group(1))
    return 0


def get_method_name(args):
    training_info = args_to_info(args)
    return get_method_name_raw(
        model_info=training_info.model, val_size=training_info.dataset.val_size
    )


def get_method_name_raw(model_info: ModelInfo, val_size: float):
    method_name = model_info.method
    if method_name == "fairret":
        method_name = get_fairret_name(model_info, method_name)
    elif method_name == "ensemble":
        method_name = f"21{method_name}_multihead"
    if model_info.rebalanced:
        method_name += "_rebalanced"
    if val_size:
        method_name += f"_val{float_to_str(val_size)}"
    backbone_str = (
        "" if model_info.backbone == "mobilenetv3" else f"_{model_info.backbone}"
    )
    return f"{method_name}{backbone_str}"


def get_fairret_name(model_info: ModelInfo, method_name):
    scaling_str = str(model_info.scaling_factor).replace(".", "_")
    method_name = f"{method_name}_scale{scaling_str}"
    return method_name


def get_model_path(info: TrainingInfo, iteration: int = 0) -> Path:
    model_dir = (
        CHECKPOINT_DIR
        / info.dataset.name
        / get_method_name_raw(model_info=info.model, val_size=info.val_size)
    )
    if iteration > 0 or (
        info.model.method == "ensemble" and info.model.backbone != "mobilenetv3"
    ):
        model_dir /= f"iteration{iteration}"
    try:
        return next(model_dir.glob("*.ckpt"))
    except StopIteration:
        raise FileNotFoundError(f"No checkpoint found in directory: {model_dir}")


def get_fairensemble_file_path(
    dataset_name: str,
    iteration: int,
    fairness_metric: FairnessMetric,
    method: FairMethod,
    split: SplitType,
    backbone: str = "efficientnet_s",
) -> Path:
    backbone_suffix = "" if backbone == "efficientnet_s" else f"_{backbone}"
    FILE_PATTERN = "{dataset}-iteration{iteration}-{fairness_metric}-{method}-ensemble-{split}{backbone}.csv"
    path = OUTPUT_DIR / FILE_PATTERN.format(
        dataset=dataset_name,
        iteration=iteration,
        fairness_metric=fairness_metric,
        method=method,
        split=split,
        backbone=backbone_suffix,
    )
    logger.debug(f"OxEnsemble  file path: {path.name}")
    return path


normalise_method_names = (
    pl.when(pl.col("method").str.ends_with("multi"))
    .then(pl.lit("OxEnsemble"))
    .when(pl.col("method").str.contains("fairret"))
    .then(pl.lit("fairret"))
    .otherwise(pl.col("method"))
    .alias("method")
)

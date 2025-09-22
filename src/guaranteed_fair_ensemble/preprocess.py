from itertools import batched

import pandas as pd
import torch
import torch.nn as nn
from joblib import Memory
from loguru import logger
from tqdm import tqdm

import guaranteed_fair_ensemble.backbone
import guaranteed_fair_ensemble.names
from guaranteed_fair_ensemble.config import get_dataset_info
from guaranteed_fair_ensemble.data.registry import DatasetSpec, get_dataset
from guaranteed_fair_ensemble.data_models import ModelInfo, TrainingInfo
from guaranteed_fair_ensemble.directories import DATA_DIR
from guaranteed_fair_ensemble.models.fairensemble_lit import (
    FairEnsemble,
    load_guaranteed_fair_ensemble_from_checkpoint,
)
from guaranteed_fair_ensemble.torch_utils import get_device
from guaranteed_fair_ensemble.transforms import get_transforms

memory = Memory(location=DATA_DIR / "joblib_cache", verbose=0)


def get_models_multihead(
    training_info: TrainingInfo,
    spec: DatasetSpec,
    iteration: int,
) -> FairEnsemble:
    """
    Load the multi-head ensemble model.
    """

    ckpt_path = guaranteed_fair_ensemble.names.get_model_path(
        info=training_info, iteration=iteration
    )
    logger.debug(f"Loading model from {ckpt_path}")

    # Calculate number of heads per ensemble member
    num_heads_per_member = (
        spec.cfg.num_protected_classes + 1 if spec.cfg.num_protected_classes != 2 else 2
    )

    # Create backbone with ensemble_members * num_heads_per_member heads
    backbone = guaranteed_fair_ensemble.backbone.get_backbone(
        name=training_info.model.backbone,
        num_heads=training_info.model.ensemble_members * num_heads_per_member,
        freeze=True,  # Assuming backbone is frozen
    )

    model = load_guaranteed_fair_ensemble_from_checkpoint(
        ckpt_path=ckpt_path,
        backbone=backbone,
        num_heads_per_member=num_heads_per_member,
        device=get_device(),
    )

    # Return the model
    return model


def batch_predict(
    transformed_images: torch.Tensor,
    model: FairEnsemble,
    batch_size: int = 512,
):
    all_features = []
    device = get_device()
    model = model.to(device)
    index = list(range(transformed_images.shape[0]))
    for batch_idx in tqdm(batched(index, batch_size)):
        batch_images = transformed_images[torch.tensor(batch_idx)].to(model.device)
        with torch.no_grad():
            batch_features = model.extract_features(batch_images).cpu()
        all_features.append(batch_features)
    all_features = torch.cat(all_features, dim=0).cpu()
    # Remove model from device
    model = model.to("cpu")
    return all_features


def get_image_dict(spec: DatasetSpec) -> dict[str, torch.Tensor]:
    img_dir = DATA_DIR / spec.cfg.img_relpath
    # Load images into memory ------------------------------------------------------
    image_paths = list(img_dir.glob("*.jpg"))
    img_dict = spec.load_images(image_paths)
    return img_dict


def get_all_transformed_images(
    dataset_name: str, backbone_name: str = "efficientnet_s"
) -> torch.Tensor:
    spec = get_dataset(dataset_name)
    img_dict = get_image_dict(spec)
    logger.info(f"Loaded {len(img_dict)} images into memory")
    full_df = spec.load_and_clean_data(DATA_DIR)
    transforms = get_transforms(is_train=False, backbone_name=backbone_name)
    images = _get_image_list(full_df, img_dict, img_col=spec.cfg.path_col)
    logger.info(f"Transforming {len(images)} images")
    transformed_images = transform_images(images, transforms)
    logger.info(f"Transformed images shape: {transformed_images.shape}")
    return transformed_images


def _get_image_list(
    df: pd.DataFrame, img_dict: dict[str, torch.Tensor], img_col: str = "image"
) -> list[torch.Tensor]:
    images = []
    for img_id in df[img_col]:
        if img_id in img_dict:
            images.append(img_dict[img_id])
        else:
            raise ValueError(f"Image ID {img_id} not found in img_dict")
    return images


def transform_images(images: list[torch.Tensor], transforms: nn.Module) -> torch.Tensor:
    transformed_images = torch.stack([transforms(img) for img in images])
    return transformed_images


@memory.cache
def get_features(
    dataset_name: str, backbone_name: str = "efficientnet_s"
) -> torch.Tensor:
    spec = get_dataset(dataset_name)
    transformed_images = get_all_transformed_images(dataset_name, backbone_name)
    model_info = ModelInfo(method="ensemble", backbone=backbone_name)
    dataset_info = get_dataset_info(dataset_name)
    training_info = TrainingInfo(dataset=dataset_info, model=model_info)
    logger.info("Loading model for feature extraction")
    model = get_models_multihead(
        training_info=training_info, spec=spec, iteration=0
    )  # Iteration doesn't matter - they're all frozen
    logger.info("Extracting features")
    all_features = batch_predict(transformed_images, model, batch_size=128)
    logger.info("All done!")
    return all_features

"""train_classifier.py

Unified training entry-point that supports **all** image-classification methods in the
project - including the Domain-Independent / Discriminative variants **and** the
ensemble pipeline - for any registered dataset.

It replaces the older `train_fitzpatrick17k.py` *and* `train_domain_independent.py`
while keeping backwards-compatible CLI arguments (see ``guaranteed_fair_ensemble.config.parse_args``).

Main flow
──────────
1. Parse CLI -> ``args``
2. Fix PRNG seeds for deterministic runs
3. Resolve dataset via the registry -> ``spec``
4. Prepare WandB (optional)
5. Read + clean dataframe, load images to RAM
6. Build dataset / dataloaders (single or ensemble)
7. Train ➜ predict ➜ compute metrics ➜ (optional) log to WandB
"""

import random
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import torch
import wandb
from lightning import LightningModule
from loguru import logger
from tqdm import tqdm

import guaranteed_fair_ensemble.config
import guaranteed_fair_ensemble.names
import guaranteed_fair_ensemble.predict
from guaranteed_fair_ensemble.backbone import get_backbone, get_model_for_method
from guaranteed_fair_ensemble.data.base import DatasetSpec
from guaranteed_fair_ensemble.data.registry import get_dataset
from guaranteed_fair_ensemble.data_models import TrainingInfo
from guaranteed_fair_ensemble.datasets import (
    create_data_loaders,
    create_dataset,
    create_ensemble_data_loaders,
    create_ensemble_dataset,
    get_training_mask,
    split_data,
)
from guaranteed_fair_ensemble.directories import DATA_DIR
from guaranteed_fair_ensemble.lit_model import get_lit_model_for_method
from guaranteed_fair_ensemble.torch_utils import reverse_one_hot
from guaranteed_fair_ensemble.trainer import setup_trainer, train_model
from guaranteed_fair_ensemble.transforms import get_transforms

# ────────────────────────────────────────────────────────────────────────────────
# Utilities
# ────────────────────────────────────────────────────────────────────────────────


def _set_reproducibility(seed: int) -> None:
    """Fix all relevant PRNG seeds for deterministic behaviour."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def generate_ensemble_predictions(
    lit_model: LightningModule,
    test_loader,
    device,
):
    """
    Generate predictions for all ensemble members in a single pass through the test dataset.

    Args:
        lit_model: The trained ensemble model
        test_loader: DataLoader for the test dataset
        device: Device to run inference on
        ensemble_members: Number of ensemble members
        num_heads_per_member: Number of heads per ensemble member

    Returns:
        tuple: (all_folds_predictions, all_targets, all_protected_attrs)
    """
    lit_model.eval()
    all_folds_predictions = []
    all_targets = []
    all_protected_attrs = []

    print("Generating predictions for all folds in a single pass...")
    with torch.no_grad():
        pred_model = lit_model.to(device)
        for batch in tqdm(test_loader, desc="Generating predictions"):
            x, y, _ = batch
            x = x.to(device)

            # Get outputs for all heads at once
            outputs = pred_model(x)

            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(outputs)

            # Store the predictions, targets, and protected attributes
            all_folds_predictions.append(probs.cpu())
            all_targets.append(y[:, 0].cpu())
            sensitive_attr = reverse_one_hot(y[:, 1:])
            all_protected_attrs.append(
                sensitive_attr.cpu()
            )  # Assuming first protected attribute is used

    # Concatenate all predictions and other data
    all_folds_predictions = torch.cat(all_folds_predictions).numpy()
    all_targets = torch.cat(all_targets).numpy()
    all_protected_attrs = torch.cat(all_protected_attrs).numpy()

    return all_folds_predictions, all_targets, all_protected_attrs


def save_fold_predictions(
    fold_idx,
    all_folds_predictions,
    all_targets,
    all_protected_attrs,
    fold_dir,
    num_heads_per_member,
    overwrite=False,
):
    """
    Extract and save predictions for a specific fold.

    Args:
        fold_idx: Index of the fold
        all_folds_predictions: NumPy array of predictions for all folds
        all_targets: NumPy array of target values
        all_protected_attrs: NumPy array of protected attributes
        fold_dir: Directory to save fold predictions
        num_heads_per_member: Number of heads per ensemble member
        overwrite: Whether to overwrite existing files

    Returns:
        DataFrame of predictions for this fold
    """
    prediction_path = fold_dir / "predictions_predictions.csv"
    if prediction_path.exists() and not overwrite:
        print(
            f"Prediction file {prediction_path} already exists. Skipping fold {fold_idx + 1}."
        )
        return None

    # Extract this fold's heads' predictions
    start_idx = fold_idx * num_heads_per_member
    end_idx = start_idx + num_heads_per_member
    fold_predictions = all_folds_predictions[:, start_idx:end_idx]

    # Create prediction dataframe
    inputs = {
        "prediction": fold_predictions[:, 0],  # First head is classification
        "true_label": all_targets,
        "protected_attr": all_protected_attrs,
    }

    # Add remaining heads
    if num_heads_per_member > 1:
        for i in range(1, num_heads_per_member):
            inputs[f"protected_pred_{i}"] = fold_predictions[:, i]

    pred_df = pd.DataFrame(inputs)

    # Save predictions
    pred_df.to_csv(prediction_path, index=False)
    print(f"Predictions saved to {prediction_path}")

    return pred_df


def _single_run(
    *,
    args,
    cfg,
    train_df,
    test_df,
    img_dict: dict[str, torch.Tensor],
    train_tfms,
    eval_tfms,
    data_dir: Path,
    results_dir: Path,
    device: torch.device,
    ensemble_members: int,
    iteration: int | None = None,
) -> None:
    """Run the classic train -> val -> test loop for *one* model."""
    num_heads = cfg.num_protected_classes + 1 if cfg.num_protected_classes > 2 else 2

    # Handle iteration subdirectory if provided
    if iteration is not None:
        results_dir = results_dir / f"iteration{iteration}"
        results_dir.mkdir(parents=True, exist_ok=True)

    pred_path = guaranteed_fair_ensemble.predict.get_pred_path(
        output_dir=results_dir, file_prefix=args.training_method
    )

    if pred_path.exists() and not args.overwrite:
        print(f"Prediction file {pred_path} already exists. Skipping training.")
        return

    print(
        f"Running single-model pipeline{' for iteration ' + str(iteration) if iteration is not None else ''} …",
        flush=True,
    )

    train_split, val_split = split_data(
        df=train_df,
        cfg=cfg,
        random_seed=args.seed if iteration is None else args.seed + iteration,
        test_size=args.val_size,
    )

    train_ds = create_dataset(
        df=train_split,
        img_dict=img_dict,
        cfg=cfg,
        transform_ops=train_tfms,
        rebalance=args.rebalance,
    )

    val_ds = create_dataset(
        df=val_split,
        img_dict=img_dict,
        cfg=cfg,
        transform_ops=eval_tfms,
        rebalance=args.rebalance,
    )

    train_loader, val_loader = create_data_loaders(
        train_dataset=train_ds,
        val_dataset=val_ds,
        batch_size=args.batch_size,
        rebalance=args.rebalance,
    )

    model = get_model_for_method(
        method=args.training_method,
        backbone_name=args.backbone,
        num_heads=num_heads
        if args.training_method != "erm_ensemble"
        else ensemble_members,
    )

    lit_model, ckpt_path = train_model(
        args=args,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        data_dir=data_dir,
        iteration=iteration,
    )
    print(f"Training completed. Best checkpoint: {ckpt_path}")

    # ── Test
    test_ds = create_dataset(
        df=test_df,
        img_dict=img_dict,
        cfg=cfg,
        transform_ops=eval_tfms,
        rebalance=False,
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    pred_df = guaranteed_fair_ensemble.predict.predict_and_save(
        model=lit_model,
        test_loader=test_loader,
        output_dir=results_dir,
        file_prefix=args.training_method,
        device=device,
    )

    metrics = guaranteed_fair_ensemble.predict.compute_and_save_metrics(
        predictions=np.vstack(pred_df["prediction"].values),
        targets=pred_df["true_label"].values,
        groups=pred_df["protected_attr"].values,
        output_dir=results_dir,
        file_prefix=args.training_method,
    )

    # If part of an iteration, prefix metrics with iteration number
    if iteration is not None:
        metrics = {f"iteration{iteration}_{k}": v for k, v in metrics.items()}
    wandb.log(metrics)
    wandb.finish()

    print(
        f"Single-model training and evaluation completed successfully{' for iteration ' + str(iteration) if iteration is not None else ''}!",
        flush=True,
    )


# ────────────────────────────────────────────────────────────────────────────────
# Main entry-point
# ────────────────────────────────────────────────────────────────────────────────


def _ensemble_run(
    *,
    training_info: TrainingInfo,
    spec: DatasetSpec,
    train_df,
    test_df,
    img_dict,
    data_dir,
    results_dir,
    train_tfms,
    eval_tfms,
    device,
    iteration: int = 0,
    overwrite: bool = False,
) -> None:
    """Run the ensemble training pipeline for one iteration."""
    iter_results_dir = results_dir / f"iteration{iteration}"
    iter_results_dir.mkdir(parents=True, exist_ok=True)

    # *** CRITICAL: Use the same stratified k-fold split for both approaches ***
    seed = training_info.seed + iteration
    train_mask = get_training_mask(
        training_info=training_info, train_df=train_df, seed=seed, spec=spec
    )

    # Calculate number of heads per ensemble member
    num_heads_per_member = (
        spec.cfg.num_protected_classes + 1 if spec.cfg.num_protected_classes > 2 else 2
    )

    # Create the multi-head ensemble dataset with consistent rebalancing
    train_ds = create_ensemble_dataset(
        df=train_df,
        img_dict=img_dict,
        cfg=spec.cfg,
        fold_mask=train_mask,
        transform_ops=train_tfms,
        rebalance=training_info.model.rebalanced,
    )

    # Create validation dataset (can use same approach as it's only for validation)
    val_ds = create_ensemble_dataset(
        df=train_df,
        img_dict=img_dict,
        cfg=spec.cfg,
        fold_mask=train_mask,
        transform_ops=eval_tfms,
        rebalance=False,  # No rebalancing for validation
    )

    # Create dataloaders with proper sampling
    train_loader = create_ensemble_data_loaders(
        dataset=train_ds,
        batch_size=training_info.batch_size,
        rebalance=training_info.model.rebalanced,
        is_val=False,
    )

    val_loader = create_ensemble_data_loaders(
        dataset=val_ds,
        batch_size=training_info.batch_size,
        rebalance=False,
        is_val=True,
    )

    # Create a backbone model with k * num_heads total outputs

    backbone = get_backbone(
        name=training_info.model.backbone,
        num_heads=training_info.model.ensemble_members * num_heads_per_member,
        freeze=True,
    )

    # Pass fold indices to the multi-head model
    lit_model = get_lit_model_for_method(
        "ensemble_multi_head",
        backbone,
        num_heads_per_member=num_heads_per_member,
        training_mask=train_mask,
        lr=training_info.model.learning_rate,
    )

    # Setup trainer
    trainer, checkpoint_cb = setup_trainer(
        training_info,
        lit_model,
        data_dir,
        fold_num=None,
        iteration=iteration,
        use_validation=True,
    )

    # Train model
    trainer.fit(lit_model, train_loader, val_loader)
    ckpt_path = checkpoint_cb.best_model_path
    print(f"Training completed. Best checkpoint: {ckpt_path}")

    # Test dataset for evaluation
    test_ds = create_dataset(
        df=test_df,
        img_dict=img_dict,
        cfg=spec.cfg,
        transform_ops=eval_tfms,
        rebalance=False,
    )

    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=training_info.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Generate predictions
    all_folds_predictions, all_targets, all_protected_attrs = (
        generate_ensemble_predictions(
            lit_model,
            test_loader,
            device,
        )
    )

    # Process each fold's predictions and compute metrics
    for fold_idx in range(training_info.model.ensemble_members):
        fold_dir = iter_results_dir / f"fold{fold_idx + 1}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        # Save predictions for this fold
        pred_df = save_fold_predictions(
            fold_idx,
            all_folds_predictions,
            all_targets,
            all_protected_attrs,
            fold_dir,
            num_heads_per_member,
            overwrite=overwrite,
        )

        if pred_df is not None:
            # Compute metrics
            metrics = guaranteed_fair_ensemble.predict.compute_and_save_metrics(
                predictions=all_folds_predictions[:, fold_idx * num_heads_per_member],
                targets=all_targets,
                groups=all_protected_attrs,
                output_dir=fold_dir,
                file_prefix="metrics",
            )

            prefix = f"iteration{iteration}_" if iteration is not None else ""
            wandb.log(
                {f"{prefix}fold{fold_idx + 1}_{k}": v for k, v in metrics.items()}
            )


def is_valid_checkpoint(checkpoint: Path, minimum_epoch: int = 4) -> bool:
    epoch_index = checkpoint.stem.find("epoch=") + 6
    epoch = int(checkpoint.stem[epoch_index : epoch_index + 2])
    if epoch < minimum_epoch:
        print(f"Checkpoint {checkpoint} is invalid (epoch < {minimum_epoch}).")
        return False
    return True


def has_checkpoint(checkpoint_dir: Path, iteration: int = 1) -> bool:
    checkpoints = list(checkpoint_dir.rglob("*.ckpt"))
    if not all(is_valid_checkpoint(checkpoint) for checkpoint in checkpoints):
        print(f"Invalid checkpoints found in {checkpoint_dir}.")
        return False
    if len(checkpoints) == 0:
        print(f"No checkpoints found in {checkpoint_dir}.")
        return False
    if (
        any(checkpoint.parent == checkpoint_dir for checkpoint in checkpoints)
        and iteration == 0
    ):
        return True
    if any(f"iteration{iteration}" in str(checkpoint) for checkpoint in checkpoints):
        print(f"Checkpoint for iteration {iteration} found in {checkpoint_dir}.")
        return True
    return False


def all_checkpoints_exist(
    checkpoint_dir: Path, iterations: int = 3, minimum_epoch: int = 2
) -> bool:
    checkpoints = list(checkpoint_dir.rglob("*.ckpt"))
    iteration_checkpoints = 0
    for checkpoint in checkpoints:
        epoch_index = checkpoint.stem.find("epoch=") + 6
        epoch = int(checkpoint.stem[epoch_index : epoch_index + 2])
        if epoch < minimum_epoch:
            return False
    if (
        any(checkpoint.parent == checkpoint_dir for checkpoint in checkpoints)
        and iterations == 1
    ):
        return True
    if iteration_checkpoints < iterations - 1:
        print(f"Missing {iterations - 1} iteration checkpoints in {checkpoint_dir}")
        return False
    return len(checkpoints) == iterations


class TrainingData(NamedTuple):
    images: list[torch.Tensor]  # TODO: Normalize size of images (ideally)
    labels: torch.Tensor
    indices: torch.Tensor


def prepare_data(
    df: pd.DataFrame, spec: DatasetSpec, img_dict: dict[str, torch.Tensor]
) -> TrainingData:
    protected_attributes = torch.tensor(
        df[spec.cfg.protected_col].values, dtype=torch.long
    )
    protected_attributes = (
        torch.nn.functional.one_hot(protected_attributes)
        if spec.cfg.num_protected_classes > 2
        else protected_attributes.unsqueeze(1)
    )
    labels = torch.tensor(df[spec.cfg.target_col].values, dtype=torch.long)
    combined_labels = torch.cat((labels.unsqueeze(1), protected_attributes), dim=1).to(
        dtype=torch.float32
    )
    indices = torch.tensor(df.index.values, dtype=torch.long)
    images = [img_dict[path] for path in df[spec.cfg.path_col].values]
    return TrainingData(images=images, labels=combined_labels, indices=indices)


def main() -> None:
    args = guaranteed_fair_ensemble.config.parse_args()

    training_info = guaranteed_fair_ensemble.config.args_to_info(args)
    print(f"{training_info=}")
    _set_reproducibility(training_info.seed)
    method_name = guaranteed_fair_ensemble.names.get_method_name_raw(
        model_info=training_info.model, val_size=training_info.val_size
    )
    iterations = training_info.iterations

    # dataset spec ----------------------------------------------------------------
    spec = get_dataset(training_info.dataset.name)

    # Device ----------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("medium")
    logger.info(f"Using device: {device}")

    # Paths & data ------------------------------------------------------
    data_dir = DATA_DIR
    results_dir = data_dir / training_info.dataset.name / method_name
    logger.debug(f"Method name: {method_name}")
    checkpoint_dir = data_dir / "checkpoints" / training_info.dataset.name / method_name
    if all(has_checkpoint(checkpoint_dir, iteration=i) for i in range(iterations)):
        logger.warning(
            "Checkpoints for all iterations already exist. Skipping training."
        )
        return

    if not results_dir.exists():
        results_dir.mkdir(parents=True, exist_ok=True)

    img_dir = data_dir / spec.cfg.img_relpath
    if not img_dir.exists():
        raise FileNotFoundError(f"Image directory {img_dir} not found")

    df_full = spec.load_and_clean_data(data_dir=data_dir)

    # Load images into memory ------------------------------------------------------
    image_paths = list(img_dir.glob("*.jpg"))
    img_dict = spec.load_images(image_paths)

    # Transforms ------------------------------------------------------------------
    train_tfms = get_transforms(
        is_train=True, backbone_name=training_info.model.backbone
    )
    eval_tfms = get_transforms(
        is_train=False, backbone_name=training_info.model.backbone
    )

    # Check if we're running iterations
    if iterations > 1:
        print(f"Running {iterations} iterations (skipping iteration 0)...")
        # Skip iteration 0 as requested
        start_iteration = 1
    else:
        # Original single-run behavior
        start_iteration = 0
        iterations = 1

    wandb_mode = "offline" if training_info.wandb.offline else "online"
    run_name = f"{args.dataset}-{method_name}"
    # Add iteration info to wandb run name if running iterations
    if iterations > 1 and start_iteration > 0:
        run_name = f"{run_name}-iteration{start_iteration}"

    wandb.init(
        project=training_info.wandb.project,
        entity=None,
        name=run_name,
        mode=wandb_mode,
        config=vars(args),
    )

    wandb.log({"dataset_size": len(df_full)})

    for current_iteration in range(start_iteration, iterations):
        if has_checkpoint(checkpoint_dir, iteration=current_iteration):
            print(
                f"Checkpoint for iteration {current_iteration} already exists. Skipping training."
            )
            continue

        iter_seed = training_info.seed + current_iteration
        print(f"\n{'=' * 40}")
        print(f"RUNNING ITERATION {current_iteration} WITH SEED {iter_seed}")
        print(f"{'=' * 40}\n")

        # Set seed for this iteration
        _set_reproducibility(iter_seed)

        # Train/Test split for this iteration -------------------------------------------------------------
        train_df, test_df = split_data(
            df=df_full,
            cfg=spec.cfg,
            test_size=training_info.test_size,
            random_seed=iter_seed,
        )

        # Run the appropriate pipeline
        if training_info.model.method == "ensemble":
            _ensemble_run(
                training_info=training_info,
                spec=spec,
                train_df=train_df,
                test_df=test_df,
                img_dict=img_dict,
                data_dir=data_dir,
                results_dir=results_dir,
                train_tfms=train_tfms,
                eval_tfms=eval_tfms,
                device=device,
                iteration=current_iteration,
                overwrite=args.overwrite,
            )
        else:
            _single_run(
                args=args,
                cfg=spec.cfg,
                train_df=train_df,
                test_df=test_df,
                img_dict=img_dict,
                train_tfms=train_tfms,
                eval_tfms=eval_tfms,
                data_dir=data_dir,
                results_dir=results_dir,
                device=device,
                ensemble_members=training_info.model.ensemble_members,
                iteration=current_iteration if iterations > 1 else None,
            )

        if current_iteration < iterations - 1:
            # Finish the current wandb run before starting the next iteration
            wandb.finish()

            # Start a new wandb run for the next iteration
            if current_iteration < iterations - 1:
                wandb_mode = "offline" if training_info.wandb.offline else "online"
                next_run_name = f"{training_info.dataset.name}-{method_name}"
                # Add iteration info
                next_run_name = f"{next_run_name}-iteration{current_iteration + 1}"

                wandb.init(
                    project=training_info.wandb.project,
                    entity=None,
                    name=next_run_name,
                    mode=wandb_mode,
                    config=vars(args),
                )

    logger.info(
        f"\nAll {'iterations' if iterations > 1 else 'runs'} completed successfully."
    )

    wandb.finish()


if __name__ == "__main__":
    main()

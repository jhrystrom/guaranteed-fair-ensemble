from pathlib import Path

import torch
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.profilers import SimpleProfiler

import guaranteed_fair_ensemble.config
from guaranteed_fair_ensemble.data_models import TrainingInfo
from guaranteed_fair_ensemble.lit_model import get_lit_model_for_method
from guaranteed_fair_ensemble.models.domain_independent_lit import (
    DomainIndependentLitModule,
)
from guaranteed_fair_ensemble.names import get_method_name_raw


def setup_trainer(
    training_info: TrainingInfo,
    model,  # noqa: ARG001
    data_dir: Path,
    fold_num: int | None = None,
    iteration: int = 0,
    use_validation: bool = True,
):
    """
    Set up and configure a PyTorch Lightning Trainer

    Args:
        training_info: Info about training
        model: Model to train
        data_dir: Base data directory
        fold_num: Optional fold number for K-fold training
        iteration: Optional iteration number for multiple runs
        use_validation: Flag to use validation set

    Returns:
        tuple of (trainer, checkpoint_callback)
    """
    # Setup folder name based on training method
    dataset_dir = training_info.dataset.name
    method_name = get_method_name_raw(
        training_info.model, val_size=training_info.dataset.val_size
    )

    # Create & clean checkpoint directory
    if iteration is not None:
        base_ckpt_dir = (
            data_dir
            / "checkpoints"
            / dataset_dir
            / method_name
            / f"iteration{iteration}"
        )
    else:
        base_ckpt_dir = data_dir / "checkpoints" / dataset_dir / method_name

    if fold_num is not None:
        ckpt_dir = base_ckpt_dir / f"fold{fold_num}"
    else:
        ckpt_dir = base_ckpt_dir

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    for file_path in ckpt_dir.glob("*.ckpt"):
        file_path.unlink()

    # Checkpoint callback
    checkpoint_cb = ModelCheckpoint(
        monitor="val_loss" if use_validation else "train_loss",
        mode="min",
        dirpath=ckpt_dir,
        filename="best-{epoch:02d}-{val_loss:.3f}"
        if use_validation
        else "best-{epoch:02d}-{train_loss:.3f}",
        save_top_k=1,
    )

    # Initialize WandB logger if not disabled
    logger = None
    # Set run name based on method, fold and iteration
    run_name = training_info.model.method

    # Add fold info if applicable
    if fold_num is not None:
        run_name = f"{run_name}_fold{fold_num}"

    # Add iteration info if applicable
    if iteration is not None:
        run_name = f"{run_name}_iteration{iteration}"

    logger = WandbLogger(
        project=training_info.wandb.project,
        entity=None,
        name=run_name,
        log_model=False,
        save_dir=str(data_dir / "wandb"),
    )

    # Log hyperparameters
    if logger is not None:
        # Create hyperparameters dict
        hparams = {
            "backbone": training_info.model.backbone,
            "batch_size": training_info.batch_size,
            "max_epochs": training_info.model.max_epochs,
            "scaling_factor": training_info.model.scaling_factor,
            "training_method": training_info.model.method,
            "random_seed": training_info.seed,
        }

        # Add iteration-specific info if applicable
        if iteration is not None:
            hparams["iteration"] = iteration
            hparams["iteration_seed"] = training_info.seed + iteration

            logger.log_hyperparams(hparams)

    # Configure trainer
    callbacks = [checkpoint_cb]
    trainer = Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        max_epochs=training_info.model.max_epochs,
        callbacks=callbacks,
        logger=logger,
        deterministic=True,
        log_every_n_steps=10,
        # profiler=SimpleProfiler(filename="profile_training", dirpath=Path()),
    )

    return trainer, checkpoint_cb


def train_model(
    args,
    model,
    train_loader,
    val_loader,
    data_dir: Path,
    fold_num: int | None = None,
    iteration: int | None = None,
):
    """
    Train a model with the specified configuration

    Args:
        args: Command line arguments
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        data_dir: Base data directory
        fold_num: Optional fold number for K-fold training
        iteration: Optional iteration number for multiple runs
        early_stopping: Flag to enable early stopping

    Returns:
        tuple of (trained_model, checkpoint_path)
    """
    # Create Lightning module for the selected method
    lit_model = get_lit_model_for_method(
        args.training_method,
        model,
        scaling=args.scaling_factor,
        lr=args.learning_rate,
    )

    # Use the iteration from args if passed, otherwise use the function parameter
    current_iteration = getattr(args, "current_iteration", iteration)

    # Setup trainer
    training_info = guaranteed_fair_ensemble.config.args_to_info(args)
    trainer, checkpoint_cb = setup_trainer(
        training_info,
        lit_model,
        data_dir,
        fold_num,
        iteration=current_iteration,
        use_validation=val_loader is not None,
    )

    # Train the model
    trainer.fit(lit_model, train_loader, val_dataloaders=val_loader)
    print(f"Best model validation loss: {checkpoint_cb.best_model_score:.3f}")

    # Return best checkpoint path
    best_path = checkpoint_cb.best_model_path
    print(f"Best checkpoint saved to: {best_path}")

    return lit_model, best_path


def load_model_from_checkpoint(checkpoint_path, model, method, device):
    """
    Load a trained model from checkpoint

    Args:
        checkpoint_path: Path to the checkpoint file
        model: Model instance to load weights into
        method: Training method used
        device: Device to load the model onto

    Returns:
        Loaded model
    """
    # Import the appropriate Lightning module class

    # Get the appropriate Lightning module class based on method
    if method == "domain_independent":
        # For domain independent model, load the specific class
        lit_model = DomainIndependentLitModule.load_from_checkpoint(
            checkpoint_path,
            model=model,
            map_location=device,
        )
    else:
        # For other methods, use the factory function
        lit_cls = get_lit_model_for_method(method, model)
        lit_model = lit_cls.__class__.load_from_checkpoint(
            checkpoint_path,
            model=model,
            map_location=device,
        )

    # Move to device and set to evaluation mode
    lit_model.eval().to(device)

    return lit_model

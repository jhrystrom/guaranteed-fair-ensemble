import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from guaranteed_fair_ensemble.constants import DATASET_HPARAMS, DEFAULT_SEED
from guaranteed_fair_ensemble.data_models import (
    DatasetInfo,
    ModelInfo,
    TrainingInfo,
    WandbInfo,
)

load_dotenv(override=True)


def get_dataset_info(dataset_name: str) -> DatasetInfo:
    for dataset in DATASET_HPARAMS:
        if dataset.name != dataset_name:
            continue
        return dataset
    raise ValueError(f"Dataset '{dataset_name}' not found in registry.")


def args_to_info(args: argparse.Namespace) -> TrainingInfo:
    model_info = ModelInfo(
        method=args.training_method,
        backbone=args.backbone,
        scaling_factor=args.scaling_factor,
        rebalanced=args.rebalance,
        ensemble_members=args.ensemble_members,
        max_epochs=args.max_epochs,
    )
    data_info = get_dataset_info(args.dataset)
    wandb_info = WandbInfo(
        project=args.wandb_project,
    )
    return TrainingInfo(
        dataset=data_info,
        model=model_info,
        wandb=wandb_info,
        iterations=args.iterations,
        seed=args.seed,
        batch_size=args.batch_size,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Training on a custom image dataset with fairness constraints"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing results",
    )

    # Data paths
    parser.add_argument(
        "--dataset",
        type=str,
        default="fitzpatrick17k",
        choices=[ds.name for ds in DATASET_HPARAMS],
        help="Registered dataset to use (see datasets/ package)",
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path(os.getenv("DATA_DIR", ".")),
        help="Root folder that contains the raw datasets",
    )
    # Model configuration
    parser.add_argument(
        "--learning_rate",
        "-lr",
        type=float,
        default=0.001,
        help="Learning rate for the optimizer",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="mobilenetv3",
        choices=[
            "mobilenetv3",
            "resnet18",
            "resnet50",
            "efficientnet",
            "efficientnet_s",
        ],
        help="Which ImageNet backbone to use",
    )
    parser.add_argument(
        "--scaling_factor",
        type=float,
        default=0.5,
        help="Scaling on the attribute-head loss",
    )

    # Training configuration
    parser.add_argument(
        "--training_method",
        type=str,
        default="standard",
        choices=[
            "standard",
            "rebalance",
            "domain_independent",
            "domain_discriminative",
            "erm",
            "fairret",
            "ensemble",
        ],
        help="Training method to use",
    )
    parser.add_argument(
        "--ensemble-members",
        type=int,
        default=21,
        help="Number of ensemble members (only used if training_method is 'ensemble')",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
        help="Batch size for all loaders",
    )
    parser.add_argument(
        "--limit_train_batches",
        type=float,
        default=1.0,
        help="How many training batches per epoch (useful for prototyping)",
    )
    parser.add_argument(
        "--limit_val_batches",
        type=float,
        default=0.0,
        help="How many validation batches per epoch (0.0 = full)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Number of train/test iterations to run",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=20,
        help="Number of epochs per fold",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Proportion of data to use for testing",
    )
    parser.add_argument(
        "--val_size",
        type=float,
        default=0.2,
        help="Proportion of data to use for validation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--rebalance",
        action="store_true",
        help="Use rebalancing during training",
    )

    # WandB related arguments
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="fairness-image-classification",
        help="Name of the WandB project to log to",
    )
    # Fairness
    parser.add_argument(
        "--fairness_metric", type=str, choices=["min_recall", "equal_opportunity"]
    )

    return parser.parse_args()

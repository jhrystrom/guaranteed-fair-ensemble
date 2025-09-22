from collections.abc import Hashable

# get number of cpus
from multiprocessing import cpu_count

import lightning as L
import numpy as np
import pandas as pd
import polars as pl
import torch
from loguru import logger
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from torch.utils.data import DataLoader, Dataset

from guaranteed_fair_ensemble.data.base import DatasetConfig, DatasetSpec
from guaranteed_fair_ensemble.data_models import MinimalData, TrainingInfo
from guaranteed_fair_ensemble.torch_utils import custom_one_hot


def construct_minimal_data(
    spec: DatasetSpec, all_features: torch.Tensor, dataframe: pd.DataFrame
) -> MinimalData:
    _groups = dataframe[spec.cfg.protected_col].to_numpy()
    _labels = dataframe[spec.cfg.target_col].to_numpy().astype(np.int8)
    _features = all_features[dataframe.index.to_numpy()]
    return MinimalData(features=_features, groups=_groups, labels=_labels)


def get_training_mask(
    training_info: TrainingInfo,
    train_df: pd.DataFrame,
    seed: int,
    spec: DatasetSpec,
    debug: bool = False,
) -> torch.Tensor:
    polars_df = pl.DataFrame(train_df).with_columns(
        (
            pl.col(spec.cfg.target_col).cast(pl.Utf8)
            + "_"
            + pl.col(spec.cfg.protected_col).cast(pl.Utf8)
        ).alias("stratify_col")
    )
    if debug:
        smallest_group = (
            polars_df["stratify_col"].value_counts().sort("count")["stratify_col"][0]
        )
        mini_group_vec = (polars_df["stratify_col"] == smallest_group).to_numpy()

    splitter = StratifiedShuffleSplit(
        n_splits=training_info.model.ensemble_members,
        random_state=seed,
        test_size=training_info.val_size,
    )
    data_mask = np.zeros(
        shape=(polars_df.height, training_info.model.ensemble_members), dtype=np.bool
    )
    for classifier_idx, (train_idx, _) in enumerate(
        splitter.split(polars_df, polars_df["stratify_col"])
    ):
        data_mask[train_idx, classifier_idx] = True

    if debug:
        vec_sums = (data_mask & mini_group_vec[:, None]).sum(axis=0)
        logger.debug(f"Minimum group size: {vec_sums.min()}")

    return torch.tensor(data_mask, dtype=torch.bool)


def get_image_item(
    idx: int,
    df: pd.DataFrame,
    img_dict: dict,
    path_col: str,
    target_col: str,
    protected_col: str,
    num_protected_classes: int,
    transform_ops=None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Standalone version of __getitem__ for profiling"""
    row = df.iloc[idx]
    img_key = row[path_col]

    # Get the pre-processed image directly from the dictionary
    image = img_dict[img_key]
    if transform_ops is not None:
        try:
            image = transform_ops(image)
        except OSError as e:
            print(f"Error applying transforms to image {img_key}: {e}")
            raise

    y_target = torch.tensor(float(row[target_col]))
    y_prot = custom_one_hot(
        torch.tensor(row[protected_col]),
        num_classes=num_protected_classes,
    )
    # labels of shape (batch_size, 1 + num_protected_classes)
    labels = torch.cat((y_target.unsqueeze(0), y_prot), dim=0)

    return image, labels, torch.tensor(idx)


class CustomImageDataset(Dataset):
    """Dataset class for loading images with both target and protected attributes"""

    def __init__(
        self,
        df: pd.DataFrame,
        img_dict: dict,  # Pre-loaded dictionary of images
        cfg: DatasetConfig,
        transform_ops=None,
    ):
        self.df = df.reset_index(drop=True)
        self.img_dict = img_dict
        self.path_col = cfg.path_col
        self.target_col = cfg.target_col
        self.protected_col = cfg.protected_col
        self.transform_ops = transform_ops
        self.cfg = cfg

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return get_image_item(
            idx,
            self.df,
            self.img_dict,
            self.path_col,
            self.target_col,
            self.protected_col,
            num_protected_classes=self.cfg.num_protected_classes,
            transform_ops=self.transform_ops,
        )


class RebalancedImageDataset(CustomImageDataset):
    """
    Dataset with rebalancing for fairness via weighted sampling

    Implements resampling method that upsamples the minority groups so that all
    subgroups (combinations of target label and protected attribute groups) appear
    during training with equal chances.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        img_dict: dict,
        cfg: DatasetConfig,
        transform_ops=None,
    ):
        super().__init__(df, img_dict, cfg, transform_ops)

        # Calculate sample weights for rebalancing
        self.weights = self._calculate_weights()

    def _calculate_weights(self) -> torch.Tensor:
        """
        Compute per-sample weights so every (target, protected_group) subgroup
        is sampled with equal probability.

        The method is completely agnostic to:
        * how many protected groups there are
        * what codes they use (they only need to be hashable)

        Returns
        -------
        torch.Tensor
            1-D float tensor whose length equals ``len(self.df)`` and whose
            elements sum to ``len(self.df)``.
        """
        # Fast, vectorised extraction of the two columns we care about
        targets: np.ndarray = self.df[self.target_col].astype(int).to_numpy()
        groups: np.ndarray = self.df[self.protected_col].to_numpy()

        # Build a flat index of (target, group) pairs and count their frequency
        subgroup_keys = list(zip(targets, groups))
        subgroup_counts: dict[tuple[int, Hashable], int] = (
            pd.Series(subgroup_keys).value_counts().to_dict()
        )

        # Inverse-frequency weight for each row
        inv_freq = np.fromiter(
            (1.0 / subgroup_counts[key] for key in subgroup_keys),
            dtype=float,
            count=len(self.df),
        )

        # Renormalise so weights.sum() == n_samples
        inv_freq *= len(inv_freq) / inv_freq.sum()

        return torch.as_tensor(inv_freq, dtype=torch.float32)


def split_data(
    df: pd.DataFrame, cfg: DatasetConfig, test_size: float = 0.2, random_seed: int = 4
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into train and test sets with stratification"""
    return train_test_split(
        df,
        stratify=df[cfg.target_col],
        test_size=test_size,
        random_state=random_seed,
    )


def create_dataset(
    df: pd.DataFrame,
    img_dict: dict,
    cfg: DatasetConfig,
    transform_ops=None,
    rebalance: bool = False,
) -> torch.utils.data.Dataset:
    """
    Create the appropriate dataset based on the training method

    Args:
        df: DataFrame with data
        img_dict: Dictionary mapping image IDs to tensors
        cfg: DatasetConfig object with configuration
        transform_ops: Torchvision transforms to apply
        rebalance: Whether to use rebalancing for fairness
        **kwargs: Additional arguments for specific dataset classes

    Returns:
        Dataset instance appropriate for the specified method
    """
    return (
        RebalancedImageDataset(df, img_dict, cfg, transform_ops)
        if rebalance
        else CustomImageDataset(df, img_dict, cfg, transform_ops)
    )


def create_data_loaders(
    train_dataset,
    val_dataset,
    batch_size: int | None = None,
    rebalance: bool = False,
) -> tuple[DataLoader, DataLoader | None]:
    """
    Create data loaders that adapt to dataset size

    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        batch_size: Batch size (will be adapted if None)
        rebalance: Whether to use rebalancing for fairness

    Returns:
        Tuple of (train_loader, val_loader)
    """
    dataset_size = len(train_dataset)

    # 2. Configure number of workers based on dataset size
    if dataset_size <= 500:
        num_workers = 0  # Use main process for small datasets
        persistent_workers = False
    else:
        # Scale workers based on dataset size and available CPUs
        from multiprocessing import cpu_count

        available_cpus = min(cpu_count() - 1, 32)

        num_workers = min(available_cpus, 4) if dataset_size <= 2000 else available_cpus

        persistent_workers = num_workers > 0

    # 3. Configure pin_memory based on CUDA availability
    pin_memory = torch.cuda.is_available()

    # 4. Set up the loader kwargs based on the above determinations
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "persistent_workers": persistent_workers,
        "pin_memory": pin_memory,
    }

    # 5. Create train loader with method-specific settings
    if rebalance:
        from torch.utils.data import WeightedRandomSampler

        sampler = WeightedRandomSampler(
            weights=train_dataset.weights,
            num_samples=len(train_dataset),
            replacement=True,
        )

        train_loader = DataLoader(
            train_dataset,
            sampler=sampler,
            **loader_kwargs,
        )
    else:
        train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)

    # 6. Validation loader
    val_loader = (
        DataLoader(val_dataset, shuffle=False, **loader_kwargs) if val_dataset else None
    )

    if val_loader is None:
        print("No validation loader created, as val_dataset is None.")

    print(
        f"Created data loaders with batch_size={batch_size}, "
        f"workers={num_workers}, persistent={persistent_workers}"
    )

    return train_loader, val_loader


# Add these classes and functions to your guaranteed_fair_ensemble/datasets.py file


class EnsembleMultiHeadDataset(CustomImageDataset):
    """
    Dataset class specifically for multi-head ensemble training.
    Preserves fold structures and applies consistent rebalancing across folds.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        img_dict: dict,
        cfg: DatasetConfig,
        fold_mask: torch.Tensor,  # (N, num_folds)
        transform_ops=None,
        rebalance: bool = False,
    ):
        super().__init__(df, img_dict, cfg, transform_ops)
        self.fold_mask = fold_mask
        self.rebalance = rebalance

        # If rebalancing is requested, compute weights
        if rebalance:
            self.weights = self._calculate_ensemble_weights()

    def _calculate_ensemble_weights(self) -> torch.Tensor:
        """
        Calculate rebalancing weights for multi-head ensemble training.

        This ensures that the weights are consistent with what would be used
        in the standard ensemble approach where each fold has its own weights.
        """
        # Initialize weights tensor with zeros
        weights = torch.zeros(len(self.df), dtype=torch.float32)
        num_folds = self.fold_mask.shape[1]

        # For each fold, calculate the weights as they would be in standard ensemble
        for fold_idx in range(num_folds):
            train_idx = self.fold_mask[:, fold_idx].nonzero(as_tuple=True)[0]
            # Extract the fold's training data
            fold_df = self.df.iloc[train_idx]

            # Create a temporary RebalancedImageDataset to calculate weights
            temp_ds = RebalancedImageDataset(
                df=fold_df,
                img_dict=self.img_dict,
                cfg=self.cfg,
                transform_ops=self.transform_ops,
            )

            # Assign computed weights to the corresponding indices in our weights tensor
            for i, idx in enumerate(train_idx):
                weights[idx] = temp_ds.weights[i]

        return weights


def create_ensemble_dataset(
    df: pd.DataFrame,
    img_dict: dict,
    cfg: DatasetConfig,
    fold_mask: torch.Tensor,
    transform_ops=None,
    rebalance: bool = False,
) -> torch.utils.data.Dataset:
    """
    Create a dataset for ensemble training.

    Args:
        df: DataFrame with data
        img_dict: Dictionary mapping image IDs to tensors
        cfg: DatasetConfig object with configuration
        fold_mask: (n, num_classifiers) mask for training
        transform_ops: Torchvision transforms to apply
        rebalance: Whether to use rebalancing for fairness
        use_multi_head: Whether to use the multi-head ensemble approach

    Returns:
        Dataset instance appropriate for the specified method
    """
    return EnsembleMultiHeadDataset(
        df=df,
        img_dict=img_dict,
        cfg=cfg,
        fold_mask=fold_mask,
        transform_ops=transform_ops,
        rebalance=rebalance,
    )


def create_ensemble_data_loaders(
    dataset,
    batch_size: int,
    rebalance: bool = False,
    is_val: bool = False,
) -> torch.utils.data.DataLoader:
    """
    Create a data loader for ensemble training with consistent shuffle settings.

    Args:
        dataset: Dataset instance (either standard or multi-head)
        batch_size: Batch size
        rebalance: Whether to use rebalancing for fairness
        is_val: Whether this is a validation dataloader

    Returns:
        DataLoader configured for the dataset type
    """
    # Common loader parameters - matching the original implementation
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": min(cpu_count() - 1, 32),
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": min(cpu_count() - 1, 4) > 0,
        "prefetch_factor": 4,
    }

    # For validation data, always use shuffle=False as in the original
    if is_val:
        return torch.utils.data.DataLoader(
            dataset,
            shuffle=False,  # Validation data is never shuffled
            **loader_kwargs,
        )

    # For training data with rebalancing
    if rebalance:
        from torch.utils.data import WeightedRandomSampler

        # Multi-head dataset has its own weights
        if isinstance(dataset, EnsembleMultiHeadDataset) or hasattr(dataset, "weights"):
            sampler = WeightedRandomSampler(
                weights=dataset.weights,
                num_samples=len(dataset),
                replacement=True,
            )
        else:
            raise ValueError("Rebalancing requested but dataset has no weights")

        return torch.utils.data.DataLoader(
            dataset,
            sampler=sampler,  # Use sampler instead of shuffle when rebalancing
            **loader_kwargs,
        )

    # For training data without rebalancing, use shuffle=True
    return torch.utils.data.DataLoader(
        dataset,
        shuffle=True,  # Training data is shuffled when not using a sampler
        **loader_kwargs,
    )

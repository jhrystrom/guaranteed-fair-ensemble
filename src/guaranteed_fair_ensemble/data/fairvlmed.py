from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import torch
from tqdm import tqdm

from guaranteed_fair_ensemble.data.base import DatasetConfig
from guaranteed_fair_ensemble.directories import DATA_DIR

__all__ = ["cfg", "load_and_clean_data", "load_images"]

cfg = DatasetConfig(
    name="fairvlmed",
    csv_relpath=Path("fairvlmed_summary.csv"),
    img_relpath=Path("fairvlmed_npz"),
    path_col="filename",
    target_col="glaucoma",
    protected_col="race",
    num_protected_classes=3,
)


def load_and_clean_data(
    data_dir: Path = DATA_DIR,
) -> pd.DataFrame:
    """
    Load and clean the dataset

    Args:
        csv_path: Path to the CSV file
        img_dir: Directory containing the images
        path_col: Column name for image paths
        target_col: Column name for target labels
        protected_col: Column name for protected attributes
        data_dir: Root data directory

    Returns:
        Cleaned DataFrame
    """
    output_path = data_dir / "fairvlmed_clean.csv"
    if output_path.exists():
        print(f"Loading existing cleaned data from {output_path}")
        return pd.read_csv(output_path)

    # load CSV
    df_full = pl.read_csv(DATA_DIR / cfg.csv_relpath)

    # verify columns exist
    assert cfg.path_col in df_full.columns, (
        f"Path column {cfg.path_col} not found in CSV"
    )
    assert cfg.target_col in df_full.columns, (
        f"Target column {cfg.target_col} not found in CSV"
    )
    assert cfg.protected_col in df_full.columns, (
        f"Protected column {cfg.protected_col} not found in CSV"
    )

    df_parsed = df_full.with_columns(
        pl.col(cfg.path_col).str.strip_suffix(".npz"),
        pl.col(cfg.target_col) == "yes",
        pl.col(cfg.protected_col).replace({"white": 0, "asian": 1, "black": 2}),
    ).select(
        cfg.path_col,
        cfg.target_col,
        cfg.protected_col,
    )

    # Get list of available images
    img_dir = data_dir / cfg.img_relpath
    assert img_dir.exists(), f"Image directory {img_dir} does not exist"
    image_paths = list(img_dir.glob("*.npz"))
    image_ids = {path.stem for path in image_paths}
    df_filtered = df_parsed.filter(pl.col(cfg.path_col).is_in(image_ids))

    df_filtered.group_by(cfg.protected_col, cfg.target_col).agg(pl.len()).sort(
        cfg.protected_col, cfg.target_col
    )

    # Save the cleaned data
    df_filtered.write_csv(output_path)

    print(f"Found {len(df_filtered)} images in the CSV")
    return df_filtered.to_pandas()


def load_images(paths: list[Path]):  # noqa: ARG001
    """
    Ignore the paths and read the npz
    """
    actual_paths = list((DATA_DIR / cfg.img_relpath).glob("*.npz"))
    return {path.stem: read_npz(path) for path in tqdm(actual_paths)}


def read_npz(path: Path) -> torch.Tensor:
    """
    Read an npz file into a tensor with 3 channels (by repeating the grayscale data)
    """
    data = np.load(path)["slo_fundus"]
    # Convert to torch tensor (single channel)
    tensor_data = torch.tensor(data)

    # Create a 3-channel tensor by stacking the same grayscale data 3 times
    # Shape changes from [H, W] to [3, H, W]
    three_channel = torch.stack([tensor_data, tensor_data, tensor_data], dim=0)

    return three_channel

from pathlib import Path

import pandas as pd
import polars as pl

from guaranteed_fair_ensemble.data.base import DatasetConfig
from guaranteed_fair_ensemble.directories import DATA_DIR
from guaranteed_fair_ensemble.image import load_images

__all__ = ["cfg", "load_and_clean_data", "load_images"]

cfg = DatasetConfig(
    name="papila",
    csv_relpath=Path("papila_combined.csv"),
    img_relpath=Path("FundusImages"),
    path_col="Path",
    target_col="Diagnosis",
    protected_col="Sex",
    num_protected_classes=2,
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
    output_path = data_dir / "papila_cleaned.csv"
    if output_path.exists():
        print(f"Loading existing cleaned data from {output_path}")
        return pd.read_csv(output_path)

    assert (DATA_DIR / cfg.csv_relpath).exists(), (
        f"CSV file {cfg.csv_relpath} does not exist"
    )
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

    df_full = df_full.with_columns(pl.col(cfg.path_col).str.strip_suffix(".jpg"))

    # Get list of available images
    img_dir = data_dir / cfg.img_relpath
    assert img_dir.exists(), f"Image directory {img_dir} does not exist"
    image_paths = list(img_dir.glob("*.jpg"))
    image_ids = {path.stem for path in image_paths}

    # Filter to only include rows with valid protected attributes and existing images
    df_filtered = df_full.filter(
        (pl.col(cfg.protected_col) != -1) & pl.col(cfg.path_col).is_in(image_ids)
    )

    # Save the cleaned data
    df_filtered.write_csv(output_path)

    print(f"Found {len(df_full)} images in the CSV")
    return df_filtered.to_pandas()

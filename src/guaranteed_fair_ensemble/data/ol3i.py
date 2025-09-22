import pickle
from pathlib import Path

import pandas as pd
import polars as pl
from torch import Tensor

from guaranteed_fair_ensemble.data.base import DatasetConfig
from guaranteed_fair_ensemble.directories import DATA_DIR
from guaranteed_fair_ensemble.image import load_images as original_load_images

__all__ = ["cfg", "load_and_clean_data", "load_images"]

cfg = DatasetConfig(
    name="ol3i",
    csv_relpath=Path("l3_clinical_data.csv"),
    img_relpath=Path("ol3i_images"),
    path_col="anon_id",
    target_col="label_1y",
    protected_col="sex",
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
    output_path = data_dir / "ol3i_cleaned.csv"
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

    df_parsed = df_full.select(
        cfg.target_col, cfg.path_col, cfg.protected_col
    ).with_columns((pl.col(cfg.protected_col) == "male").cast(pl.Int8))
    df_parsed.group_by(cfg.protected_col, cfg.target_col).agg(pl.len()).sort(
        cfg.protected_col, cfg.target_col
    )

    # Get list of available images
    img_dir = data_dir / cfg.img_relpath
    assert img_dir.exists(), f"Image directory {img_dir} does not exist"
    image_paths = list(img_dir.glob("*.jpg"))
    image_ids = {path.stem for path in image_paths}

    # Filter to only include rows with valid protected attributes and existing images
    df_filtered = df_parsed.filter(
        (pl.col(cfg.protected_col) != -1) & pl.col(cfg.path_col).is_in(image_ids)
    )
    df_parsed

    # Save the cleaned data
    df_filtered.write_csv(output_path)

    print(f"Found {len(df_filtered)} images in the CSV")
    return df_filtered.to_pandas()


def load_images(image_paths: list[Path]) -> dict[str, Tensor]:
    data_dir = image_paths[0].parent.parent
    pickle_path = data_dir / "ol3i_images.pkl"
    if pickle_path.exists():
        print(f"Loading existing images from {pickle_path}")
        return pickle.loads(pickle_path.read_bytes())
    images = original_load_images(image_paths=image_paths)
    pickle_path.write_bytes(pickle.dumps(images))
    return images

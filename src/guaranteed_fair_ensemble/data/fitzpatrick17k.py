from pathlib import Path

import pandas as pd

from guaranteed_fair_ensemble.data.base import DatasetConfig
from guaranteed_fair_ensemble.image import load_images

__all__ = ["cfg", "load_and_clean_data", "load_images"]

cfg = DatasetConfig(
    name="fitzpatrick17k",
    csv_relpath=Path("fitzpatrick17k.csv"),
    img_relpath=Path("Fitzpatrick17k/images"),
    path_col="md5hash",
    target_col="three_partition_label",
    protected_col="fitzpatrick_scale",
    num_protected_classes=3,
)


def load_and_clean_data(
    data_dir: Path,
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
    output_path = data_dir / "fitzpatrick17k_cleaned.csv"
    if output_path.exists():
        print(f"Loading existing cleaned data from {output_path}")
        return pd.read_csv(output_path)

    assert cfg.csv_relpath.exists(), f"CSV file {cfg.csv_relpath} does not exist"
    # load CSV
    df_full = pd.read_csv(cfg.csv_relpath)

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

    # Get list of available images
    img_dir = data_dir / cfg.img_relpath
    assert img_dir.exists(), f"Image directory {img_dir} does not exist"
    image_paths = list(img_dir.glob("*.jpg"))
    image_ids = {path.stem for path in image_paths}

    # Filter to only include rows with valid protected attributes and existing images
    df_full = df_full[
        (df_full[cfg.protected_col] != -1) & df_full[cfg.path_col].isin(image_ids)
    ]
    # Remap protected col based on 1-4, 5, 6
    df_full[cfg.protected_col] = df_full[cfg.protected_col].replace(
        {1: 0, 2: 0, 3: 0, 4: 0, 5: 1, 6: 2}
    )

    # Convert target to boolean/binary (assuming "malignant" as positive class)
    df_full[cfg.target_col] = df_full[cfg.target_col] == "malignant"

    # Save the cleaned data
    df_full.to_csv(output_path, index=False)

    print(f"Found {len(df_full)} images in the CSV")
    return df_full

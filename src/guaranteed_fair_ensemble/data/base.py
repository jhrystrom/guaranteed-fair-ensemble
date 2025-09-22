from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import pandas as pd
import torch


@dataclass
class DatasetConfig:
    name: str
    csv_relpath: Path
    img_relpath: Path
    path_col: str
    target_col: str
    protected_col: str
    num_protected_classes: int = 3


class DatasetSpec(Protocol):
    cfg: DatasetConfig

    def load_and_clean_data(self, data_dir: Path) -> pd.DataFrame: ...

    def load_images(self, image_paths: list[Path]) -> dict[str, torch.Tensor]: ...

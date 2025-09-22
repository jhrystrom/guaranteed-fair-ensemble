from pathlib import Path

import torch
import torchvision.io
from tqdm import tqdm


def read_image(path: Path) -> torch.Tensor:
    """Read an image file into a tensor"""
    return torchvision.io.decode_image(path)


def load_images(image_paths: list[Path]) -> dict:
    """Load all images into a dictionary"""
    return {path.stem: read_image(path) for path in tqdm(image_paths)}

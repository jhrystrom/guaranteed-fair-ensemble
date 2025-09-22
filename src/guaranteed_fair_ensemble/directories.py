import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def get_main_dir(name: str = "data", create_dir: bool = True) -> Path:
    path = Path(__file__).parent.parent.parent / name
    if not path.exists() and create_dir:
        path.mkdir()
    return path


DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
PLOT_DIR = get_main_dir(name="plots")
OUTPUT_DIR = get_main_dir(name="output")
CHECKPOINT_DIR = DATA_DIR / "checkpoints"

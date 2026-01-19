"""
Combinatorial launcher for train_classifier.py

For each (dataset, model_spec) pair the script builds a CLI argument list,
patches `sys.argv`, and calls train_classifier.main().
"""

import gc
import shlex
import sys
from pprint import pprint
from typing import Any, Final

from train_classifier import main as train_main

from guaranteed_fair_ensemble.constants import DATASET_HPARAMS

# ────────────────────────────────────────────────────────────────────────────────
# Sweep definition
# ────────────────────────────────────────────────────────────────────────────────


MODEL_SPECS: Final[list[dict[str, Any]]] = [
    # {
    #     "training_method": "erm_ensemble",
    # },
    # {
    #     "training_method": "erm",
    # },
    # {
    #     "training_method": "ensemble",
    # },
    # {
    #     "training_method": "domain_independent",
    # },
    # {
    #     "training_method": "domain_discriminative",
    # },
    # {
    #     "training_method": "fairret",
    #     "scaling_factor": 0.5,
    # },
    {
        "training_method": "fairret",
        "scaling_factor": 0.75,
    },
    # {
    #     "training_method": "fairret",
    #     "scaling_factor": 1.0,
    # },
    # {
    #     "training_method": "fairret",
    #     "scaling_factor": 1.25,
    # },
    # {
    #     "training_method": "fairret",
    #     "scaling_factor": 1.5,
    # },
]

ITERATIONS = [1, 3]

# Optional: arguments common to *all* runs (leave empty to skip)
GLOBAL_ARGS: Final[dict[str, Any]] = {
    "max_epochs": 50,
    "rebalance": True,
    "ensemble-members": 21,
    "backbone": "mobilenetv3",
    "learning_rate": 0.001,
    "overwrite": False,
}


# ────────────────────────────────────────────────────────────────────────────────
# Utilities
# ────────────────────────────────────────────────────────────────────────────────
def dict_to_cli(arg_dict: dict[str, Any]) -> list[str]:
    """Convert mapping → flat CLI list suitable for sys.argv[1:]."""
    cli: list[str] = []
    for key, value in arg_dict.items():
        flag = f"--{key}"
        if isinstance(value, bool):
            if value:  # include only if True
                cli.append(flag)
        else:
            cli += [flag, str(value)]
    return cli


def build_grid() -> list[dict[str, Any]]:
    """Cartesian product of DATASETS x MODEL_SPECS merged with GLOBAL_ARGS."""
    grid: list[dict[str, Any]] = []
    for dataset in DATASET_HPARAMS:
        for num_iterations in ITERATIONS:
            for model_cfg in MODEL_SPECS:
                dataset_dict = dataset.to_cmd
                exp_cfg = {**dataset_dict, **GLOBAL_ARGS, **model_cfg}
                exp_cfg["iterations"] = num_iterations
                grid.append(exp_cfg)
    # Remove duplicates
    grid = [dict(t) for t in {tuple(d.items()) for d in grid}]
    return grid


def run_experiments() -> None:
    """Iterate over the generated grid and invoke train_classifier.main()."""
    grid = build_grid()
    total = len(grid)
    pprint(grid)

    for idx, args in enumerate(grid, start=1):
        cli_args = dict_to_cli(args)
        pretty = " ".join(shlex.quote(tok) for tok in cli_args)

        print(f"\n===== Experiment {idx}/{total}: {pretty} =====", flush=True)

        # Backup & patch sys.argv -------------------------------------------------
        argv_backup = sys.argv.copy()
        sys.argv = ["train_classifier.py", *cli_args]

        try:
            train_main()
        except SystemExit as exc:
            # Re-raise on non-zero exit codes so CI fails loudly
            if exc.code not in (0, None):
                raise
        finally:
            # Restore pristine state for the next run
            sys.argv = argv_backup
            gc.collect()  # help Python free CPU-side tensors
            try:
                import torch

                torch.cuda.empty_cache()  # free GPU memory between runs
            except ModuleNotFoundError:
                pass


# ────────────────────────────────────────────────────────────────────────────────
# Entry-point
# ────────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_experiments()

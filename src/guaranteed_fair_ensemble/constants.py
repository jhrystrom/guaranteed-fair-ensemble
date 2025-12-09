from typing import Final, Literal

from guaranteed_fair_ensemble.data_models import DatasetInfo

# Constants
DEFAULT_SEED = 4


# Configs
DATASET_HPARAMS: Final[list[DatasetInfo]] = [
    DatasetInfo(name="fitzpatrick17k", val_size=0.33, test_size=0.25),
    DatasetInfo(name="ham10000", val_size=0.2, test_size=0.2),
    DatasetInfo(
        name="fairvlmed",
        val_size=0.1,
        test_size=0.1,
        fairness_metric="equal_opportunity",
    ),
]

SIMPLE_BASELINES = [
    "erm",
    "domain_independent",
    "domain_discriminative",
]
ALL_BASELINES = [*SIMPLE_BASELINES, "fairret", "ensemble", "oxonfair", "hpp_ensemble"]
ALL_METHODS = ["multiensemble", *ALL_BASELINES]
FAIRRET_SCALES = [0.5, 0.75, 1.0, 1.25, 1.5]
ITERATIONS = 3

PRETTY_METHOD_NAMES = {
    "domain_discriminative": "DomainDisc",
    "domain_independent": "DomainInd",
    "fairret": "fairret",
    "erm": "ERM",
    "ensemble": "Ensemble (ERM)",
    "oxonfair": "OxonFair",
    "hpp_ensemble": "Ensemble (HPP)",
}

if __name__ == "__main__":
    from dataclasses import asdict

    asdict(DATASET_HPARAMS[0])

# datasets/__init__.py
from importlib import import_module

from .base import DatasetSpec

_REGISTRY: dict[str, DatasetSpec] = {}


def _lazy_import(name: str) -> DatasetSpec:
    if name not in _REGISTRY:
        try:
            module = import_module(f"guaranteed_fair_ensemble.data.{name}")
        except ModuleNotFoundError as e:
            raise ValueError(f"Unknown dataset '{name}'") from e
        # the module itself is the spec - no classes to instantiate
        _REGISTRY[name] = module  # type: ignore
    return _REGISTRY[name]


def get_dataset(name: str) -> DatasetSpec:
    return _lazy_import(name)

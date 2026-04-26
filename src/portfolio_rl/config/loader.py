"""Load YAML configuration files into strict typed schemas."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from portfolio_rl.config.schemas import (
    DataConfig,
    EnvConfig,
    FeaturesConfig,
    UniverseConfig,
)


def load_yaml(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as config_file:
        loaded = yaml.safe_load(config_file)

    if not isinstance(loaded, dict):
        raise ValueError(f"Expected mapping in config file: {config_path}")
    return loaded


def load_universe_config(path: str | Path) -> UniverseConfig:
    return UniverseConfig.model_validate(load_yaml(path))


def load_data_config(path: str | Path) -> DataConfig:
    return DataConfig.model_validate(load_yaml(path))


def load_features_config(path: str | Path) -> FeaturesConfig:
    return FeaturesConfig.model_validate(load_yaml(path))


def load_env_config(path: str | Path) -> EnvConfig:
    return EnvConfig.model_validate(load_yaml(path))

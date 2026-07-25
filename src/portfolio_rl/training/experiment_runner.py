"""Deterministic experiment-matrix planning for Phase 3."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

from portfolio_rl.config.loader import (
    load_data_config,
    load_phase3_experiment_config,
    load_yaml,
)
from portfolio_rl.config.schemas import EnvConfig, TrainPPOConfig


@dataclass(frozen=True)
class ExperimentRunPlan:
    """One validated child run in an experiment matrix."""

    run_id: str
    seed: int
    total_timesteps: int
    overrides: dict[str, Any]


def expand_experiment_matrix(
    config_path: str | Path,
    *,
    root: str | Path = ".",
) -> list[ExperimentRunPlan]:
    """Load, expand, and validate an experiment config without executing runs."""
    root_path = Path(root)
    resolved_config_path = _resolve_path(root_path, config_path)
    experiment_config = load_phase3_experiment_config(resolved_config_path)

    data_config_path = _resolve_path(root_path, experiment_config.base_data_config)
    env_config_path = _resolve_path(root_path, experiment_config.base_env_config)
    train_config_path = _resolve_path(root_path, experiment_config.base_train_config)
    load_data_config(data_config_path)
    base_env = load_yaml(env_config_path)
    base_train = load_yaml(train_config_path)
    EnvConfig.model_validate(base_env)
    validated_base_train = TrainPPOConfig.model_validate(base_train)

    total_timesteps = (
        experiment_config.total_timesteps
        if experiment_config.total_timesteps is not None
        else validated_base_train.total_timesteps
    )
    override_keys = sorted(experiment_config.overrides)
    override_values = [experiment_config.overrides[key] for key in override_keys]

    plans: list[ExperimentRunPlan] = []
    run_ids: set[str] = set()
    for seed in experiment_config.seeds:
        combinations = product(*override_values) if override_values else [()]
        for values in combinations:
            overrides = dict(zip(override_keys, values, strict=True))
            _validate_child_config(
                base_env=base_env,
                base_train=base_train,
                seed=seed,
                total_timesteps=total_timesteps,
                overrides=overrides,
            )
            run_id = _run_id(
                prefix=experiment_config.run_id_prefix,
                seed=seed,
                total_timesteps=total_timesteps,
                overrides=overrides,
            )
            if run_id in run_ids:
                raise ValueError(f"duplicate experiment run id: {run_id}")
            run_ids.add(run_id)
            plans.append(
                ExperimentRunPlan(
                    run_id=run_id,
                    seed=seed,
                    total_timesteps=total_timesteps,
                    overrides=overrides,
                )
            )
    return plans


def _validate_child_config(
    *,
    base_env: dict[str, Any],
    base_train: dict[str, Any],
    seed: int,
    total_timesteps: int,
    overrides: dict[str, Any],
) -> None:
    env_config = deepcopy(base_env)
    train_config = deepcopy(base_train)
    train_config["seed"] = seed
    train_config["total_timesteps"] = total_timesteps

    for path, value in overrides.items():
        if path in {"seed", "total_timesteps"}:
            raise ValueError(
                f"override path {path!r} has a dedicated experiment config field"
            )
        if path.startswith("env."):
            _set_existing_path(env_config, path.removeprefix("env."), value)
        else:
            _set_existing_path(train_config, path, value)

    EnvConfig.model_validate(env_config)
    TrainPPOConfig.model_validate(train_config)


def _set_existing_path(config: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    if not path or any(not part for part in parts):
        raise ValueError(f"invalid override path: {path!r}")

    target: dict[str, Any] = config
    for part in parts[:-1]:
        nested = target.get(part)
        if not isinstance(nested, dict):
            raise ValueError(f"unknown override path: {path}")
        target = nested
    if parts[-1] not in target:
        raise ValueError(f"unknown override path: {path}")
    target[parts[-1]] = value


def _run_id(
    *,
    prefix: str,
    seed: int,
    total_timesteps: int,
    overrides: dict[str, Any],
) -> str:
    payload = {
        "seed": seed,
        "total_timesteps": total_timesteps,
        "overrides": overrides,
    }
    canonical_payload = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    digest = hashlib.sha256(canonical_payload.encode("utf-8")).hexdigest()[:10]
    return f"{prefix}_{digest}"


def _resolve_path(root: Path, path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate

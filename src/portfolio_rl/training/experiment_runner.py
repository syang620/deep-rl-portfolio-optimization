"""Deterministic experiment-matrix planning for Phase 3."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import subprocess
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
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


def write_experiment_matrix_plan(
    config_path: str | Path,
    *,
    root: str | Path = ".",
    output_root: str | Path = "artifacts/experiment_matrices",
    force: bool = False,
) -> dict[str, Path]:
    """Write an auditable matrix plan without executing training."""
    root_path = Path(root)
    resolved_config_path = _resolve_path(root_path, config_path)
    experiment_config = load_phase3_experiment_config(resolved_config_path)
    plans = expand_experiment_matrix(resolved_config_path, root=root_path)

    matrix_dir = (
        _resolve_path(root_path, output_root) / experiment_config.experiment_name
    )
    outputs = {
        "manifest": matrix_dir / "experiment_matrix_manifest.json",
        "runs": matrix_dir / "runs.csv",
        "summary": matrix_dir / "summary.md",
    }
    existing_outputs = [path for path in outputs.values() if path.exists()]
    if existing_outputs and not force:
        existing = ", ".join(str(path) for path in existing_outputs)
        raise FileExistsError(
            f"experiment matrix plan already exists: {existing}; pass force=True"
        )

    generated_at = datetime.now(UTC).isoformat()
    git_commit = _git_commit(root_path)
    base_config_paths = {
        "data": _resolve_path(root_path, experiment_config.base_data_config),
        "env": _resolve_path(root_path, experiment_config.base_env_config),
        "train": _resolve_path(root_path, experiment_config.base_train_config),
    }
    manifest = {
        "schema_version": 1,
        "experiment_name": experiment_config.experiment_name,
        "generated_at": generated_at,
        "git_commit": git_commit,
        "source_config": _config_reference(resolved_config_path, root_path),
        "base_configs": {
            name: _config_reference(path, root_path)
            for name, path in base_config_paths.items()
        },
        "run_count": len(plans),
        "runs": [_manifest_run(plan) for plan in plans],
    }
    override_keys = sorted({key for plan in plans for key in plan.overrides})

    matrix_dir.mkdir(parents=True, exist_ok=True)
    outputs["manifest"].write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    outputs["runs"].write_text(
        _runs_csv(plans, override_keys),
        encoding="utf-8",
    )
    outputs["summary"].write_text(
        _summary_markdown(
            experiment_name=experiment_config.experiment_name,
            generated_at=generated_at,
            git_commit=git_commit,
            source_config=_display_path(resolved_config_path, root_path),
            plans=plans,
            override_keys=override_keys,
        ),
        encoding="utf-8",
    )
    return outputs


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


def _manifest_run(plan: ExperimentRunPlan) -> dict[str, Any]:
    return {
        "run_id": plan.run_id,
        "seed": plan.seed,
        "total_timesteps": plan.total_timesteps,
        "status": "planned",
        "overrides": plan.overrides,
    }


def _config_reference(path: Path, root: Path) -> dict[str, str]:
    return {
        "path": _display_path(path, root),
        "sha256": _sha256_file(path),
    }


def _display_path(path: Path, root: Path) -> str:
    resolved_path = path.resolve()
    resolved_root = root.resolve()
    try:
        return resolved_path.relative_to(resolved_root).as_posix()
    except ValueError:
        return str(resolved_path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit(root: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _runs_csv(
    plans: list[ExperimentRunPlan],
    override_keys: list[str],
) -> str:
    output = io.StringIO(newline="")
    fieldnames = [
        "run_id",
        "seed",
        "total_timesteps",
        "status",
        *override_keys,
    ]
    writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for plan in plans:
        row = {
            "run_id": plan.run_id,
            "seed": plan.seed,
            "total_timesteps": plan.total_timesteps,
            "status": "planned",
        }
        row.update(
            {key: _format_value(plan.overrides[key]) for key in override_keys}
        )
        writer.writerow(row)
    return output.getvalue()


def _summary_markdown(
    *,
    experiment_name: str,
    generated_at: str,
    git_commit: str | None,
    source_config: str,
    plans: list[ExperimentRunPlan],
    override_keys: list[str],
) -> str:
    headers = [
        "run_id",
        "seed",
        "total_timesteps",
        "status",
        *override_keys,
    ]
    lines = [
        f"# Experiment Matrix: {experiment_name}",
        "",
        f"- Generated: {generated_at}",
        f"- Git commit: {git_commit or 'unavailable'}",
        f"- Source config: `{source_config}`",
        f"- Planned runs: {len(plans)}",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for plan in plans:
        values = [
            plan.run_id,
            str(plan.seed),
            str(plan.total_timesteps),
            "planned",
            *[_format_value(plan.overrides[key]) for key in override_keys],
        ]
        cells = " | ".join(value.replace("|", "\\|") for value in values)
        lines.append(f"| {cells} |")
    return "\n".join(lines) + "\n"


def _format_value(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _resolve_path(root: Path, path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate

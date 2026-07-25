"""Deterministic experiment-matrix planning for Phase 3."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import subprocess
import tempfile
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import product
from pathlib import Path
from typing import Any, Literal

import yaml

from portfolio_rl.config.loader import (
    load_data_config,
    load_phase3_experiment_config,
    load_yaml,
)
from portfolio_rl.config.schemas import (
    EnvConfig,
    Phase3ExperimentConfig,
    TrainPPOConfig,
)


@dataclass(frozen=True)
class ExperimentRunPlan:
    """One validated child run in an experiment matrix."""

    run_id: str
    seed: int
    total_timesteps: int
    overrides: dict[str, Any]


@dataclass(frozen=True)
class ExperimentRunResult:
    """Outcome of executing or safely skipping one planned run."""

    run_id: str
    status: Literal["completed", "skipped"]
    model_path: Path


@dataclass(frozen=True)
class ExperimentMatrixResult:
    """Outcome of one bounded sequential matrix execution."""

    results: tuple[ExperimentRunResult, ...]
    attempted_count: int
    completed_count: int
    skipped_count: int
    remaining_count: int


def execute_experiment_matrix(
    config_path: str | Path,
    *,
    max_runs: int,
    root: str | Path = ".",
    matrix_output_root: str | Path = "artifacts/experiment_matrices",
    experiment_output_root: str | Path = "artifacts/experiments",
) -> ExperimentMatrixResult:
    """Execute a bounded number of planned runs in persisted matrix order."""
    if isinstance(max_runs, bool) or max_runs <= 0:
        raise ValueError("max_runs must be a positive integer")

    root_path = Path(root)
    resolved_config_path = _resolve_path(root_path, config_path)
    experiment_config = load_phase3_experiment_config(resolved_config_path)
    matrix_dir = (
        _resolve_path(root_path, matrix_output_root)
        / experiment_config.experiment_name
    )
    manifest = _read_matrix_manifest(
        _matrix_output_paths(matrix_dir)["manifest"]
    )
    _verify_matrix_manifest(
        manifest=manifest,
        experiment_config=experiment_config,
        config_path=resolved_config_path,
        root=root_path,
    )
    plans = expand_experiment_matrix(resolved_config_path, root=root_path)
    for plan in plans:
        _verify_manifest_run(manifest, plan)

    runs = manifest.get("runs")
    if (
        not isinstance(runs, list)
        or len(runs) != len(plans)
        or manifest.get("run_count") != len(plans)
    ):
        raise ValueError("persisted matrix run inventory is incomplete")
    records = {
        record.get("run_id"): record
        for record in runs
        if isinstance(record, dict)
    }
    if len(records) != len(plans):
        raise ValueError("persisted matrix run inventory contains duplicates")

    blocked = [
        plan.run_id
        for plan in plans
        if records[plan.run_id].get("status") in {"running", "failed"}
    ]
    if blocked:
        raise RuntimeError(
            "matrix contains unresolved running or failed runs: "
            + ", ".join(blocked)
        )
    invalid_statuses = [
        plan.run_id
        for plan in plans
        if records[plan.run_id].get("status") not in {"planned", "completed"}
    ]
    if invalid_statuses:
        raise ValueError(
            "matrix contains unsupported run statuses: "
            + ", ".join(invalid_statuses)
        )

    results_by_run_id: dict[str, ExperimentRunResult] = {}
    for plan in plans:
        if records[plan.run_id].get("status") == "completed":
            results_by_run_id[plan.run_id] = execute_experiment_run(
                resolved_config_path,
                plan.run_id,
                root=root_path,
                matrix_output_root=matrix_output_root,
                experiment_output_root=experiment_output_root,
            )

    attempted_count = 0
    for plan in plans:
        status = records[plan.run_id].get("status")
        if status != "planned" or attempted_count >= max_runs:
            continue
        results_by_run_id[plan.run_id] = execute_experiment_run(
            resolved_config_path,
            plan.run_id,
            root=root_path,
            matrix_output_root=matrix_output_root,
            experiment_output_root=experiment_output_root,
        )
        attempted_count += 1

    results = tuple(
        results_by_run_id[plan.run_id]
        for plan in plans
        if plan.run_id in results_by_run_id
    )
    skipped_count = sum(result.status == "skipped" for result in results)
    return ExperimentMatrixResult(
        results=results,
        attempted_count=attempted_count,
        completed_count=skipped_count + attempted_count,
        skipped_count=skipped_count,
        remaining_count=sum(
            record.get("status") == "planned" for record in records.values()
        )
        - attempted_count,
    )


def execute_experiment_run(
    config_path: str | Path,
    run_id: str,
    *,
    root: str | Path = ".",
    matrix_output_root: str | Path = "artifacts/experiment_matrices",
    experiment_output_root: str | Path = "artifacts/experiments",
) -> ExperimentRunResult:
    """Execute exactly one persisted, validated experiment run."""
    root_path = Path(root)
    resolved_config_path = _resolve_path(root_path, config_path)
    experiment_config = load_phase3_experiment_config(resolved_config_path)
    matrix_dir = (
        _resolve_path(root_path, matrix_output_root)
        / experiment_config.experiment_name
    )
    outputs = _matrix_output_paths(matrix_dir)
    manifest = _read_matrix_manifest(outputs["manifest"])
    _verify_matrix_manifest(
        manifest=manifest,
        experiment_config=experiment_config,
        config_path=resolved_config_path,
        root=root_path,
    )

    plans = expand_experiment_matrix(resolved_config_path, root=root_path)
    selected_plan = next((plan for plan in plans if plan.run_id == run_id), None)
    if selected_plan is None:
        raise ValueError(f"run id is not present in experiment matrix: {run_id}")
    _verify_manifest_run(manifest, selected_plan)

    data_config_path = _resolve_path(root_path, experiment_config.base_data_config)
    base_env = load_yaml(
        _resolve_path(root_path, experiment_config.base_env_config)
    )
    base_train = load_yaml(
        _resolve_path(root_path, experiment_config.base_train_config)
    )
    env_config, train_config = _resolve_child_configs(
        base_env=base_env,
        base_train=base_train,
        seed=selected_plan.seed,
        total_timesteps=selected_plan.total_timesteps,
        overrides=selected_plan.overrides,
    )
    env_yaml = _yaml_text(env_config)
    train_yaml = _yaml_text(train_config)
    experiment_dir = (
        _resolve_path(root_path, experiment_output_root) / selected_plan.run_id
    )
    model_path = experiment_dir / "model.zip"

    if experiment_dir.exists():
        if _completed_run_matches(
            experiment_dir=experiment_dir,
            plan=selected_plan,
            root=root_path,
            data_config_path=data_config_path,
            env_yaml=env_yaml,
            train_yaml=train_yaml,
        ):
            _set_run_status(
                manifest,
                selected_plan.run_id,
                status="completed",
                model_path=_display_path(model_path, root_path),
                skipped_at=datetime.now(UTC).isoformat(),
            )
            _write_matrix_artifacts(manifest, outputs)
            return ExperimentRunResult(
                run_id=selected_plan.run_id,
                status="skipped",
                model_path=model_path,
            )
        raise FileExistsError(
            "experiment directory exists but is incomplete or conflicts with "
            f"the persisted plan: {experiment_dir}"
        )

    _set_run_status(
        manifest,
        selected_plan.run_id,
        status="running",
        started_at=datetime.now(UTC).isoformat(),
    )
    _write_matrix_artifacts(manifest, outputs)
    try:
        with tempfile.TemporaryDirectory(
            prefix=f"{selected_plan.run_id}_"
        ) as temp_dir:
            temp_path = Path(temp_dir)
            env_config_path = temp_path / "env.yaml"
            train_config_path = temp_path / "train_ppo.yaml"
            env_config_path.write_text(env_yaml, encoding="utf-8")
            train_config_path.write_text(train_yaml, encoding="utf-8")
            trained_model_path = _run_ppo_training(
                root=root_path,
                data_config_path=data_config_path,
                env_config_path=env_config_path,
                train_config_path=train_config_path,
                output_dir_override=experiment_dir,
                run_id=selected_plan.run_id,
            )
        complete_bundle = _completed_run_matches(
            experiment_dir=experiment_dir,
            plan=selected_plan,
            root=root_path,
            data_config_path=data_config_path,
            env_yaml=env_yaml,
            train_yaml=train_yaml,
        )
        if Path(trained_model_path) != model_path or not complete_bundle:
            raise RuntimeError(
                "training did not produce a complete matching artifact bundle: "
                f"{experiment_dir}"
            )
    except Exception as exc:
        _set_run_status(
            manifest,
            selected_plan.run_id,
            status="failed",
            failed_at=datetime.now(UTC).isoformat(),
            error_type=type(exc).__name__,
            error_message=str(exc),
        )
        _write_matrix_artifacts(manifest, outputs)
        raise

    _set_run_status(
        manifest,
        selected_plan.run_id,
        status="completed",
        completed_at=datetime.now(UTC).isoformat(),
        model_path=_display_path(model_path, root_path),
    )
    _write_matrix_artifacts(manifest, outputs)
    return ExperimentRunResult(
        run_id=selected_plan.run_id,
        status="completed",
        model_path=model_path,
    )


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
    outputs = _matrix_output_paths(matrix_dir)
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
    matrix_dir.mkdir(parents=True, exist_ok=True)
    _write_matrix_artifacts(manifest, outputs)
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
            _resolve_child_configs(
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


def _resolve_child_configs(
    *,
    base_env: dict[str, Any],
    base_train: dict[str, Any],
    seed: int,
    total_timesteps: int,
    overrides: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
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

    validated_env = EnvConfig.model_validate(env_config)
    validated_train = TrainPPOConfig.model_validate(train_config)
    return (
        validated_env.model_dump(mode="json"),
        validated_train.model_dump(mode="json"),
    )


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


def _matrix_output_paths(matrix_dir: Path) -> dict[str, Path]:
    return {
        "manifest": matrix_dir / "experiment_matrix_manifest.json",
        "runs": matrix_dir / "runs.csv",
        "summary": matrix_dir / "summary.md",
    }


def _read_matrix_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            f"persisted experiment matrix manifest not found: {path}"
        )
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"expected JSON object in matrix manifest: {path}")
    return loaded


def _verify_matrix_manifest(
    *,
    manifest: dict[str, Any],
    experiment_config: Phase3ExperimentConfig,
    config_path: Path,
    root: Path,
) -> None:
    expected_base_configs = {
        "data": _config_reference(
            _resolve_path(root, experiment_config.base_data_config),
            root,
        ),
        "env": _config_reference(
            _resolve_path(root, experiment_config.base_env_config),
            root,
        ),
        "train": _config_reference(
            _resolve_path(root, experiment_config.base_train_config),
            root,
        ),
    }
    expected = {
        "schema_version": 1,
        "experiment_name": experiment_config.experiment_name,
        "git_commit": _git_commit(root),
        "source_config": _config_reference(config_path, root),
        "base_configs": expected_base_configs,
    }
    mismatches = [
        key for key, value in expected.items() if manifest.get(key) != value
    ]
    if mismatches:
        raise ValueError(
            "persisted matrix does not match current code/config inputs: "
            + ", ".join(mismatches)
        )


def _verify_manifest_run(
    manifest: dict[str, Any],
    plan: ExperimentRunPlan,
) -> None:
    runs = manifest.get("runs")
    if not isinstance(runs, list):
        raise ValueError("matrix manifest runs must be a list")
    record = next(
        (
            candidate
            for candidate in runs
            if isinstance(candidate, dict)
            and candidate.get("run_id") == plan.run_id
        ),
        None,
    )
    expected = {
        "run_id": plan.run_id,
        "seed": plan.seed,
        "total_timesteps": plan.total_timesteps,
        "overrides": plan.overrides,
    }
    record_mismatch = record is None or any(
        record.get(key) != value for key, value in expected.items()
    )
    if record_mismatch:
        raise ValueError(
            f"persisted matrix run does not match current plan: {plan.run_id}"
        )


def _set_run_status(
    manifest: dict[str, Any],
    run_id: str,
    *,
    status: str,
    **metadata: Any,
) -> None:
    runs = manifest.get("runs")
    if not isinstance(runs, list):
        raise ValueError("matrix manifest runs must be a list")
    for record in runs:
        if isinstance(record, dict) and record.get("run_id") == run_id:
            if status == "running":
                for key in [
                    "completed_at",
                    "failed_at",
                    "skipped_at",
                    "model_path",
                    "error_type",
                    "error_message",
                ]:
                    record.pop(key, None)
            record["status"] = status
            record.update(metadata)
            return
    raise ValueError(f"run id is not present in matrix manifest: {run_id}")


def _write_matrix_artifacts(
    manifest: dict[str, Any],
    outputs: dict[str, Path],
) -> None:
    runs = manifest.get("runs")
    if not isinstance(runs, list) or not all(
        isinstance(record, dict) for record in runs
    ):
        raise ValueError("matrix manifest runs must contain JSON objects")
    run_records = list(runs)
    override_keys = sorted(
        {
            key
            for record in run_records
            for key in _mapping(record.get("overrides"))
        }
    )
    _write_text_atomic(
        outputs["manifest"],
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    )
    _write_text_atomic(
        outputs["runs"],
        _runs_csv(run_records, override_keys),
    )
    _write_text_atomic(
        outputs["summary"],
        _summary_markdown(manifest, run_records, override_keys),
    )


def _runs_csv(
    run_records: list[dict[str, Any]],
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
    for record in run_records:
        overrides = _mapping(record.get("overrides"))
        row = {
            "run_id": record.get("run_id"),
            "seed": record.get("seed"),
            "total_timesteps": record.get("total_timesteps"),
            "status": record.get("status"),
        }
        row.update(
            {key: _format_value(overrides.get(key)) for key in override_keys}
        )
        writer.writerow(row)
    return output.getvalue()


def _summary_markdown(
    manifest: dict[str, Any],
    run_records: list[dict[str, Any]],
    override_keys: list[str],
) -> str:
    headers = [
        "run_id",
        "seed",
        "total_timesteps",
        "status",
        *override_keys,
    ]
    source_config = _mapping(manifest.get("source_config")).get(
        "path", "unavailable"
    )
    lines = [
        f"# Experiment Matrix: {manifest.get('experiment_name')}",
        "",
        f"- Generated: {manifest.get('generated_at')}",
        f"- Git commit: {manifest.get('git_commit') or 'unavailable'}",
        f"- Source config: `{source_config}`",
        f"- Planned runs: {len(run_records)}",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for record in run_records:
        overrides = _mapping(record.get("overrides"))
        values = [
            str(record.get("run_id", "")),
            str(record.get("seed", "")),
            str(record.get("total_timesteps", "")),
            str(record.get("status", "")),
            *[_format_value(overrides.get(key)) for key in override_keys],
        ]
        cells = " | ".join(value.replace("|", "\\|") for value in values)
        lines.append(f"| {cells} |")
    return "\n".join(lines) + "\n"


def _format_value(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _write_text_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as temp_file:
            temp_file.write(content)
            temp_path = Path(temp_file.name)
        temp_path.replace(path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def _yaml_text(config: dict[str, Any]) -> str:
    return yaml.safe_dump(config, sort_keys=False)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _completed_run_matches(
    *,
    experiment_dir: Path,
    plan: ExperimentRunPlan,
    root: Path,
    data_config_path: Path,
    env_yaml: str,
    train_yaml: str,
) -> bool:
    required_paths = {
        "model": experiment_dir / "model.zip",
        "metrics": experiment_dir / "metrics_validation.json",
        "manifest": experiment_dir / "manifest.json",
        "data": experiment_dir / "config.yaml",
        "env": experiment_dir / "env.yaml",
        "train": experiment_dir / "train_ppo.yaml",
        "feature_spec": experiment_dir / "feature_spec_v1.json",
    }
    if not all(path.is_file() for path in required_paths.values()):
        return False
    try:
        run_manifest = json.loads(
            required_paths["manifest"].read_text(encoding="utf-8")
        )
    except (json.JSONDecodeError, OSError):
        return False
    if not isinstance(run_manifest, dict):
        return False

    feature_spec_path = root / "artifacts/feature_specs/feature_spec_v1.json"
    expected = {
        "run_id": plan.run_id,
        "git_commit": _git_commit(root),
        "seed": plan.seed,
        "total_timesteps": plan.total_timesteps,
        "data_config_hash": _sha256_file(data_config_path),
        "env_config_hash": _sha256_text(env_yaml),
        "train_config_hash": _sha256_text(train_yaml),
        "feature_spec_hash": _sha256_file(feature_spec_path),
    }
    if any(run_manifest.get(key) != value for key, value in expected.items()):
        return False
    copied_hashes = {
        "data_config_hash": _sha256_file(required_paths["data"]),
        "env_config_hash": _sha256_file(required_paths["env"]),
        "train_config_hash": _sha256_file(required_paths["train"]),
        "feature_spec_hash": _sha256_file(required_paths["feature_spec"]),
    }
    return all(
        run_manifest.get(key) == value for key, value in copied_hashes.items()
    )


def _run_ppo_training(**kwargs: Any) -> Path:
    from portfolio_rl.training.train_ppo import run_ppo_training

    return run_ppo_training(**kwargs)


def _resolve_path(root: Path, path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate

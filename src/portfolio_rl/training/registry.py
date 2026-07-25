"""Experiment registry utilities for Phase 3 model selection."""

from __future__ import annotations

import hashlib
import json
from math import isfinite
from pathlib import Path
from typing import Any

import pandas as pd

from portfolio_rl.config.loader import load_yaml


REGISTRY_COLUMNS = [
    "run_id",
    "experiment_name",
    "matrix_status",
    "created_at",
    "git_commit",
    "feature_version",
    "seed",
    "algorithm",
    "total_timesteps",
    "action_temperature",
    "learning_rate",
    "ent_coef",
    "n_steps",
    "batch_size",
    "metric_for_best_model",
    "validation_total_return",
    "validation_cagr",
    "validation_sharpe_ratio",
    "validation_max_drawdown",
    "validation_average_weekly_turnover",
    "validation_transaction_cost_drag",
    "best_validation_sharpe_ratio",
    "model_path",
    "best_model_path",
    "selection_checkpoint",
    "selection_model_path",
    "selection_validation_total_return",
    "selection_validation_sharpe_ratio",
    "selection_validation_max_drawdown",
    "selection_validation_average_weekly_turnover",
    "selection_validation_transaction_cost_drag",
    "manifest_path",
    "artifact_complete",
    "reproducible",
    "metrics_complete",
    "selection_eligible",
    "eligibility_issues",
]

REQUIRED_ARTIFACTS = [
    "model.zip",
    "best_model.zip",
    "metrics_validation.json",
    "best_metrics_validation.json",
    "manifest.json",
    "config.yaml",
    "env.yaml",
    "train_ppo.yaml",
    "feature_spec_v1.json",
    "validation_nav.parquet",
    "validation_weights.parquet",
    "validation_trades.parquet",
    "validation_costs.parquet",
]
REQUIRED_METRICS = [
    "total_return",
    "sharpe_ratio",
    "max_drawdown",
    "average_weekly_turnover",
    "transaction_cost_drag",
]
HASHED_ARTIFACTS = {
    "data_config_hash": "config.yaml",
    "env_config_hash": "env.yaml",
    "train_config_hash": "train_ppo.yaml",
    "feature_spec_hash": "feature_spec_v1.json",
}


def build_experiment_registry(
    experiment_root: str | Path = "artifacts/experiments",
    matrix_root: str | Path | None = "artifacts/experiment_matrices",
) -> pd.DataFrame:
    """Build a tabular inventory from experiment artifact bundles."""
    root = Path(experiment_root)
    matrix_records = _load_matrix_records(matrix_root)
    rows = [
        _experiment_row(manifest_path, matrix_records)
        for manifest_path in sorted(root.glob("*/manifest.json"))
    ]
    return pd.DataFrame(rows, columns=REGISTRY_COLUMNS)


def write_experiment_registry(
    *,
    experiment_root: str | Path = "artifacts/experiments",
    matrix_root: str | Path | None = "artifacts/experiment_matrices",
    output_prefix: str | Path = "artifacts/experiments/registry",
) -> dict[str, Path]:
    """Write registry CSV, Parquet, and Markdown artifacts."""
    registry = build_experiment_registry(experiment_root, matrix_root)
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)

    csv_path = prefix.with_suffix(".csv")
    parquet_path = prefix.with_suffix(".parquet")
    markdown_path = prefix.with_suffix(".md")
    registry.to_csv(csv_path, index=False)
    registry.to_parquet(parquet_path, index=False)
    markdown_path.write_text(_registry_markdown(registry), encoding="utf-8")
    return {
        "csv": csv_path,
        "parquet": parquet_path,
        "markdown": markdown_path,
    }


def _experiment_row(
    manifest_path: Path,
    matrix_records: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    run_dir = manifest_path.parent
    manifest = _read_json(manifest_path)
    run_id = str(manifest.get("run_id") or run_dir.name)
    matrix_record = matrix_records.get(run_id)
    env_config, env_issue = _read_yaml_artifact(run_dir / "env.yaml")
    train_config, train_issue = _read_yaml_artifact(
        run_dir / "train_ppo.yaml"
    )
    metrics, metrics_issue = _read_json_artifact(
        run_dir / "metrics_validation.json"
    )
    best_metrics, best_metrics_issue = _read_json_artifact(
        run_dir / "best_metrics_validation.json"
    )
    read_issues = [
        issue
        for issue in [
            env_issue,
            train_issue,
            metrics_issue,
            best_metrics_issue,
        ]
        if issue is not None
    ]
    eligibility = _eligibility(
        run_dir,
        manifest,
        metrics,
        best_metrics,
        matrix_record,
        read_issues,
    )

    ppo_config = _mapping(train_config.get("ppo"))
    evaluation_config = _mapping(train_config.get("evaluation"))
    row = {
        "run_id": run_id,
        "experiment_name": (
            matrix_record.get("experiment_name") if matrix_record else None
        ),
        "matrix_status": (
            matrix_record.get("status") if matrix_record else None
        ),
        "created_at": manifest.get("created_at"),
        "git_commit": manifest.get("git_commit"),
        "feature_version": manifest.get("feature_version"),
        "seed": manifest.get("seed"),
        "algorithm": manifest.get("algorithm"),
        "total_timesteps": manifest.get("total_timesteps"),
        "action_temperature": env_config.get("action_temperature"),
        "learning_rate": ppo_config.get("learning_rate"),
        "ent_coef": ppo_config.get("ent_coef"),
        "n_steps": ppo_config.get("n_steps"),
        "batch_size": ppo_config.get("batch_size"),
        "metric_for_best_model": evaluation_config.get("metric_for_best_model"),
        "validation_total_return": metrics.get("total_return"),
        "validation_cagr": metrics.get("cagr"),
        "validation_sharpe_ratio": metrics.get("sharpe_ratio"),
        "validation_max_drawdown": metrics.get("max_drawdown"),
        "validation_average_weekly_turnover": metrics.get(
            "average_weekly_turnover"
        ),
        "validation_transaction_cost_drag": metrics.get("transaction_cost_drag"),
        "best_validation_sharpe_ratio": best_metrics.get("sharpe_ratio"),
        "model_path": _path_if_exists(run_dir / "model.zip"),
        "best_model_path": _path_if_exists(run_dir / "best_model.zip"),
        "selection_checkpoint": "best_checkpoint",
        "selection_model_path": _path_if_exists(run_dir / "best_model.zip"),
        "selection_validation_total_return": best_metrics.get("total_return"),
        "selection_validation_sharpe_ratio": best_metrics.get("sharpe_ratio"),
        "selection_validation_max_drawdown": best_metrics.get("max_drawdown"),
        "selection_validation_average_weekly_turnover": best_metrics.get(
            "average_weekly_turnover"
        ),
        "selection_validation_transaction_cost_drag": best_metrics.get(
            "transaction_cost_drag"
        ),
        "manifest_path": str(manifest_path),
        **eligibility,
    }
    return row


def _load_matrix_records(
    matrix_root: str | Path | None,
) -> dict[str, dict[str, Any]]:
    if matrix_root is None:
        return {}
    root = Path(matrix_root)
    records: dict[str, dict[str, Any]] = {}
    for manifest_path in sorted(
        root.glob("*/experiment_matrix_manifest.json")
    ):
        manifest = _read_json(manifest_path)
        experiment_name = manifest.get("experiment_name")
        runs = manifest.get("runs")
        if not isinstance(experiment_name, str) or not experiment_name:
            raise ValueError(
                f"matrix manifest has no experiment_name: {manifest_path}"
            )
        if not isinstance(runs, list):
            raise ValueError(
                f"matrix manifest runs must be a list: {manifest_path}"
            )
        for run in runs:
            if not isinstance(run, dict) or not isinstance(
                run.get("run_id"), str
            ):
                raise ValueError(
                    f"matrix manifest contains an invalid run: {manifest_path}"
                )
            run_id = run["run_id"]
            if run_id in records:
                raise ValueError(
                    "run id appears in multiple matrix manifests: "
                    f"{run_id}"
                )
            records[run_id] = {
                "experiment_name": experiment_name,
                "status": run.get("status"),
                "git_commit": manifest.get("git_commit"),
            }
    return records


def _eligibility(
    run_dir: Path,
    manifest: dict[str, Any],
    metrics: dict[str, Any],
    best_metrics: dict[str, Any],
    matrix_record: dict[str, Any] | None,
    read_issues: list[str],
) -> dict[str, Any]:
    issues = list(read_issues)
    missing_artifacts = [
        name for name in REQUIRED_ARTIFACTS if not (run_dir / name).is_file()
    ]
    issues.extend(f"missing_artifact:{name}" for name in missing_artifacts)
    artifact_complete = not missing_artifacts

    run_commit = manifest.get("git_commit")
    if not isinstance(run_commit, str) or not run_commit:
        issues.append("missing_git_commit")
    if matrix_record is None:
        issues.append("missing_matrix_provenance")
        matrix_commit = None
    else:
        matrix_commit = matrix_record.get("git_commit")
        if matrix_record.get("status") != "completed":
            issues.append(
                f"matrix_status:{matrix_record.get('status') or 'missing'}"
            )
        if not isinstance(matrix_commit, str) or not matrix_commit:
            issues.append("missing_matrix_git_commit")
        elif run_commit != matrix_commit:
            issues.append("git_commit_mismatch")

    hash_issues = _artifact_hash_issues(run_dir, manifest)
    issues.extend(hash_issues)
    reproducible = (
        artifact_complete
        and matrix_record is not None
        and isinstance(run_commit, str)
        and bool(run_commit)
        and run_commit == matrix_commit
        and not hash_issues
    )

    invalid_metrics = [
        name
        for name in REQUIRED_METRICS
        if not _is_finite_number(metrics.get(name))
    ]
    invalid_best_metrics = [
        name
        for name in REQUIRED_METRICS
        if not _is_finite_number(best_metrics.get(name))
    ]
    issues.extend(f"invalid_metric:{name}" for name in invalid_metrics)
    issues.extend(
        f"invalid_best_metric:{name}" for name in invalid_best_metrics
    )
    metrics_complete = not invalid_metrics and not invalid_best_metrics
    issues.extend(_validation_nav_issues(run_dir / "validation_nav.parquet"))

    return {
        "artifact_complete": artifact_complete,
        "reproducible": reproducible,
        "metrics_complete": metrics_complete,
        "selection_eligible": not issues,
        "eligibility_issues": ";".join(issues),
    }


def _artifact_hash_issues(
    run_dir: Path,
    manifest: dict[str, Any],
) -> list[str]:
    issues = []
    for manifest_key, filename in HASHED_ARTIFACTS.items():
        path = run_dir / filename
        expected_hash = manifest.get(manifest_key)
        if not isinstance(expected_hash, str) or not expected_hash:
            issues.append(f"missing_manifest_hash:{manifest_key}")
        elif path.is_file() and _sha256_file(path) != expected_hash:
            issues.append(f"hash_mismatch:{filename}")
    return issues


def _validation_nav_issues(path: Path) -> list[str]:
    if not path.is_file():
        return []
    try:
        nav = pd.read_parquet(path)
    except Exception:
        return ["unreadable_artifact:validation_nav.parquet"]
    if "nav" not in nav.columns or nav.empty:
        return ["invalid_validation_nav"]
    values = pd.to_numeric(nav["nav"], errors="coerce")
    if not values.map(_is_finite_number).all() or (values <= 0.0).any():
        return ["invalid_validation_nav"]
    return []


def _is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and isfinite(float(value))
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"expected JSON object: {path}")
    return loaded


def _read_json_artifact(path: Path) -> tuple[dict[str, Any], str | None]:
    if not path.exists():
        return {}, None
    try:
        return _read_json(path), None
    except Exception:
        return {}, f"unreadable_artifact:{path.name}"


def _read_yaml_artifact(path: Path) -> tuple[dict[str, Any], str | None]:
    if not path.exists():
        return {}, None
    try:
        return load_yaml(path), None
    except Exception:
        return {}, f"unreadable_artifact:{path.name}"


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _path_if_exists(path: Path) -> str | None:
    return str(path) if path.exists() else None


def _registry_markdown(registry: pd.DataFrame) -> str:
    eligible_count = (
        int(registry["selection_eligible"].sum()) if not registry.empty else 0
    )
    lines = [
        "# Experiment Registry",
        "",
        f"Total runs: {len(registry)}",
        f"Selection eligible: {eligible_count}",
        f"Selection ineligible: {len(registry) - eligible_count}",
        "",
    ]
    if registry.empty:
        return "\n".join(lines) + "\n"

    visible_columns = [
        "run_id",
        "experiment_name",
        "matrix_status",
        "total_timesteps",
        "action_temperature",
        "ent_coef",
        "validation_sharpe_ratio",
        "validation_total_return",
        "validation_average_weekly_turnover",
        "best_validation_sharpe_ratio",
        "selection_checkpoint",
        "selection_validation_sharpe_ratio",
        "selection_validation_total_return",
        "selection_eligible",
        "eligibility_issues",
    ]
    lines.extend(_markdown_table(registry.loc[:, visible_columns]))
    return "\n".join(lines) + "\n"


def _markdown_table(frame: pd.DataFrame) -> list[str]:
    headers = list(frame.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in frame.itertuples(index=False):
        lines.append(
            "| "
            + " | ".join(_format_markdown_cell(value) for value in row)
            + " |"
        )
    return lines


def _format_markdown_cell(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)

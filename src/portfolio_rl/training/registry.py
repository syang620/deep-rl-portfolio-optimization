"""Experiment registry utilities for Phase 3 model selection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from portfolio_rl.config.loader import load_yaml


REGISTRY_COLUMNS = [
    "run_id",
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
    "manifest_path",
]


def build_experiment_registry(
    experiment_root: str | Path = "artifacts/experiments",
) -> pd.DataFrame:
    """Build a tabular inventory from experiment artifact bundles."""
    root = Path(experiment_root)
    rows = [
        _experiment_row(manifest_path)
        for manifest_path in sorted(root.glob("*/manifest.json"))
    ]
    return pd.DataFrame(rows, columns=REGISTRY_COLUMNS)


def write_experiment_registry(
    *,
    experiment_root: str | Path = "artifacts/experiments",
    output_prefix: str | Path = "artifacts/experiments/registry",
) -> dict[str, Path]:
    """Write registry CSV, Parquet, and Markdown artifacts."""
    registry = build_experiment_registry(experiment_root)
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


def _experiment_row(manifest_path: Path) -> dict[str, Any]:
    run_dir = manifest_path.parent
    manifest = _read_json(manifest_path)
    env_config = _read_yaml_if_exists(run_dir / "env.yaml")
    train_config = _read_yaml_if_exists(run_dir / "train_ppo.yaml")
    metrics = _read_json_if_exists(run_dir / "metrics_validation.json")
    best_metrics = _read_json_if_exists(run_dir / "best_metrics_validation.json")

    ppo_config = _mapping(train_config.get("ppo"))
    evaluation_config = _mapping(train_config.get("evaluation"))
    row = {
        "run_id": manifest.get("run_id") or run_dir.name,
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
        "manifest_path": str(manifest_path),
    }
    return row


def _read_json(path: Path) -> dict[str, Any]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"expected JSON object: {path}")
    return loaded


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return _read_json(path)


def _read_yaml_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return load_yaml(path)


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _path_if_exists(path: Path) -> str | None:
    return str(path) if path.exists() else None


def _registry_markdown(registry: pd.DataFrame) -> str:
    lines = [
        "# Experiment Registry",
        "",
        f"Total runs: {len(registry)}",
        "",
    ]
    if registry.empty:
        return "\n".join(lines) + "\n"

    visible_columns = [
        "run_id",
        "total_timesteps",
        "action_temperature",
        "ent_coef",
        "validation_sharpe_ratio",
        "validation_total_return",
        "validation_average_weekly_turnover",
        "best_validation_sharpe_ratio",
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

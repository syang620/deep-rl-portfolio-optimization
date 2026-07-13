from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from portfolio_rl.training.registry import (
    build_experiment_registry,
    write_experiment_registry,
)


def test_registry_loads_complete_experiment(tmp_path: Path) -> None:
    _write_experiment(tmp_path / "complete")

    registry = build_experiment_registry(tmp_path)
    row = registry.iloc[0]

    assert list(registry["run_id"]) == ["complete"]
    assert row["created_at"] == "2026-07-12T00:00:00+00:00"
    assert row["git_commit"] == "abc123"
    assert row["feature_version"] == "v1"
    assert row["seed"] == 42
    assert row["algorithm"] == "PPO"
    assert row["total_timesteps"] == 500000
    assert row["validation_sharpe_ratio"] == 1.25
    assert row["best_validation_sharpe_ratio"] == 1.40
    assert row["model_path"].endswith("complete/model.zip")
    assert row["best_model_path"].endswith("complete/best_model.zip")
    assert row["manifest_path"].endswith("complete/manifest.json")


def test_registry_handles_missing_best_metrics(tmp_path: Path) -> None:
    _write_experiment(tmp_path / "missing_best", include_best=False)

    registry = build_experiment_registry(tmp_path)
    row = registry.iloc[0]

    assert pd.isna(row["best_validation_sharpe_ratio"])
    assert pd.isna(row["best_model_path"])


def test_registry_extracts_env_and_train_config_values(tmp_path: Path) -> None:
    _write_experiment(tmp_path / "config_values")

    registry = build_experiment_registry(tmp_path)
    row = registry.iloc[0]

    assert row["action_temperature"] == 0.5
    assert row["learning_rate"] == 0.0003
    assert row["ent_coef"] == 0.01
    assert row["n_steps"] == 2080
    assert row["batch_size"] == 260
    assert row["metric_for_best_model"] == "sharpe_ratio"


def test_registry_exports_csv_parquet_markdown(tmp_path: Path) -> None:
    _write_experiment(tmp_path / "first", run_id="first")
    _write_experiment(tmp_path / "second", run_id="second", seed=7)

    outputs = write_experiment_registry(
        experiment_root=tmp_path,
        output_prefix=tmp_path / "registry",
    )

    csv_frame = pd.read_csv(outputs["csv"])
    parquet_frame = pd.read_parquet(outputs["parquet"])
    markdown = outputs["markdown"].read_text(encoding="utf-8")

    assert list(outputs) == ["csv", "parquet", "markdown"]
    assert list(csv_frame["run_id"]) == ["first", "second"]
    assert list(parquet_frame["seed"]) == [42, 7]
    assert "# Experiment Registry" in markdown
    assert "Total runs: 2" in markdown
    assert "| run_id | total_timesteps | action_temperature |" in markdown


def _write_experiment(
    run_dir: Path,
    *,
    run_id: str | None = None,
    seed: int = 42,
    include_best: bool = True,
) -> None:
    run_dir.mkdir(parents=True)
    run_id = run_id or run_dir.name

    _write_json(
        run_dir / "manifest.json",
        {
            "run_id": run_id,
            "created_at": "2026-07-12T00:00:00+00:00",
            "git_commit": "abc123",
            "feature_version": "v1",
            "seed": seed,
            "algorithm": "PPO",
            "total_timesteps": 500000,
        },
    )
    (run_dir / "env.yaml").write_text(
        "action_temperature: 0.5\n",
        encoding="utf-8",
    )
    (run_dir / "train_ppo.yaml").write_text(
        """
ppo:
  learning_rate: 0.0003
  ent_coef: 0.01
  n_steps: 2080
  batch_size: 260
evaluation:
  metric_for_best_model: sharpe_ratio
""".lstrip(),
        encoding="utf-8",
    )
    _write_json(
        run_dir / "metrics_validation.json",
        {
            "total_return": 0.20,
            "cagr": 0.18,
            "sharpe_ratio": 1.25,
            "max_drawdown": -0.08,
            "average_weekly_turnover": 0.12,
            "transaction_cost_drag": 0.01,
        },
    )
    (run_dir / "model.zip").write_text("model", encoding="utf-8")

    if include_best:
        _write_json(
            run_dir / "best_metrics_validation.json",
            {"sharpe_ratio": 1.40},
        )
        (run_dir / "best_model.zip").write_text("best", encoding="utf-8")


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")

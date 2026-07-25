from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

import scripts.list_experiments as list_experiments_script
from portfolio_rl.training.registry import (
    build_experiment_registry,
    write_experiment_registry,
)


def test_registry_loads_complete_experiment(tmp_path: Path) -> None:
    _write_experiment(tmp_path / "complete")

    registry = build_experiment_registry(tmp_path, matrix_root=None)
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
    assert row["selection_checkpoint"] == "best_checkpoint"
    assert row["selection_model_path"].endswith("complete/best_model.zip")
    assert row["selection_validation_total_return"] == 0.22
    assert row["selection_validation_sharpe_ratio"] == 1.40
    assert row["selection_validation_max_drawdown"] == -0.06
    assert row["selection_validation_average_weekly_turnover"] == 0.08
    assert row["selection_validation_transaction_cost_drag"] == 0.004
    assert row["manifest_path"].endswith("complete/manifest.json")
    assert bool(row["artifact_complete"]) is True
    assert bool(row["selection_eligible"]) is False
    assert "missing_matrix_provenance" in row["eligibility_issues"]


def test_registry_handles_missing_best_metrics(tmp_path: Path) -> None:
    _write_experiment(tmp_path / "missing_best", include_best=False)

    registry = build_experiment_registry(tmp_path, matrix_root=None)
    row = registry.iloc[0]

    assert pd.isna(row["best_validation_sharpe_ratio"])
    assert pd.isna(row["best_model_path"])
    assert pd.isna(row["selection_model_path"])
    assert bool(row["artifact_complete"]) is False
    assert bool(row["metrics_complete"]) is False
    assert bool(row["selection_eligible"]) is False
    assert "missing_artifact:best_model.zip" in row["eligibility_issues"]
    assert "missing_artifact:best_metrics_validation.json" in row[
        "eligibility_issues"
    ]


def test_registry_selects_final_endpoint_when_score_is_higher(
    tmp_path: Path,
) -> None:
    _write_experiment(
        tmp_path / "final_wins",
        final_sharpe=1.50,
        best_sharpe=1.40,
    )

    row = build_experiment_registry(
        tmp_path,
        matrix_root=None,
    ).iloc[0]

    assert row["selection_checkpoint"] == "final_endpoint"
    assert row["selection_model_path"].endswith("final_wins/model.zip")
    assert row["selection_validation_sharpe_ratio"] == 1.50
    assert row["selection_validation_average_weekly_turnover"] == 0.12


def test_registry_prefers_best_checkpoint_on_exact_tie(tmp_path: Path) -> None:
    _write_experiment(
        tmp_path / "tie",
        final_sharpe=1.40,
        best_sharpe=1.40,
    )

    row = build_experiment_registry(tmp_path, matrix_root=None).iloc[0]

    assert row["selection_checkpoint"] == "best_checkpoint"
    assert row["selection_model_path"].endswith("tie/best_model.zip")
    assert row["selection_validation_average_weekly_turnover"] == 0.08


def test_registry_uses_total_return_for_final_nav_selection(
    tmp_path: Path,
) -> None:
    _write_experiment(
        tmp_path / "final_nav",
        metric_for_best_model="final_nav",
        final_return=0.25,
        best_return=0.22,
    )

    row = build_experiment_registry(tmp_path, matrix_root=None).iloc[0]

    assert row["selection_checkpoint"] == "final_endpoint"
    assert row["selection_validation_total_return"] == 0.25


def test_registry_extracts_env_and_train_config_values(tmp_path: Path) -> None:
    _write_experiment(tmp_path / "config_values")

    registry = build_experiment_registry(tmp_path, matrix_root=None)
    row = registry.iloc[0]

    assert row["action_temperature"] == 0.5
    assert row["learning_rate"] == 0.0003
    assert row["ent_coef"] == 0.01
    assert row["n_steps"] == 2080
    assert row["batch_size"] == 260
    assert row["metric_for_best_model"] == "sharpe_ratio"


def test_registry_exports_csv_parquet_markdown(tmp_path: Path) -> None:
    experiment_root = tmp_path / "experiments"
    matrix_root = tmp_path / "matrices"
    _write_experiment(experiment_root / "first", run_id="first")
    _write_experiment(experiment_root / "second", run_id="second", seed=7)
    _write_matrix(matrix_root, ["first", "second"])

    outputs = write_experiment_registry(
        experiment_root=experiment_root,
        matrix_root=matrix_root,
        output_prefix=tmp_path / "registry",
    )

    csv_frame = pd.read_csv(outputs["csv"])
    parquet_frame = pd.read_parquet(outputs["parquet"])
    markdown = outputs["markdown"].read_text(encoding="utf-8")

    assert list(outputs) == ["csv", "parquet", "markdown"]
    assert list(csv_frame["run_id"]) == ["first", "second"]
    assert list(parquet_frame["seed"]) == [42, 7]
    assert list(parquet_frame["selection_eligible"]) == [True, True]
    assert "# Experiment Registry" in markdown
    assert "Total runs: 2" in markdown
    assert "Selection eligible: 2" in markdown
    assert "Selection ineligible: 0" in markdown
    assert "| run_id | experiment_name | matrix_status |" in markdown


def test_completed_matrix_run_is_selection_eligible(tmp_path: Path) -> None:
    experiment_root = tmp_path / "experiments"
    matrix_root = tmp_path / "matrices"
    _write_experiment(experiment_root / "candidate")
    _write_matrix(matrix_root, ["candidate"])

    registry = build_experiment_registry(experiment_root, matrix_root)
    row = registry.iloc[0]

    assert row["experiment_name"] == "test_matrix"
    assert row["matrix_status"] == "completed"
    assert bool(row["artifact_complete"]) is True
    assert bool(row["reproducible"]) is True
    assert bool(row["metrics_complete"]) is True
    assert bool(row["selection_eligible"]) is True
    assert row["eligibility_issues"] == ""


def test_legacy_run_without_commit_remains_visible_but_ineligible(
    tmp_path: Path,
) -> None:
    _write_experiment(tmp_path / "legacy", git_commit=None)

    registry = build_experiment_registry(tmp_path, matrix_root=None)
    row = registry.iloc[0]

    assert row["run_id"] == "legacy"
    assert bool(row["selection_eligible"]) is False
    assert "missing_git_commit" in row["eligibility_issues"]
    assert "missing_matrix_provenance" in row["eligibility_issues"]


@pytest.mark.parametrize("status", ["planned", "running", "failed"])
def test_noncompleted_matrix_status_is_ineligible(
    tmp_path: Path,
    status: str,
) -> None:
    experiment_root = tmp_path / "experiments"
    matrix_root = tmp_path / "matrices"
    _write_experiment(experiment_root / "candidate")
    _write_matrix(matrix_root, ["candidate"], status=status)

    row = build_experiment_registry(experiment_root, matrix_root).iloc[0]

    assert bool(row["selection_eligible"]) is False
    assert f"matrix_status:{status}" in row["eligibility_issues"]


def test_registry_reports_artifact_and_metric_integrity_issues(
    tmp_path: Path,
) -> None:
    experiment_root = tmp_path / "experiments"
    matrix_root = tmp_path / "matrices"
    run_ids = [
        "missing",
        "hash",
        "metric",
        "best_metric",
        "nav",
        "unreadable",
        "corrupt",
    ]
    for run_id in run_ids:
        _write_experiment(experiment_root / run_id)
    (experiment_root / "missing" / "validation_trades.parquet").unlink()
    (experiment_root / "hash" / "config.yaml").write_text(
        "changed: true\n",
        encoding="utf-8",
    )
    metric_path = experiment_root / "metric" / "metrics_validation.json"
    metrics = json.loads(metric_path.read_text(encoding="utf-8"))
    metrics["sharpe_ratio"] = float("nan")
    _write_json(metric_path, metrics)
    best_metric_path = (
        experiment_root / "best_metric" / "best_metrics_validation.json"
    )
    best_metrics = json.loads(best_metric_path.read_text(encoding="utf-8"))
    best_metrics["transaction_cost_drag"] = None
    _write_json(best_metric_path, best_metrics)
    pd.DataFrame({"nav": [1.0, -0.1]}).to_parquet(
        experiment_root / "nav" / "validation_nav.parquet",
        index=False,
    )
    (
        experiment_root / "unreadable" / "validation_nav.parquet"
    ).write_text("not parquet", encoding="utf-8")
    (
        experiment_root / "corrupt" / "metrics_validation.json"
    ).write_text("not json", encoding="utf-8")
    _write_matrix(matrix_root, run_ids)

    registry = build_experiment_registry(experiment_root, matrix_root)
    issues = registry.set_index("run_id")["eligibility_issues"].to_dict()

    assert "missing_artifact:validation_trades.parquet" in issues["missing"]
    assert "hash_mismatch:config.yaml" in issues["hash"]
    assert "invalid_metric:sharpe_ratio" in issues["metric"]
    assert (
        "invalid_best_metric:transaction_cost_drag"
        in issues["best_metric"]
    )
    assert "invalid_validation_nav" in issues["nav"]
    assert (
        "unreadable_artifact:validation_nav.parquet"
        in issues["unreadable"]
    )
    assert "unreadable_artifact:metrics_validation.json" in issues["corrupt"]
    assert not registry["selection_eligible"].any()


def test_registry_rejects_duplicate_matrix_run_ids(tmp_path: Path) -> None:
    experiment_root = tmp_path / "experiments"
    matrix_root = tmp_path / "matrices"
    _write_experiment(experiment_root / "duplicate")
    _write_matrix(
        matrix_root,
        ["duplicate"],
        experiment_name="first_matrix",
    )
    _write_matrix(
        matrix_root,
        ["duplicate"],
        experiment_name="second_matrix",
    )

    with pytest.raises(ValueError, match="multiple matrix manifests"):
        build_experiment_registry(experiment_root, matrix_root)


def test_registry_cli_passes_matrix_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    captured: dict[str, object] = {}
    output_path = tmp_path / "registry.csv"

    def fake_write(**kwargs):
        captured.update(kwargs)
        return {"csv": output_path}

    monkeypatch.setattr(
        list_experiments_script,
        "write_experiment_registry",
        fake_write,
    )
    list_experiments_script.main(
        [
            "--experiment-root",
            str(tmp_path / "experiments"),
            "--matrix-root",
            str(tmp_path / "matrices"),
            "--output",
            str(tmp_path / "registry"),
        ]
    )

    assert captured["experiment_root"] == tmp_path / "experiments"
    assert captured["matrix_root"] == tmp_path / "matrices"
    assert captured["output_prefix"] == tmp_path / "registry"
    assert f"csv: {output_path}" in capsys.readouterr().out


def _write_experiment(
    run_dir: Path,
    *,
    run_id: str | None = None,
    seed: int = 42,
    include_best: bool = True,
    git_commit: str | None = "abc123",
    metric_for_best_model: str = "sharpe_ratio",
    final_return: float = 0.20,
    final_sharpe: float = 1.25,
    best_return: float = 0.22,
    best_sharpe: float = 1.40,
) -> None:
    run_dir.mkdir(parents=True)
    run_id = run_id or run_dir.name

    (run_dir / "config.yaml").write_text("source: test\n", encoding="utf-8")
    (run_dir / "env.yaml").write_text(
        "action_temperature: 0.5\n",
        encoding="utf-8",
    )
    (run_dir / "train_ppo.yaml").write_text(
        f"""
ppo:
  learning_rate: 0.0003
  ent_coef: 0.01
  n_steps: 2080
  batch_size: 260
evaluation:
  metric_for_best_model: {metric_for_best_model}
""".lstrip(),
        encoding="utf-8",
    )
    _write_json(run_dir / "feature_spec_v1.json", {"feature_version": "v1"})
    _write_json(
        run_dir / "metrics_validation.json",
        {
            "total_return": final_return,
            "cagr": 0.18,
            "sharpe_ratio": final_sharpe,
            "max_drawdown": -0.08,
            "average_weekly_turnover": 0.12,
            "transaction_cost_drag": 0.01,
        },
    )
    (run_dir / "model.zip").write_text("model", encoding="utf-8")
    pd.DataFrame({"nav": [1.01, 1.02]}).to_parquet(
        run_dir / "validation_nav.parquet",
        index=False,
    )
    pd.DataFrame({"SPY": [0.5]}).to_parquet(
        run_dir / "validation_weights.parquet",
        index=False,
    )
    pd.DataFrame({"SPY": [0.01]}).to_parquet(
        run_dir / "validation_trades.parquet",
        index=False,
    )
    pd.DataFrame({"transaction_cost_fraction": [0.0001]}).to_parquet(
        run_dir / "validation_costs.parquet",
        index=False,
    )
    _write_json(
        run_dir / "manifest.json",
        {
            "run_id": run_id,
            "created_at": "2026-07-12T00:00:00+00:00",
            "git_commit": git_commit,
            "feature_version": "v1",
            "seed": seed,
            "algorithm": "PPO",
            "total_timesteps": 500000,
            "data_config_hash": _sha256(run_dir / "config.yaml"),
            "env_config_hash": _sha256(run_dir / "env.yaml"),
            "train_config_hash": _sha256(run_dir / "train_ppo.yaml"),
            "feature_spec_hash": _sha256(
                run_dir / "feature_spec_v1.json"
            ),
        },
    )

    if include_best:
        _write_json(
            run_dir / "best_metrics_validation.json",
            {
                "total_return": best_return,
                "sharpe_ratio": best_sharpe,
                "max_drawdown": -0.06,
                "average_weekly_turnover": 0.08,
                "transaction_cost_drag": 0.004,
            },
        )
        (run_dir / "best_model.zip").write_text("best", encoding="utf-8")


def _write_matrix(
    matrix_root: Path,
    run_ids: list[str],
    *,
    experiment_name: str = "test_matrix",
    status: str = "completed",
) -> None:
    matrix_dir = matrix_root / experiment_name
    matrix_dir.mkdir(parents=True)
    _write_json(
        matrix_dir / "experiment_matrix_manifest.json",
        {
            "schema_version": 1,
            "experiment_name": experiment_name,
            "git_commit": "abc123",
            "run_count": len(run_ids),
            "runs": [
                {"run_id": run_id, "status": status}
                for run_id in run_ids
            ],
        },
    )


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

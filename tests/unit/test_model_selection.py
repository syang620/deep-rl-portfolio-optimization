from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import scripts.summarize_experiment as summarize_experiment_script
from portfolio_rl.evaluation.model_selection import (
    aggregate_seed_stability,
    write_seed_stability_report,
)


def test_seed_stability_aggregates_configuration_across_seeds(
    tmp_path: Path,
) -> None:
    manifest_path = _write_matrix(
        tmp_path / "matrix.json",
        [
            _matrix_run("run_7", 7),
            _matrix_run("run_42", 42),
            _matrix_run("run_101", 101),
        ],
    )
    registry = _registry(
        [
            _registry_row("run_7", 1.0, total_return=0.10),
            _registry_row("run_42", 2.0, total_return=0.20),
            _registry_row("run_101", 3.0, total_return=0.30),
        ]
    )

    stability = aggregate_seed_stability(registry, manifest_path)
    row = stability.iloc[0]

    assert len(stability) == 1
    assert row["metric_source"] == "best_checkpoint"
    assert row["planned_seed_count"] == 3
    assert row["eligible_seed_count"] == 3
    assert row["ineligible_seed_count"] == 0
    assert row["coverage_ratio"] == 1.0
    assert row["planned_seeds"] == "[7,42,101]"
    assert row["eligible_seeds"] == "[7,42,101]"
    assert bool(row["ranking_ready"]) is True
    assert row["validation_sharpe_ratio_mean"] == 2.0
    assert row["validation_sharpe_ratio_median"] == 2.0
    assert row["validation_sharpe_ratio_std"] == 1.0
    assert row["validation_sharpe_ratio_min"] == 1.0
    assert row["validation_sharpe_ratio_max"] == 3.0
    assert row["validation_total_return_median"] == 0.20
    assert row["validation_total_return_std"] == pytest.approx(0.10)


def test_seed_stability_separates_override_configurations(
    tmp_path: Path,
) -> None:
    manifest_path = _write_matrix(
        tmp_path / "matrix.json",
        [
            _matrix_run(
                "low_temp",
                42,
                overrides={"env.action_temperature": 0.25},
            ),
            _matrix_run(
                "high_temp",
                42,
                overrides={"env.action_temperature": 1.0},
            ),
        ],
    )
    registry = _registry(
        [
            _registry_row("low_temp", 1.0),
            _registry_row("high_temp", 2.0),
        ]
    )

    stability = aggregate_seed_stability(registry, manifest_path)

    assert len(stability) == 2
    assert stability["configuration_id"].is_unique
    assert list(stability["overrides"]) == [
        '{"env.action_temperature":0.25}',
        '{"env.action_temperature":1.0}',
    ]
    assert stability["ranking_ready"].all()


def test_seed_stability_reports_missing_and_ineligible_seeds(
    tmp_path: Path,
) -> None:
    manifest_path = _write_matrix(
        tmp_path / "matrix.json",
        [
            _matrix_run("eligible", 7),
            _matrix_run("ineligible", 42),
            _matrix_run("missing", 101),
        ],
    )
    registry = _registry(
        [
            _registry_row("eligible", 1.0),
            _registry_row(
                "ineligible",
                2.0,
                eligible=False,
                issues="missing_git_commit",
            ),
        ]
    )

    row = aggregate_seed_stability(registry, manifest_path).iloc[0]
    issues = json.loads(row["ineligibility_issues"])

    assert row["planned_seed_count"] == 3
    assert row["eligible_seed_count"] == 1
    assert row["ineligible_seed_count"] == 2
    assert row["coverage_ratio"] == pytest.approx(1 / 3)
    assert bool(row["ranking_ready"]) is False
    assert json.loads(row["ineligible_run_ids"]) == [
        "ineligible",
        "missing",
    ]
    assert issues == {
        "ineligible": "missing_git_commit",
        "missing": "missing_registry_run",
    }


def test_one_seed_has_no_sample_dispersion(tmp_path: Path) -> None:
    manifest_path = _write_matrix(
        tmp_path / "matrix.json",
        [_matrix_run("smoke", 42)],
        experiment_name="smoke_matrix",
    )
    registry = _registry(
        [_registry_row("smoke", 1.25, experiment_name="smoke_matrix")]
    )

    row = aggregate_seed_stability(registry, manifest_path).iloc[0]

    assert bool(row["ranking_ready"]) is True
    assert pd.isna(row["validation_sharpe_ratio_std"])
    assert pd.isna(row["validation_total_return_std"])


def test_seed_stability_rejects_malformed_inputs(tmp_path: Path) -> None:
    manifest_path = _write_matrix(
        tmp_path / "matrix.json",
        [_matrix_run("candidate", 42)],
    )
    registry = _registry([_registry_row("candidate", 1.0)])

    with pytest.raises(ValueError, match="missing required columns"):
        aggregate_seed_stability(
            registry.drop(columns=["selection_eligible"]),
            manifest_path,
        )

    mismatched = registry.copy()
    mismatched.loc[0, "experiment_name"] = "other_matrix"
    with pytest.raises(ValueError, match="does not match"):
        aggregate_seed_stability(mismatched, manifest_path)

    unsupported_checkpoint = registry.copy()
    unsupported_checkpoint.loc[0, "selection_checkpoint"] = "final_endpoint"
    with pytest.raises(ValueError, match="unsupported selection checkpoint"):
        aggregate_seed_stability(unsupported_checkpoint, manifest_path)

    missing_model = registry.copy()
    missing_model.loc[0, "selection_model_path"] = ""
    with pytest.raises(ValueError, match="no selection model path"):
        aggregate_seed_stability(missing_model, manifest_path)

    duplicate_manifest = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )
    duplicate_manifest["runs"].append(duplicate_manifest["runs"][0])
    duplicate_manifest["run_count"] = 2
    manifest_path.write_text(
        json.dumps(duplicate_manifest),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate run_id"):
        aggregate_seed_stability(registry, manifest_path)


def test_seed_stability_writes_csv_markdown_and_cli_outputs(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest_path = _write_matrix(
        tmp_path / "matrix.json",
        [_matrix_run("candidate", 42)],
    )
    registry_path = tmp_path / "registry.csv"
    _registry([_registry_row("candidate", 1.0)]).to_csv(
        registry_path,
        index=False,
    )
    output_dir = tmp_path / "output"

    outputs = write_seed_stability_report(
        registry_path=registry_path,
        matrix_manifest_path=manifest_path,
        output_dir=output_dir,
    )
    markdown = outputs["markdown"].read_text(encoding="utf-8")

    assert list(outputs) == ["csv", "markdown"]
    assert pd.read_csv(outputs["csv"]).loc[0, "ranking_ready"]
    assert (
        "Validation-only best-checkpoint statistics; the test split was not "
        "accessed."
        in markdown
    )
    assert "Ranking ready: 1" in markdown

    cli_output_dir = tmp_path / "cli_output"
    summarize_experiment_script.main(
        [
            "--registry",
            str(registry_path),
            "--matrix-manifest",
            str(manifest_path),
            "--output-dir",
            str(cli_output_dir),
        ]
    )
    output = capsys.readouterr().out
    assert f"csv: {cli_output_dir / 'seed_stability.csv'}" in output
    assert f"markdown: {cli_output_dir / 'seed_stability.md'}" in output


def _registry(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _registry_row(
    run_id: str,
    sharpe: float,
    *,
    experiment_name: str = "test_matrix",
    eligible: bool = True,
    issues: str = "",
    total_return: float = 0.10,
) -> dict[str, object]:
    return {
        "run_id": run_id,
        "experiment_name": experiment_name,
        "selection_eligible": eligible,
        "eligibility_issues": issues,
        "selection_checkpoint": "best_checkpoint",
        "selection_model_path": f"artifacts/experiments/{run_id}/best_model.zip",
        "selection_validation_sharpe_ratio": sharpe,
        "selection_validation_total_return": total_return,
        "selection_validation_max_drawdown": -0.08,
        "selection_validation_average_weekly_turnover": 0.12,
        "selection_validation_transaction_cost_drag": 0.01,
        "validation_sharpe_ratio": 99.0,
        "validation_total_return": 99.0,
    }


def _matrix_run(
    run_id: str,
    seed: int,
    *,
    overrides: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "run_id": run_id,
        "seed": seed,
        "total_timesteps": 500000,
        "status": "completed",
        "overrides": overrides or {},
    }


def _write_matrix(
    path: Path,
    runs: list[dict[str, object]],
    *,
    experiment_name: str = "test_matrix",
) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "experiment_name": experiment_name,
                "run_count": len(runs),
                "runs": runs,
            }
        ),
        encoding="utf-8",
    )
    return path

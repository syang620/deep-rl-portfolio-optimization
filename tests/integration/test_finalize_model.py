from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from portfolio_rl.training.finalize_model import finalize_selected_model

EXPERIMENT = "phase3_candidate"
CONFIGURATION_ID = "abc123"
RUN_ID = "phase3_seed_42"
MODEL_VERSION = "ppo_v1_seed42"


def test_finalize_model_copies_required_artifacts(tmp_path: Path) -> None:
    inputs = _packaging_fixture(tmp_path)

    result = finalize_selected_model(
        **inputs,
        model_version=MODEL_VERSION,
        representative_seed=42,
        root=tmp_path,
    )

    assert result.output_dir == tmp_path / "artifacts/final_model" / MODEL_VERSION
    assert (result.output_dir / "model.zip").read_bytes() == b"checkpoint"
    selected_model = _read_json(result.output_dir / "selected_model.json")
    assert selected_model["representative_seed"] == 42
    assert selected_model["test_split_used"] is False
    assert selected_model["final_test_status"] == "not_run"

    manifest = _read_json(result.output_dir / "manifest.json")
    files = {record["path"]: record for record in manifest["files"]}
    assert "model.zip" in files
    assert "selected_model_card.md" in files
    assert files["model.zip"]["sha256"] == _sha256(result.output_dir / "model.zip")


def test_finalize_model_rejects_test_tainted_artifact(tmp_path: Path) -> None:
    inputs = _packaging_fixture(tmp_path)
    diagnostics_path = (
        tmp_path / "artifacts/diagnostics" / EXPERIMENT / "allocation_summary.json"
    )
    diagnostics = _read_json(diagnostics_path)
    diagnostics["test_split_used"] = True
    _write_json(diagnostics_path, diagnostics)

    with pytest.raises(ValueError, match="not test-free"):
        finalize_selected_model(
            **inputs,
            model_version=MODEL_VERSION,
            representative_seed=42,
            root=tmp_path,
        )

    assert not (tmp_path / "artifacts/final_model" / MODEL_VERSION).exists()


def test_finalize_model_honors_sensitivity_stop(tmp_path: Path) -> None:
    inputs = _packaging_fixture(tmp_path)
    sensitivity_path = (
        tmp_path / "artifacts/sensitivity" / EXPERIMENT / "sensitivity_manifest.json"
    )
    sensitivity = _read_json(sensitivity_path)
    sensitivity["campaign_interpretation"]["stop_before_packaging"] = True
    _write_json(sensitivity_path, sensitivity)

    with pytest.raises(ValueError, match="block final packaging"):
        finalize_selected_model(
            **inputs,
            model_version=MODEL_VERSION,
            representative_seed=42,
            root=tmp_path,
        )


def test_finalize_model_refuses_to_overwrite_version(tmp_path: Path) -> None:
    inputs = _packaging_fixture(tmp_path)
    destination = tmp_path / "artifacts/final_model" / MODEL_VERSION
    destination.mkdir(parents=True)

    with pytest.raises(FileExistsError, match="already exists"):
        finalize_selected_model(
            **inputs,
            model_version=MODEL_VERSION,
            representative_seed=42,
            root=tmp_path,
        )


def _packaging_fixture(tmp_path: Path) -> dict[str, Path]:
    artifact_root = tmp_path / "artifacts"
    run_dir = artifact_root / "experiments" / RUN_ID
    selection_dir = artifact_root / "model_selection" / EXPERIMENT
    robustness_dir = artifact_root / "robustness" / EXPERIMENT
    diagnostics_dir = artifact_root / "diagnostics" / EXPERIMENT
    sensitivity_dir = artifact_root / "sensitivity" / EXPERIMENT
    statistical_dir = artifact_root / "statistical_validation" / EXPERIMENT
    config_dir = tmp_path / "configs"
    for directory in [
        run_dir,
        selection_dir,
        robustness_dir,
        diagnostics_dir,
        sensitivity_dir,
        statistical_dir,
        config_dir,
    ]:
        directory.mkdir(parents=True)

    snapshots = {
        "config.yaml": "split: validation\n",
        "env.yaml": "transaction_cost_bps: 5\n",
        "train_ppo.yaml": "seed: 42\n",
        "feature_spec_v1.json": '{"version": "v1"}\n',
    }
    for name, content in snapshots.items():
        (run_dir / name).write_text(content, encoding="utf-8")
    (run_dir / "best_model.zip").write_bytes(b"checkpoint")
    _write_json(
        run_dir / "best_metrics_validation.json",
        {
            "total_return": 0.11,
            "sharpe_ratio": 1.35,
            "max_drawdown": -0.04,
            "average_weekly_turnover": 0.08,
            "transaction_cost_drag": 0.004,
        },
    )
    _write_json(
        run_dir / "manifest.json",
        {
            "run_id": RUN_ID,
            "seed": 42,
            "data_config_hash": _sha256(run_dir / "config.yaml"),
            "env_config_hash": _sha256(run_dir / "env.yaml"),
            "train_config_hash": _sha256(run_dir / "train_ppo.yaml"),
            "feature_spec_hash": _sha256(run_dir / "feature_spec_v1.json"),
        },
    )
    (config_dir / "evaluation.yaml").write_text(
        "selection: validation\n", encoding="utf-8"
    )
    (config_dir / "universe.yaml").write_text("assets: [SPY, SHY]\n", encoding="utf-8")

    selected_path = selection_dir / "selected_configuration.json"
    _write_json(
        selected_path,
        {
            "configuration_id": CONFIGURATION_ID,
            "experiment_name": EXPERIMENT,
            "eligible_seeds": [7, 42, 101, 202, 999],
            "gate_results": {"finite_metrics": True, "seed_coverage": True},
            "validation_only": True,
            "test_split_used": False,
            "baseline_deltas": {
                "equal_weight_weekly": {
                    "total_return": 0.01,
                    "sharpe_ratio": 0.10,
                }
            },
            "sources": {
                "evaluation_config": {"sha256": _sha256(config_dir / "evaluation.yaml")}
            },
        },
    )
    (selection_dir / "validation_selection_report.md").write_text(
        "# Selection\n", encoding="utf-8"
    )

    model_hash = _sha256(run_dir / "best_model.zip")
    common = {
        "configuration_id": CONFIGURATION_ID,
        "experiment_name": EXPERIMENT,
        "validation_only": True,
        "test_split_used": False,
    }
    _write_json(
        robustness_dir / "robustness_manifest.json",
        {
            **common,
            "diagnostics": {"costs_complete": True, "regimes_complete": True},
            "models": [
                {"seed": 42, "model": {"sha256": model_hash}},
            ],
        },
    )
    (robustness_dir / "robustness_report.md").write_text(
        "# Robustness\n", encoding="utf-8"
    )
    _write_json(
        diagnostics_dir / "allocation_summary.json",
        {
            **common,
            "all_validation_metrics_reconciled": True,
            "selected_models": [{"seed": 42, "sha256": model_hash}],
            "campaign_warnings": {},
        },
    )
    (diagnostics_dir / "diagnostics_report.md").write_text(
        "# Diagnostics\n", encoding="utf-8"
    )
    _write_json(
        sensitivity_dir / "sensitivity_manifest.json",
        {
            **common,
            "all_observed_actions_reconciled": True,
            "selected_models": [{"seed": 42, "sha256": model_hash}],
            "campaign_interpretation": {
                "stop_before_packaging": False,
                "recommendation": "does_not_block",
            },
        },
    )
    (sensitivity_dir / "sensitivity_report.md").write_text(
        "# Sensitivity\n", encoding="utf-8"
    )
    _write_json(
        statistical_dir / "bootstrap_summary.json",
        {
            "validation_only": True,
            "test_split_used": False,
            "warnings": ["Selection bias remains."],
        },
    )
    (statistical_dir / "bootstrap_report.md").write_text(
        "# Bootstrap\n", encoding="utf-8"
    )

    registry_path = artifact_root / "experiments" / "registry.csv"
    pd.DataFrame(
        [
            {
                "run_id": RUN_ID,
                "experiment_name": EXPERIMENT,
                "seed": 42,
                "feature_version": "v1",
                "selection_eligible": True,
                "selection_checkpoint": "best_checkpoint",
                "selection_model_path": str(
                    (run_dir / "best_model.zip").relative_to(tmp_path)
                ),
                "selection_validation_total_return": 0.11,
                "selection_validation_sharpe_ratio": 1.35,
                "selection_validation_max_drawdown": -0.04,
                "selection_validation_average_weekly_turnover": 0.08,
                "selection_validation_transaction_cost_drag": 0.004,
            }
        ]
    ).to_csv(registry_path, index=False)
    return {
        "selected_configuration_path": selected_path.relative_to(tmp_path),
        "registry_path": registry_path.relative_to(tmp_path),
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

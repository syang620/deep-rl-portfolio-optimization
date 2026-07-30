from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pandas as pd
import pytest
import yaml

from portfolio_rl.training.provenance import (
    ProvenanceAuditError,
    audit_phase3_campaign,
    freeze_phase3_campaign,
)

CAMPAIGN_ID = "test_campaign_config123"
EXPERIMENT = "test_campaign"
CONFIGURATION_ID = "config123"
SEEDS = [7, 42]


def test_freeze_writes_audited_inventory_and_blocks_phase3b(
    tmp_path: Path,
) -> None:
    config_path = _campaign_fixture(tmp_path)

    result = freeze_phase3_campaign(
        config_path=config_path,
        output_root="artifacts/research_freeze",
        root=tmp_path,
    )

    assert result.campaign_test_free is True
    assert result.phase3b_authorized is False
    assert result.output_dir.name == CAMPAIGN_ID
    assert {path.name for path in result.output_dir.iterdir()} == {
        "freeze_manifest.json",
        "model_inventory.csv",
        "hash_inventory.json",
        "provenance_report.md",
        "test_access_audit.json",
    }

    models = pd.read_csv(result.output_dir / "model_inventory.csv")
    assert models["seed"].tolist() == SEEDS
    audit = _read_json(result.output_dir / "test_access_audit.json")
    assert audit["campaign_test_free"] is True
    assert audit["project_test_history_clear"] is False
    assert len(audit["known_legacy_access"]) == 1
    assert audit["phase3b_authorized"] is False

    manifest = _read_json(result.output_dir / "freeze_manifest.json")
    assert manifest["provenance_passed"] is True
    assert manifest["phase3b_authorized"] is False
    for record in manifest["files"]:
        payload_path = result.output_dir / record["path"]
        assert payload_path.stat().st_size == record["size_bytes"]
        assert _sha256(payload_path) == record["sha256"]


def test_audit_rejects_model_hash_mismatch(tmp_path: Path) -> None:
    config_path = _campaign_fixture(tmp_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["models"][0]["model_sha256"] = "0" * 64
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ProvenanceAuditError, match="seed 7 model"):
        audit_phase3_campaign(config_path=config_path, root=tmp_path)


def test_audit_rejects_turnover_source_mismatch(tmp_path: Path) -> None:
    config_path = _campaign_fixture(tmp_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["turnover_contract"]["source_sha256"] = "0" * 64
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(
        ProvenanceAuditError,
        match="does not match the frozen turnover-v2 contract",
    ):
        audit_phase3_campaign(config_path=config_path, root=tmp_path)


def test_audit_rejects_seed_inventory_mismatch(tmp_path: Path) -> None:
    config_path = _campaign_fixture(tmp_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["models"].pop()
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ProvenanceAuditError, match="eligible seeds"):
        audit_phase3_campaign(config_path=config_path, root=tmp_path)


def test_audit_rejects_current_campaign_test_access(tmp_path: Path) -> None:
    config_path = _campaign_fixture(tmp_path)
    _write_json(
        tmp_path / "artifacts/backtests/current_test/metadata.json",
        {
            "split": "test",
            "model_path": ("artifacts/experiments/run_7/best_model.zip"),
            "created_at": "2026-07-28T00:00:00+00:00",
        },
    )

    with pytest.raises(ProvenanceAuditError, match="campaign model"):
        audit_phase3_campaign(config_path=config_path, root=tmp_path)


def test_audit_rejects_undeclared_legacy_test_access(
    tmp_path: Path,
) -> None:
    config_path = _campaign_fixture(tmp_path)
    _write_json(
        tmp_path / "artifacts/backtests/unknown_test/metadata.json",
        {
            "split": "test",
            "model_path": "artifacts/experiments/unknown/model.zip",
        },
    )

    with pytest.raises(ProvenanceAuditError, match="Undeclared"):
        audit_phase3_campaign(config_path=config_path, root=tmp_path)


def test_freeze_refuses_to_overwrite_existing_campaign(
    tmp_path: Path,
) -> None:
    config_path = _campaign_fixture(tmp_path)
    freeze_phase3_campaign(
        config_path=config_path,
        output_root="artifacts/research_freeze",
        root=tmp_path,
    )

    with pytest.raises(FileExistsError, match="already exists"):
        freeze_phase3_campaign(
            config_path=config_path,
            output_root="artifacts/research_freeze",
            root=tmp_path,
        )


def _campaign_fixture(tmp_path: Path) -> Path:
    source_path = tmp_path / "src/portfolio_rl/env/costs.py"
    source_path.parent.mkdir(parents=True)
    source_path.write_text(
        "def turnover(current, target):\n"
        "    return 0.5 * abs(target - current).sum()\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", str(source_path)], cwd=tmp_path, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-qm",
            "turnover v2",
        ],
        cwd=tmp_path,
        check=True,
    )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    artifact_root = tmp_path / "artifacts"
    experiment_root = artifact_root / "experiments"
    selected_path = (
        artifact_root / "model_selection" / EXPERIMENT / "selected_configuration.json"
    )
    _write_json(
        selected_path,
        {
            "configuration_id": CONFIGURATION_ID,
            "experiment_name": EXPERIMENT,
            "eligible_seeds": SEEDS,
            "gate_results": {"seed_coverage": True},
            "validation_only": True,
            "test_split_used": False,
        },
    )

    registry_rows = []
    configured_models = []
    for seed in SEEDS:
        run_id = f"run_{seed}"
        run_dir = experiment_root / run_id
        run_dir.mkdir(parents=True)
        snapshots = {
            "config.yaml": "train_end_date: '2023-12-31'\n",
            "data_quality_report_v1.json": '{"valid": true}\n',
            "env.yaml": ("initial_weights: equal_weight\ntransaction_cost_bps: 10.0\n"),
            "feature_spec_v1.json": '{"feature_version": "v1"}\n',
            "train_ppo.yaml": f"seed: {seed}\n",
        }
        for name, content in snapshots.items():
            (run_dir / name).write_text(content, encoding="utf-8")
        model_path = run_dir / "best_model.zip"
        model_path.write_bytes(f"model-{seed}".encode())
        manifest = {
            "run_id": run_id,
            "seed": seed,
            "git_commit": commit,
            "feature_version": "v1",
            "total_timesteps": 500000,
            "data_config_hash": _sha256(run_dir / "config.yaml"),
            "data_quality_report_hash": _sha256(
                run_dir / "data_quality_report_v1.json"
            ),
            "env_config_hash": _sha256(run_dir / "env.yaml"),
            "feature_spec_hash": _sha256(run_dir / "feature_spec_v1.json"),
            "train_config_hash": _sha256(run_dir / "train_ppo.yaml"),
        }
        manifest_path = run_dir / "manifest.json"
        _write_json(manifest_path, manifest)
        registry_rows.append(
            {
                "run_id": run_id,
                "experiment_name": EXPERIMENT,
                "seed": seed,
                "selection_checkpoint": "best_checkpoint",
                "selection_model_path": str(model_path.relative_to(tmp_path)),
                "manifest_path": str(manifest_path.relative_to(tmp_path)),
                "selection_eligible": True,
            }
        )
        configured_models.append(
            {
                "seed": seed,
                "run_id": run_id,
                "selection_checkpoint": "best_checkpoint",
                "model_path": str(model_path.relative_to(tmp_path)),
                "model_sha256": _sha256(model_path),
                "manifest_path": str(manifest_path.relative_to(tmp_path)),
                "manifest_sha256": _sha256(manifest_path),
            }
        )

    registry_path = experiment_root / "registry.csv"
    pd.DataFrame(registry_rows).to_csv(registry_path, index=False)
    legacy_path = artifact_root / "backtests/legacy_test/metadata.json"
    _write_json(
        legacy_path,
        {
            "split": "test",
            "model_path": "artifacts/experiments/legacy/model.zip",
            "confirm_final_test": True,
        },
    )

    config = {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "experiment_name": EXPERIMENT,
        "configuration_id": CONFIGURATION_ID,
        "repository_baseline_commit": commit,
        "selected_configuration_path": str(selected_path.relative_to(tmp_path)),
        "selected_configuration_sha256": _sha256(selected_path),
        "registry_path": str(registry_path.relative_to(tmp_path)),
        "registry_sha256": _sha256(registry_path),
        "artifact_root": "artifacts",
        "development_period": {
            "start_date": "2024-01-01",
            "end_date": "2024-12-31",
            "label": "2024 consumed development/selection period",
        },
        "turnover_contract": {
            "version": "turnover_v2_one_way_half_l1",
            "training_git_commit": commit,
            "source_path": "src/portfolio_rl/env/costs.py",
            "source_sha256": _sha256(source_path),
            "transaction_cost_bps": 10.0,
        },
        "expected_feature_version": "v1",
        "expected_total_timesteps": 500000,
        "models": configured_models,
        "test_governance": {
            "phase3b_policy": ("block_until_new_independent_holdout_is_approved"),
            "known_legacy_access": [
                {
                    "metadata_path": str(legacy_path.relative_to(tmp_path)),
                    "metadata_sha256": _sha256(legacy_path),
                    "model_path": ("artifacts/experiments/legacy/model.zip"),
                    "split": "test",
                }
            ],
        },
    }
    config_path = tmp_path / "configs/research/phase3_candidate_qualification.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

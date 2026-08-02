from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from portfolio_rl.training import pretest_freeze
from portfolio_rl.training.pretest_freeze import (
    EXPECTED_ASSET_ORDER,
    EXPECTED_SEEDS,
    PretestFreezeError,
    freeze_final_candidate,
    verify_frozen_candidate,
)

MODEL_VERSION = "ensemble_alpha025_fixture"


def test_freeze_builds_complete_immutable_package(tmp_path: Path) -> None:
    inputs = _fixture(tmp_path)

    output = freeze_final_candidate(**inputs)
    candidate = verify_frozen_candidate(output / "frozen_candidate.json")

    assert candidate.member_seeds == EXPECTED_SEEDS
    assert candidate.partial_rebalance_alpha == 0.25
    assert candidate.initial_portfolio == "equal_weight"
    manifest = _read_json(output / "freeze_manifest.json")
    assert {record["path"] for record in manifest["files"]} == {
        path.name for path in output.iterdir() if path.name != "freeze_manifest.json"
    }
    audit = _read_json(output / "test_access_audit.json")
    assert audit["test_accessed"] is False
    assert audit["phase3b_authorized"] is False


def test_freeze_rejects_existing_output_directory(tmp_path: Path) -> None:
    inputs = _fixture(tmp_path)
    destination = tmp_path / "artifacts/pretest_freeze" / MODEL_VERSION
    destination.mkdir(parents=True)

    with pytest.raises(FileExistsError, match="already exists"):
        freeze_final_candidate(**inputs)


def test_freeze_rejects_model_hash_mismatch(tmp_path: Path) -> None:
    inputs = _fixture(tmp_path)
    ensemble_path = tmp_path / "artifacts/ensemble/campaign/ensemble_manifest.json"
    ensemble = _read_json(ensemble_path)
    ensemble["members"][0]["model_sha256"] = "0" * 64
    _write_json(ensemble_path, ensemble)
    _refresh_source_hash(tmp_path, "ensemble", ensemble_path)
    _commit_fixture(tmp_path, "mutate model expectation")

    with pytest.raises(PretestFreezeError, match="member model hash mismatch"):
        freeze_final_candidate(**inputs)


def test_freeze_requires_all_five_member_models(tmp_path: Path) -> None:
    inputs = _fixture(tmp_path)
    ensemble_path = tmp_path / "artifacts/ensemble/campaign/ensemble_manifest.json"
    ensemble = _read_json(ensemble_path)
    ensemble["members"] = ensemble["members"][:-1]
    _write_json(ensemble_path, ensemble)
    _refresh_source_hash(tmp_path, "ensemble", ensemble_path)
    _commit_fixture(tmp_path, "remove ensemble member")

    with pytest.raises(PretestFreezeError, match="all five members"):
        freeze_final_candidate(**inputs)


def test_freeze_rejects_feature_spec_hash_mismatch(tmp_path: Path) -> None:
    inputs = _fixture(tmp_path)
    feature_path = tmp_path / "artifacts/experiments/shared/feature_spec_v1.json"
    feature = _read_json(feature_path)
    feature["observation_dim"] = 315
    _write_json(feature_path, feature)

    ensemble_path = tmp_path / "artifacts/ensemble/campaign/ensemble_manifest.json"
    ensemble = _read_json(ensemble_path)
    ensemble["source_hashes"]["feature_spec"]["sha256"] = _sha(feature_path)
    _write_json(ensemble_path, ensemble)
    _refresh_source_hash(tmp_path, "ensemble", ensemble_path)
    _commit_fixture(tmp_path, "mutate feature spec")

    with pytest.raises(PretestFreezeError, match="observation dimension"):
        freeze_final_candidate(**inputs)


def test_verify_detects_post_freeze_mutation(tmp_path: Path) -> None:
    output = freeze_final_candidate(**_fixture(tmp_path))
    candidate_path = output / "frozen_candidate.json"
    payload = _read_json(candidate_path)
    payload["partial_rebalance_alpha"] = 0.5
    _write_json(candidate_path, payload)

    with pytest.raises(PretestFreezeError, match="generated freeze artifact hash"):
        verify_frozen_candidate(candidate_path)


@pytest.mark.parametrize(
    ("filename", "mutation"),
    [
        (
            "frozen_candidate.json",
            lambda payload: payload.update(
                {"member_seed_order": [42, 7, 101, 202, 999]}
            ),
        ),
        (
            "baseline_definitions.json",
            lambda payload: payload["common_contract"].update(
                {"transaction_cost_bps": 25.0}
            ),
        ),
    ],
)
def test_verify_rejects_mutated_json_payload(
    tmp_path: Path, filename: str, mutation
) -> None:
    output = freeze_final_candidate(**_fixture(tmp_path))
    path = output / filename
    payload = _read_json(path)
    mutation(payload)
    _write_json(path, payload)

    with pytest.raises(PretestFreezeError, match="generated freeze artifact hash"):
        verify_frozen_candidate(output / "frozen_candidate.json")


def test_verify_rejects_mutated_acceptance_criteria(tmp_path: Path) -> None:
    output = freeze_final_candidate(**_fixture(tmp_path))
    path = output / "acceptance_criteria.yaml"
    path.write_text(path.read_text(encoding="utf-8") + "\n# mutation\n", encoding="utf-8")

    with pytest.raises(PretestFreezeError, match="generated freeze artifact hash"):
        verify_frozen_candidate(output / "frozen_candidate.json")


def test_verify_rejects_mutated_member_model(tmp_path: Path) -> None:
    output = freeze_final_candidate(**_fixture(tmp_path))
    model_path = tmp_path / "artifacts/experiments/run_7/best_model.zip"
    original = bytearray(model_path.read_bytes())
    original[0] ^= 0x01
    model_path.write_bytes(original)

    with pytest.raises(PretestFreezeError, match="frozen member model hash mismatch"):
        verify_frozen_candidate(output / "frozen_candidate.json")


def test_freeze_rejects_unexpected_test_artifact(tmp_path: Path) -> None:
    inputs = _fixture(tmp_path)
    path = tmp_path / "artifacts/unexpected/test_metadata.json"
    path.parent.mkdir(parents=True)
    _write_json(path, {"split": "test", "model_path": "other/model.zip"})
    _commit_fixture(tmp_path, "add unexpected test metadata")

    with pytest.raises(PretestFreezeError, match="unexpected test artifacts"):
        freeze_final_candidate(**inputs)


def test_verify_rejects_unexpected_test_artifact_added_after_freeze(
    tmp_path: Path,
) -> None:
    output = freeze_final_candidate(**_fixture(tmp_path))
    path = tmp_path / "artifacts/unexpected/test_metadata.json"
    path.parent.mkdir(parents=True)
    _write_json(path, {"split": "test", "model_path": "other/model.zip"})

    with pytest.raises(PretestFreezeError, match="unexpected test artifacts"):
        verify_frozen_candidate(output / "frozen_candidate.json")


def test_frozen_package_verifies_from_fresh_process(tmp_path: Path) -> None:
    output = freeze_final_candidate(**_fixture(tmp_path))
    candidate_path = output / "frozen_candidate.json"
    command = (
        "from pathlib import Path; "
        "from portfolio_rl.training.pretest_freeze import verify_frozen_candidate; "
        f"verify_frozen_candidate(Path({str(candidate_path)!r}))"
    )

    result = subprocess.run(
        [sys.executable, "-c", command],
        cwd=Path.cwd(),
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_frozen_loader_uses_one_live_portfolio_and_mean_then_overlay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = freeze_final_candidate(**_fixture(tmp_path))
    targets = {}
    for seed_index, seed in enumerate(EXPECTED_SEEDS):
        target = np.zeros(len(EXPECTED_ASSET_ORDER), dtype=np.float64)
        target[seed_index] = 1.0
        targets[seed] = target
    observations = []
    live_weights = []

    class RecordingPolicy:
        def __init__(self, target):
            self.target = target

        def target_weights(self, observation, info):
            observations.append(np.asarray(observation).copy())
            live_weights.append(np.asarray(info["current_weights"]).copy())
            return self.target

    def fake_loader(model_path, action_temperature):
        del action_temperature
        seed = int(Path(model_path).parent.name.removeprefix("run_"))
        return RecordingPolicy(targets[seed])

    monkeypatch.setattr(pretest_freeze, "load_sb3_weight_policy", fake_loader)
    policy = pretest_freeze.load_frozen_candidate_policy(
        output / "frozen_candidate.json"
    )
    current = np.full(len(EXPECTED_ASSET_ORDER), 1 / len(EXPECTED_ASSET_ORDER))
    observation = np.concatenate([np.zeros(302), current])
    executed = policy.target_weights(
        observation,
        {"current_weights": current, "asset_order": list(EXPECTED_ASSET_ORDER)},
    )

    ensemble_target = np.mean(np.stack(list(targets.values())), axis=0)
    np.testing.assert_allclose(executed, current + 0.25 * (ensemble_target - current))
    assert len(observations) == 5
    for observed, weights in zip(observations, live_weights, strict=True):
        np.testing.assert_allclose(observed, observation)
        np.testing.assert_allclose(weights, current)


def _fixture(root: Path) -> dict[str, Path | str]:
    _init_git(root)
    artifact_root = root / "artifacts"
    feature_path = artifact_root / "experiments/shared/feature_spec_v1.json"
    feature_path.parent.mkdir(parents=True)
    _write_json(
        feature_path,
        {
            "feature_version": "v1",
            "asset_order": list(EXPECTED_ASSET_ORDER),
            "observation_dim": 316,
            "per_asset_features": [],
            "global_features": [],
            "current_weight_features": list(EXPECTED_ASSET_ORDER),
        },
    )
    env = {
        "rebalance_frequency_trading_days": 5,
        "transaction_cost_bps": 10.0,
        "action_temperature": 0.5,
    }
    members = []
    for order, seed in enumerate(EXPECTED_SEEDS):
        run_id = f"run_{seed}"
        run_dir = artifact_root / "experiments" / run_id
        run_dir.mkdir(parents=True)
        model_path = run_dir / "best_model.zip"
        model_path.write_bytes(f"model-{seed}".encode())
        snapshots = {
            "config.yaml": "train_end_date: '2023-12-31'\n",
            "data_quality_report_v1.json": "{}\n",
            "env.yaml": yaml.safe_dump(env, sort_keys=True),
            "feature_spec_v1.json": feature_path.read_text(encoding="utf-8"),
            "train_ppo.yaml": f"seed: {seed}\ntotal_timesteps: 500000\n",
        }
        for name, content in snapshots.items():
            (run_dir / name).write_text(content, encoding="utf-8")
        manifest = {
            "seed": seed,
            "run_id": run_id,
            "git_commit": "training-commit",
            "data_config_hash": _sha(run_dir / "config.yaml"),
            "data_quality_report_hash": _sha(run_dir / "data_quality_report_v1.json"),
            "env_config_hash": _sha(run_dir / "env.yaml"),
            "feature_spec_hash": _sha(run_dir / "feature_spec_v1.json"),
            "train_config_hash": _sha(run_dir / "train_ppo.yaml"),
        }
        _write_json(run_dir / "manifest.json", manifest)
        members.append(
            {
                "order": order,
                "seed": seed,
                "run_id": run_id,
                "selection_checkpoint": "best_checkpoint",
                "model_path": str(model_path.relative_to(root)),
                "model_sha256": _sha(model_path),
                "action_temperature": 0.5,
            }
        )

    research_path = artifact_root / "research_freeze/campaign/freeze_manifest.json"
    research_path.parent.mkdir(parents=True)
    _write_json(
        research_path,
        {
            "campaign_id": "campaign",
            "campaign_test_free": True,
            "phase3b_authorized": False,
            "legacy_project_test_access_detected": False,
        },
    )
    _write_json(
        research_path.parent / "test_access_audit.json",
        {"known_legacy_access": []},
    )
    ensemble_path = artifact_root / "ensemble/campaign/ensemble_manifest.json"
    ensemble_path.parent.mkdir(parents=True)
    _write_json(
        ensemble_path,
        {
            "member_seed_order": list(EXPECTED_SEEDS),
            "members": members,
            "test_accessed": False,
            "development_data_label": "2024 consumed development/selection data",
            "source_hashes": {
                "feature_spec": {
                    "path": str(feature_path.relative_to(root)),
                    "sha256": _sha(feature_path),
                },
                "model_matrix": {
                    "path": "data/processed/model_matrix_daily.parquet",
                    "sha256": "d" * 64,
                },
            },
        },
    )
    init_path = artifact_root / "initialization/campaign/run_manifest.json"
    overlay_path = artifact_root / "overlay/campaign/overlay_manifest.json"
    dynamic_path = artifact_root / "dynamic/campaign/dynamic_value_manifest.json"
    for path in (init_path, overlay_path, dynamic_path):
        path.parent.mkdir(parents=True)
        _write_json(path, {"test_accessed": False})
    wf_path = artifact_root / "walk_forward/aggregation_manifest.json"
    wf_path.parent.mkdir(parents=True)
    _write_json(
        wf_path,
        {"candidate_selected": False, "folds": ["WF1", "WF2", "WF3", "WF4"]},
    )
    metrics_path = artifact_root / "walk_forward/fold_metrics.csv"
    pd.DataFrame(
        [
            {
                "fold_id": fold,
                "strategy": "ensemble_alpha_0.25",
                "active_return_vs_equal_weight": active,
                "active_sharpe_vs_equal_weight": sharpe,
                "average_weekly_turnover": 0.025,
            }
            for fold, active, sharpe in zip(
                ["WF1", "WF2", "WF3", "WF4"],
                [0.01, 0.02, 0.03, -0.01],
                [0.01, 0.02, 0.03, -0.01],
                strict=True,
            )
        ]
    ).to_csv(metrics_path, index=False)
    execution_path = artifact_root / "execution/execution_stress_manifest.json"
    execution_path.parent.mkdir(parents=True)
    _write_json(
        execution_path,
        {"test_accessed": False, "latest_evaluation_date": "2023-12-31"},
    )
    attribution_path = artifact_root / "attribution/regime_attribution_manifest.json"
    attribution_path.parent.mkdir(parents=True)
    _write_json(
        attribution_path,
        {
            "test_accessed": False,
            "evaluation_period": "2022-01-03 through 2023-12-29",
        },
    )

    config_path = root / "configs/final_candidate_acceptance.yaml"
    config_path.parent.mkdir(parents=True)
    config = _acceptance_config(
        research_path=research_path,
        init_path=init_path,
        ensemble_path=ensemble_path,
        overlay_path=overlay_path,
        dynamic_path=dynamic_path,
        wf_path=wf_path,
        metrics_path=metrics_path,
        execution_path=execution_path,
        attribution_path=attribution_path,
        root=root,
    )
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    _commit_fixture(root, "fixture")
    return {
        "repository_root": root,
        "research_freeze_path": research_path.relative_to(root),
        "ensemble_manifest_path": ensemble_path.relative_to(root),
        "walk_forward_results_path": wf_path.relative_to(root),
        "execution_stress_path": execution_path.relative_to(root),
        "regime_attribution_path": attribution_path.relative_to(root),
        "acceptance_config_path": config_path.relative_to(root),
        "model_version": MODEL_VERSION,
        "output_root": Path("artifacts/pretest_freeze"),
    }


def _acceptance_config(**paths) -> dict[str, object]:
    root = paths.pop("root")
    names = {
        "research_freeze": paths["research_path"],
        "initialization_sensitivity": paths["init_path"],
        "ensemble": paths["ensemble_path"],
        "turnover_overlay": paths["overlay_path"],
        "dynamic_value": paths["dynamic_path"],
        "walk_forward": paths["wf_path"],
        "walk_forward_metrics": paths["metrics_path"],
        "execution_stress": paths["execution_path"],
        "regime_attribution": paths["attribution_path"],
    }
    return {
        "schema_version": 1,
        "model_version": MODEL_VERSION,
        "candidate": {
            "policy_type": "five_seed_mean_target_ensemble",
            "seeds": list(EXPECTED_SEEDS),
            "ensemble_rule": "arithmetic_mean_of_member_target_weights",
            "partial_rebalance_alpha": 0.25,
            "initial_portfolio": "equal_weight",
            "rebalance_frequency_trading_days": 5,
            "transaction_cost_bps": 10.0,
            "turnover_definition": "half_l1_one_way",
            "long_only": True,
            "fully_invested": True,
        },
        "primary_hurdle": {"strategy": "equal_weight_weekly"},
        "hard_gates": {
            "positive_active_return_vs_primary": True,
            "positive_sharpe_difference_vs_primary": True,
            "require_finite_metrics": True,
            "require_positive_final_nav": True,
            "require_no_artifact_hash_mismatch": True,
            "require_no_test_overwrite": True,
        },
        "advisory_diagnostics": ["one_close_delay_result"],
        "secondary_comparisons": ["inverse_volatility"],
        "governance": {
            "phase3b_authorized": False,
            **pretest_freeze.BLOCKED_GOVERNANCE,
            "acceptance_approval_status": "approved",
        },
        "evidence_sources": {
            name: {"path": str(path.relative_to(root)), "sha256": _sha(path)}
            for name, path in names.items()
        },
    }


def _refresh_source_hash(root: Path, name: str, path: Path) -> None:
    config_path = root / "configs/final_candidate_acceptance.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["evidence_sources"][name]["sha256"] = _sha(path)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def _init_git(root: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=root, check=True)
    source_root = Path("src/portfolio_rl")
    for relative in (
        "env/action.py",
        "env/costs.py",
        "env/drift.py",
        "policies/ensemble_policy.py",
        "policies/overlays.py",
        "policies/sb3_policy.py",
        "evaluation/backtest.py",
    ):
        destination = root / "src/portfolio_rl" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((source_root / relative).read_bytes())


def _commit_fixture(root: Path, message: str) -> None:
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(["git", "commit", "-qm", message], cwd=root, check=True)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))

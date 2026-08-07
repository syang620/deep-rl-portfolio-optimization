from __future__ import annotations

import json
import subprocess
from datetime import UTC, date, datetime, time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from portfolio_rl.features.feature_spec import FeatureSpec
from portfolio_rl.phase3b import shadow_runner
from portfolio_rl.phase3b.execution import (
    ExecutionConfig,
    RecommendationSigningConfig,
)
from portfolio_rl.phase3b.frozen_candidate_loader import FrozenCandidateRuntime
from portfolio_rl.phase3b.governance import (
    GovernanceError,
    logical_json_sha256,
    sha256_file,
    ssh_public_key_fingerprint,
)
from portfolio_rl.phase3b.holdout_registry import RegisteredHoldout
from portfolio_rl.phase3b.shadow_runner import (
    generate_shadow_decision,
    verify_shadow_decision,
    write_shadow_decision,
)
from portfolio_rl.phase3b.snapshot_chain import (
    GENESIS_CHAIN_HASH,
    STRATEGIES,
    dataframe_logical_sha256,
    load_live_portfolio_state,
    load_point_in_time_snapshot,
)
from portfolio_rl.training.pretest_freeze import FrozenCandidate

ASSETS = (
    "SPY",
    "QQQ",
    "IWM",
    "EFA",
    "EEM",
    "TLT",
    "IEF",
    "SHY",
    "LQD",
    "HYG",
    "GLD",
    "DBC",
    "VNQ",
    "XLU",
)
SEEDS = (7, 42, 101, 202, 999)


class _FixedPolicy:
    def __init__(self, weights: np.ndarray, observed: list[np.ndarray]) -> None:
        self._weights = weights
        self._observed = observed

    def target_weights(self, observation: np.ndarray, info: object) -> np.ndarray:
        del info
        self._observed.append(observation.copy())
        return self._weights.copy()


def test_shadow_decision_is_deterministic_signed_and_mutation_evident(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    runtime = fixture["runtime"]
    config = fixture["config"]
    snapshot = load_point_in_time_snapshot(
        manifest_path=fixture["snapshot_manifest"],
        repository_root=tmp_path,
        config=config,
        expected_asset_order=ASSETS,
        expected_feature_version="v1",
        expected_feature_spec_sha256="7" * 64,
    )
    state = load_live_portfolio_state(
        manifest_path=fixture["state_manifest"],
        repository_root=tmp_path,
        config=config,
        expected_asset_order=ASSETS,
        decision_date=snapshot.decision_date,
    )
    generated_at = datetime(2030, 1, 3, 14, tzinfo=UTC)

    first = generate_shadow_decision(
        runtime=runtime,
        snapshot=snapshot,
        live_state=state,
        generated_at=generated_at,
        execution_config=config,
    )
    second = generate_shadow_decision(
        runtime=runtime,
        snapshot=snapshot,
        live_state=state,
        generated_at=generated_at,
        execution_config=config,
    )

    expected_mean = np.mean(
        np.stack([policy._weights for _, policy in runtime.member_policies]), axis=0
    )
    current = np.asarray(state.weights["candidate"])
    assert np.allclose(first.ensemble_target, expected_mean)
    assert np.allclose(
        first.executed_target, current + 0.25 * (expected_mean - current)
    )
    assert first == second
    assert all(
        np.array_equal(observation, fixture["observations"][0])
        for observation in fixture["observations"]
    )
    assert set(first.baseline_targets) == set(STRATEGIES) - {"candidate"}
    assert all(row.execution_delay_closes == 1 for row in first.execution_instructions)
    assert all(
        row.turnover_status.startswith("pending")
        for row in first.execution_instructions
    )

    output = write_shadow_decision(
        decision=first,
        runtime=runtime,
        execution_config=config,
        registration_dir=fixture["registration_dir"],
        runtime_identity_path=fixture["runtime_identity"],
        signing_key_path=fixture["private_key"],
        repository_root=tmp_path,
    )
    manifest = verify_shadow_decision(
        decision_dir=output,
        execution_config=config,
        registration_dir=fixture["registration_dir"],
        repository_root=tmp_path,
    )
    assert manifest["performance_computed"] is False
    assert manifest["test_accessed"] is False
    assert manifest["previous_chain_hash"] == GENESIS_CHAIN_HASH
    assert (output / "input_live_state_manifest.json").exists()

    assert (
        write_shadow_decision(
            decision=second,
            runtime=runtime,
            execution_config=config,
            registration_dir=fixture["registration_dir"],
            runtime_identity_path=fixture["runtime_identity"],
            signing_key_path=fixture["private_key"],
            repository_root=tmp_path,
        )
        == output
    )

    target_path = output / "executed_target.parquet"
    mutated = pd.read_parquet(target_path)
    mutated.loc[0, "executed_target_weight"] += 0.01
    mutated.to_parquet(target_path, index=False)
    with pytest.raises(GovernanceError, match="artifact hash mismatch"):
        verify_shadow_decision(
            decision_dir=output,
            execution_config=config,
            registration_dir=fixture["registration_dir"],
            repository_root=tmp_path,
        )


def test_snapshot_rejects_future_trailing_return(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    returns_path = tmp_path / "inputs/trailing.parquet"
    frame = pd.read_parquet(returns_path)
    frame.loc[frame.index[-1], "date"] = pd.Timestamp("2030-01-02")
    frame.to_parquet(returns_path, index=False)
    _refresh_snapshot_hashes(tmp_path, fixture["snapshot_manifest"], frame)

    with pytest.raises(GovernanceError, match="strictly past"):
        load_point_in_time_snapshot(
            manifest_path=fixture["snapshot_manifest"],
            repository_root=tmp_path,
            config=fixture["config"],
            expected_asset_order=ASSETS,
            expected_feature_version="v1",
            expected_feature_spec_sha256="7" * 64,
        )


def test_shadow_decision_rejects_recommendation_after_cutoff(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    snapshot = load_point_in_time_snapshot(
        manifest_path=fixture["snapshot_manifest"],
        repository_root=tmp_path,
        config=fixture["config"],
        expected_asset_order=ASSETS,
        expected_feature_version="v1",
        expected_feature_spec_sha256="7" * 64,
    )
    state = load_live_portfolio_state(
        manifest_path=fixture["state_manifest"],
        repository_root=tmp_path,
        config=fixture["config"],
        expected_asset_order=ASSETS,
        decision_date=snapshot.decision_date,
    )

    with pytest.raises(GovernanceError, match="after the frozen cutoff"):
        generate_shadow_decision(
            runtime=fixture["runtime"],
            snapshot=snapshot,
            live_state=state,
            generated_at=datetime(2030, 1, 3, 16, tzinfo=UTC),
            execution_config=fixture["config"],
        )


def _fixture(root: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    inputs = root / "inputs"
    registration = root / "registration"
    inputs.mkdir()
    registration.mkdir()
    scaler = root / "scaler.pkl"
    scaler.write_bytes(b"fixture-scaler")
    private_key = root / "service_key"
    subprocess.run(
        ["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(private_key)],
        check=True,
    )
    public_key = Path(f"{private_key}.pub")
    config_path = root / "execution.yaml"
    config_path.write_text("fixture: true\n", encoding="utf-8")
    candidate_hash = "1" * 64
    config = ExecutionConfig(
        config_path=config_path,
        config_sha256=sha256_file(config_path),
        candidate_manifest_sha256=candidate_hash,
        normalization_artifact_path=scaler,
        normalization_artifact_sha256=sha256_file(scaler),
        feature_payload_schema_version="phase3b_normalized_features_v1",
        live_state_schema_version="phase3b_live_weights_v1",
        recommendation_cutoff_local=time(10, 0),
        timezone="America/New_York",
        execution_delay_closes=1,
        primary_transaction_cost_bps=10.0,
        asset_cost_bps={ticker: 5.0 for ticker in ASSETS},
        signing=RecommendationSigningConfig(
            principal="phase3b-service",
            public_key_path=public_key,
            public_key_sha256=sha256_file(public_key),
            public_key_fingerprint=ssh_public_key_fingerprint(public_key),
            namespace="portfolio-rl-phase3b-recommendation-v1",
        ),
        output_root=root / "outputs",
    )
    snapshot_manifest = _snapshot(root, config)
    state_manifest = _state(root, config)
    observed: list[np.ndarray] = []
    policies = []
    for index, seed in enumerate(SEEDS):
        weights = np.arange(1, 15, dtype=np.float64) + index
        weights /= weights.sum()
        policies.append((seed, _FixedPolicy(weights, observed)))
    candidate = FrozenCandidate(
        model_version="fixture-candidate",
        member_seeds=SEEDS,
        member_model_paths=tuple(Path(f"models/{seed}.zip") for seed in SEEDS),
        member_model_hashes=tuple(str(seed).zfill(64) for seed in SEEDS),
        action_temperatures=(0.5,) * 5,
        partial_rebalance_alpha=0.25,
        initial_portfolio="equal_weight",
        asset_order=ASSETS,
        feature_version="v1",
        feature_spec_hash="7" * 64,
        environment_config_hash="8" * 64,
        transaction_cost_bps=10.0,
        rebalance_frequency_trading_days=5,
    )
    frozen_candidate = root / "candidate/frozen_candidate.json"
    feature_spec_path = root / "candidate/feature_spec_v1.json"
    frozen_candidate.parent.mkdir()
    frozen_candidate.write_text("{}\n", encoding="utf-8")
    feature_spec_path.write_text("{}\n", encoding="utf-8")
    runtime = FrozenCandidateRuntime(
        candidate=candidate,
        member_policies=tuple(policies),
        feature_spec=FeatureSpec(
            feature_version="v1",
            asset_order=list(ASSETS),
            per_asset_features=[],
            global_features=[],
            current_weight_features=[],
            observation_dim=316,
            created_at="fixture",
        ),
        baseline_definitions=_baseline_definitions(),
        frozen_candidate_path=frozen_candidate,
        candidate_manifest_sha256=candidate_hash,
        baseline_definitions_sha256="9" * 64,
        feature_spec_path=feature_spec_path,
    )
    container = _container_identity()
    runtime_identity = root / "runtime_identity.json"
    _write_json(runtime_identity, container)
    _write_json(registration / "container_identity.json", container)
    _write_json(
        registration / "holdout_registration.json", {"holdout_id": "fixture-holdout"}
    )
    _write_json(
        registration / "config_hashes.json",
        {"configs": [{"name": "execution", "sha256": config.config_sha256}]},
    )
    schedule = root / "schedule.json"
    _write_json(schedule, {"holdout_decision_dates": ["2030-01-02"]})
    _write_json(
        registration / "approval_record.json",
        {
            "registration_challenge": {
                "inputs": {"trading_session_schedule": {"path": "schedule.json"}}
            }
        },
    )
    registered = RegisteredHoldout(
        holdout_id="fixture-holdout",
        start_decision_date=date(2030, 1, 2),
        end_decision_date=date(2030, 12, 31),
        final_holding_period_end_date=date(2031, 1, 7),
        performance_unseal_not_before=datetime(2031, 1, 8, tzinfo=UTC),
        candidate_model_version="fixture-candidate",
        candidate_manifest_sha256=candidate_hash,
        container_image_digest="sha256:" + "c" * 64,
        git_commit="a" * 40,
    )
    monkeypatch.setattr(
        shadow_runner, "verify_holdout_registration", lambda *a, **k: registered
    )
    return {
        "config": config,
        "runtime": runtime,
        "observations": observed,
        "snapshot_manifest": snapshot_manifest,
        "state_manifest": state_manifest,
        "registration_dir": registration,
        "runtime_identity": runtime_identity,
        "private_key": private_key,
    }


def _snapshot(root: Path, config: ExecutionConfig) -> Path:
    feature_path = root / "inputs/features.parquet"
    returns_path = root / "inputs/trailing.parquet"
    feature = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2030-01-02"),
                "feature_version": "v1",
                **{
                    f"obs_market_{index:03d}": float(index) / 302
                    for index in range(302)
                },
            }
        ]
    )
    dates = pd.bdate_range(end="2030-01-01", periods=63)
    returns = pd.DataFrame({"date": dates})
    for index, ticker in enumerate(ASSETS):
        returns[f"return_{ticker.lower()}_1d"] = (
            np.linspace(-0.01, 0.01, 63) + index / 10000
        )
    feature.to_parquet(feature_path, index=False)
    returns.to_parquet(returns_path, index=False)
    payload = {
        "schema_version": 1,
        "snapshot_id": "snapshot-fixture",
        "feature_payload_schema_version": config.feature_payload_schema_version,
        "decision_date": "2030-01-02",
        "as_of_close": "2030-01-02T21:00:00+00:00",
        "next_trading_date": "2030-01-03",
        "generated_at": "2030-01-02T22:00:00+00:00",
        "feature_version": "v1",
        "feature_spec_sha256": "7" * 64,
        "normalization_artifact_sha256": config.normalization_artifact_sha256,
        "asset_order": list(ASSETS),
        "source_inventory": [
            {
                "source": "fixture",
                "max_observation_date": "2030-01-02",
                "available_at": "2030-01-02T21:30:00+00:00",
                "vintage_id": "fixture-v1",
            }
        ],
        "files": {
            "feature_payload": _file_record(root, feature_path, feature),
            "trailing_log_returns": _file_record(root, returns_path, returns),
        },
    }
    payload["snapshot_payload_sha256"] = logical_json_sha256(payload)
    path = root / "inputs/snapshot.json"
    _write_json(path, payload)
    return path


def _state(root: Path, config: ExecutionConfig) -> Path:
    weights_path = root / "inputs/live_weights.parquet"
    equal = 1.0 / len(ASSETS)
    frame = pd.DataFrame(
        [
            {"strategy": strategy, "ticker": ticker, "current_weight": equal}
            for strategy in STRATEGIES
            for ticker in ASSETS
        ]
    )
    frame.to_parquet(weights_path, index=False)
    payload = {
        "schema_version": 1,
        "state_id": "state-fixture",
        "live_state_schema_version": config.live_state_schema_version,
        "as_of_date": "2030-01-02",
        "asset_order": list(ASSETS),
        "previous_chain_hash": GENESIS_CHAIN_HASH,
        "initial_endowment": True,
        "weights_file": _file_record(root, weights_path, frame),
    }
    payload["state_payload_sha256"] = logical_json_sha256(payload)
    path = root / "inputs/live_state.json"
    _write_json(path, payload)
    return path


def _refresh_snapshot_hashes(
    root: Path, manifest_path: Path, frame: pd.DataFrame
) -> None:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["files"]["trailing_log_returns"] = _file_record(
        root, root / "inputs/trailing.parquet", frame
    )
    payload.pop("snapshot_payload_sha256")
    payload["snapshot_payload_sha256"] = logical_json_sha256(payload)
    _write_json(manifest_path, payload)


def _file_record(root: Path, path: Path, frame: pd.DataFrame) -> dict[str, str]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": sha256_file(path),
        "logical_sha256": dataframe_logical_sha256(frame),
    }


def _baseline_definitions() -> dict[str, object]:
    return {
        "schema_version": 1,
        "primary_hurdle": "equal_weight_weekly",
        "definitions": {
            "equal_weight_weekly": {"rule": "equal"},
            "buy_and_hold_equal_weight": {"rule": "hold"},
            "inverse_volatility": {
                "lookback_trading_days": 63,
                "volatility_floor": 1e-8,
            },
            "momentum_63d_top3_equal_weight": {"lookback_trading_days": 63, "top_k": 3},
            "spy_only": {"ticker": "SPY"},
            "shy_only": {"ticker": "SHY"},
        },
    }


def _container_identity() -> dict[str, object]:
    return {
        "schema_version": 1,
        "image_reference": "registry.example/phase3b@sha256:" + "c" * 64,
        "image_digest": "sha256:" + "c" * 64,
        "git_commit": "a" * 40,
        "input_schema_version": "phase3b-input-v1",
        "data_source_contract_version": "point-in-time-v1",
        "built_at": "2029-12-01T12:00:00+00:00",
    }


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

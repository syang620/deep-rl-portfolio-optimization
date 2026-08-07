from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from portfolio_rl.features.feature_spec import FeatureSpec
from portfolio_rl.phase3b import frozen_candidate_loader
from portfolio_rl.phase3b.frozen_candidate_loader import load_frozen_candidate_runtime
from portfolio_rl.phase3b.governance import (
    EXPECTED_CANDIDATE_MANIFEST_SHA256,
    EXPECTED_MODEL_VERSION,
    GovernanceError,
    VerifiedCandidate,
    sha256_file,
)
from portfolio_rl.training.pretest_freeze import FrozenCandidate

SEEDS = (7, 42, 101, 202, 999)
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


def test_loader_preserves_exact_member_order_and_cpu_device(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    verified = _candidate_fixture(tmp_path)
    calls: list[tuple[str, float, str]] = []
    monkeypatch.setattr(
        frozen_candidate_loader,
        "verify_candidate",
        lambda *args, **kwargs: verified,
    )
    monkeypatch.setattr(
        frozen_candidate_loader,
        "load_sb3_weight_policy",
        lambda path, action_temperature, device: (
            calls.append((Path(path).name, action_temperature, device)) or object()
        ),
    )

    runtime = load_frozen_candidate_runtime(
        repository_root=tmp_path,
        frozen_candidate_path=verified.frozen_candidate_path,
    )

    assert tuple(seed for seed, _ in runtime.member_policies) == SEEDS
    assert calls == [(f"{seed}.zip", 0.5, "cpu") for seed in SEEDS]


def test_loader_rejects_reordered_member_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    verified = _candidate_fixture(tmp_path)
    member_path = verified.frozen_candidate_path.parent / "member_models.json"
    payload = json.loads(member_path.read_text(encoding="utf-8"))
    payload["members"].reverse()
    _write_json(member_path, payload)
    monkeypatch.setattr(
        frozen_candidate_loader,
        "verify_candidate",
        lambda *args, **kwargs: verified,
    )

    with pytest.raises(GovernanceError, match="frozen seed order"):
        load_frozen_candidate_runtime(
            repository_root=tmp_path,
            frozen_candidate_path=verified.frozen_candidate_path,
        )


def _candidate_fixture(root: Path) -> VerifiedCandidate:
    package = root / "candidate"
    model_dir = root / "models"
    package.mkdir()
    model_dir.mkdir()
    feature_spec_path = model_dir / "feature_spec_v1.json"
    feature_spec = FeatureSpec(
        feature_version="v1",
        asset_order=list(ASSETS),
        per_asset_features=[],
        global_features=[],
        current_weight_features=[],
        observation_dim=316,
        created_at="fixture",
    )
    _write_json(feature_spec_path, asdict(feature_spec))
    feature_hash = sha256_file(feature_spec_path)
    candidate = FrozenCandidate(
        model_version=EXPECTED_MODEL_VERSION,
        member_seeds=SEEDS,
        member_model_paths=tuple(Path(f"models/{seed}.zip") for seed in SEEDS),
        member_model_hashes=tuple(str(seed).zfill(64) for seed in SEEDS),
        action_temperatures=(0.5,) * 5,
        partial_rebalance_alpha=0.25,
        initial_portfolio="equal_weight",
        asset_order=ASSETS,
        feature_version="v1",
        feature_spec_hash=feature_hash,
        environment_config_hash="8" * 64,
        transaction_cost_bps=10.0,
        rebalance_frequency_trading_days=5,
    )
    frozen_path = package / "frozen_candidate.json"
    _write_json(frozen_path, {"observation_dimension": 316})
    _write_json(
        package / "member_models.json",
        {
            "member_seed_order": list(SEEDS),
            "members": [{"seed": seed} for seed in SEEDS],
        },
    )
    _write_json(package / "baseline_definitions.json", _baselines())
    manifest_path = package / "freeze_manifest.json"
    manifest_path.write_text("{}\n", encoding="utf-8")
    return VerifiedCandidate(
        candidate=candidate,
        frozen_candidate_path=frozen_path,
        freeze_manifest_path=manifest_path,
        freeze_manifest_sha256=EXPECTED_CANDIDATE_MANIFEST_SHA256,
    )


def _baselines() -> dict[str, object]:
    return {
        "schema_version": 1,
        "primary_hurdle": "equal_weight_weekly",
        "definitions": {
            "equal_weight_weekly": {},
            "buy_and_hold_equal_weight": {},
            "inverse_volatility": {
                "lookback_trading_days": 63,
                "volatility_floor": 1e-8,
            },
            "momentum_63d_top3_equal_weight": {
                "lookback_trading_days": 63,
                "top_k": 3,
            },
            "spy_only": {},
            "shy_only": {},
        },
    }


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

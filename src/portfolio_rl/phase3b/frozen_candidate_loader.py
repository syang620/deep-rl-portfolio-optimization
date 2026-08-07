"""Hash-verified deterministic runtime for the frozen Phase 3 candidate."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from portfolio_rl.features.feature_spec import FeatureSpec, load_feature_spec
from portfolio_rl.phase3b.governance import (
    EXPECTED_CANDIDATE_MANIFEST_SHA256,
    EXPECTED_MODEL_VERSION,
    GovernanceError,
    logical_json_sha256,
    read_json,
    relative_path,
    resolve_path,
    sha256_file,
    verify_candidate,
)
from portfolio_rl.policies.baseline_policies import WeightPolicy
from portfolio_rl.policies.sb3_policy import load_sb3_weight_policy
from portfolio_rl.training.pretest_freeze import FrozenCandidate


@dataclass(frozen=True)
class FrozenCandidateRuntime:
    """Fresh, ordered member policies plus verified serving contracts."""

    candidate: FrozenCandidate
    member_policies: tuple[tuple[int, WeightPolicy], ...]
    feature_spec: FeatureSpec
    baseline_definitions: dict[str, object]
    frozen_candidate_path: Path
    candidate_manifest_sha256: str
    baseline_definitions_sha256: str
    feature_spec_path: Path


def load_frozen_candidate_runtime(
    *,
    repository_root: Path,
    frozen_candidate_path: Path,
    expected_manifest_sha256: str = EXPECTED_CANDIDATE_MANIFEST_SHA256,
    device: str = "cpu",
) -> FrozenCandidateRuntime:
    """Verify the PR 19 package and load fresh members in exact seed order."""
    if device != "cpu":
        raise GovernanceError("Phase 3B candidate inference must use CPU")
    root = repository_root.resolve()
    verified = verify_candidate(root, frozen_candidate_path, expected_manifest_sha256)
    if verified.candidate.model_version != EXPECTED_MODEL_VERSION:
        raise GovernanceError("unexpected frozen candidate model version")

    package_dir = verified.frozen_candidate_path.parent
    members_payload = read_json(package_dir / "member_models.json")
    members = members_payload.get("members")
    if not isinstance(members, list) or len(members) != 5:
        raise GovernanceError("frozen candidate must contain five members")
    expected_seeds = list(verified.candidate.member_seeds)
    if members_payload.get("member_seed_order") != expected_seeds:
        raise GovernanceError("frozen candidate member seed order mismatch")
    if [record.get("seed") for record in members] != expected_seeds:
        raise GovernanceError("member records are not in frozen seed order")

    feature_spec_path = resolve_path(
        root,
        Path(verified.candidate.member_model_paths[0]).parent / "feature_spec_v1.json",
    )
    if sha256_file(feature_spec_path) != verified.candidate.feature_spec_hash:
        raise GovernanceError("serving feature specification hash mismatch")
    feature_spec = load_feature_spec(feature_spec_path)
    if tuple(feature_spec.asset_order) != tuple(verified.candidate.asset_order):
        raise GovernanceError("serving feature asset order mismatch")
    if feature_spec.feature_version != verified.candidate.feature_version:
        raise GovernanceError("serving feature version mismatch")
    candidate_payload = read_json(verified.frozen_candidate_path)
    if feature_spec.observation_dim != candidate_payload.get("observation_dimension"):
        raise GovernanceError("serving observation dimension mismatch")

    baseline_path = package_dir / "baseline_definitions.json"
    baseline_definitions = read_json(baseline_path)
    _validate_baseline_definitions(baseline_definitions)

    policies: list[tuple[int, WeightPolicy]] = []
    for seed, model_path, temperature in zip(
        verified.candidate.member_seeds,
        verified.candidate.member_model_paths,
        verified.candidate.action_temperatures,
        strict=True,
    ):
        resolved_model = resolve_path(root, model_path)
        policies.append(
            (
                seed,
                load_sb3_weight_policy(
                    resolved_model,
                    action_temperature=temperature,
                    device=device,
                ),
            )
        )
    return FrozenCandidateRuntime(
        candidate=verified.candidate,
        member_policies=tuple(policies),
        feature_spec=feature_spec,
        baseline_definitions=baseline_definitions,
        frozen_candidate_path=verified.frozen_candidate_path,
        candidate_manifest_sha256=verified.freeze_manifest_sha256,
        baseline_definitions_sha256=sha256_file(baseline_path),
        feature_spec_path=feature_spec_path,
    )


def runtime_identity_payload(
    runtime: FrozenCandidateRuntime, repository_root: Path
) -> dict[str, object]:
    """Return the stable candidate identity recorded in each recommendation."""
    root = repository_root.resolve()
    return {
        "model_version": runtime.candidate.model_version,
        "candidate_manifest_sha256": runtime.candidate_manifest_sha256,
        "frozen_candidate_path": relative_path(root, runtime.frozen_candidate_path),
        "member_seed_order": list(runtime.candidate.member_seeds),
        "member_model_hashes": list(runtime.candidate.member_model_hashes),
        "action_temperatures": list(runtime.candidate.action_temperatures),
        "partial_rebalance_alpha": runtime.candidate.partial_rebalance_alpha,
        "feature_spec_path": relative_path(root, runtime.feature_spec_path),
        "feature_spec_sha256": runtime.candidate.feature_spec_hash,
        "baseline_definitions_sha256": runtime.baseline_definitions_sha256,
        "baseline_definitions_logical_sha256": logical_json_sha256(
            runtime.baseline_definitions
        ),
    }


def _validate_baseline_definitions(payload: dict[str, object]) -> None:
    if payload.get("schema_version") != 1:
        raise GovernanceError("unsupported baseline definitions schema")
    if payload.get("primary_hurdle") != "equal_weight_weekly":
        raise GovernanceError("primary baseline must be weekly equal weight")
    definitions = payload.get("definitions")
    if not isinstance(definitions, dict):
        raise GovernanceError("baseline definitions must be a mapping")
    expected = {
        "equal_weight_weekly",
        "buy_and_hold_equal_weight",
        "inverse_volatility",
        "momentum_63d_top3_equal_weight",
        "spy_only",
        "shy_only",
    }
    if set(definitions) != expected:
        raise GovernanceError("frozen baseline set mismatch")
    inverse = definitions["inverse_volatility"]
    momentum = definitions["momentum_63d_top3_equal_weight"]
    if not isinstance(inverse, dict) or not isinstance(momentum, dict):
        raise GovernanceError("frozen baseline definitions are malformed")
    if inverse.get("lookback_trading_days") != 63:
        raise GovernanceError("inverse-volatility lookback must equal 63")
    if inverse.get("volatility_floor") != 1e-8:
        raise GovernanceError("inverse-volatility floor mismatch")
    if momentum.get("lookback_trading_days") != 63 or momentum.get("top_k") != 3:
        raise GovernanceError("momentum baseline contract mismatch")

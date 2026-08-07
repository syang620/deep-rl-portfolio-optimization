"""Certification-only publication and cycle recording."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from portfolio_rl.phase3b.certification import (
    REQUIRED_CYCLE_ARTIFACTS,
    REQUIRED_CYCLE_CHECKS,
)
from portfolio_rl.phase3b.execution import ExecutionConfig
from portfolio_rl.phase3b.frozen_candidate_loader import FrozenCandidateRuntime
from portfolio_rl.phase3b.governance import (
    GovernanceError,
    canonical_json_bytes,
    logical_json_sha256,
    read_json,
)
from portfolio_rl.phase3b.shadow_runner import ShadowDecision
from portfolio_rl.phase3b.signatures import (
    create_signature_record,
    verify_signature_record,
)

CERTIFICATION_RECOMMENDATION_NAMESPACE = (
    "portfolio-rl-phase3b-certification-recommendation-v1"
)


def publish_certification_recommendation(
    *,
    decision: ShadowDecision,
    runtime: FrozenCandidateRuntime,
    execution_config: ExecutionConfig,
    certification_id: str,
    certification_identity_sha256: str,
    output_root: Path,
    service_private_key_path: Path,
    service_public_key_path: Path,
    service_principal: str,
    signed_at: datetime,
) -> Path:
    """Publish a create-only target envelope that cannot register a holdout."""
    payload = {
        "schema_version": 1,
        "artifact_type": "phase3b_certification_recommendation",
        "certification_id": certification_id,
        "certification_identity_sha256": certification_identity_sha256,
        "decision_date": decision.decision_date.isoformat(),
        "execution_date": decision.snapshot.next_trading_date.isoformat(),
        "generated_at": decision.generated_at.isoformat(),
        "snapshot_sha256": decision.snapshot.snapshot_sha256,
        "state_sha256": decision.live_state.state_sha256,
        "candidate_manifest_sha256": runtime.candidate_manifest_sha256,
        "execution_config_sha256": execution_config.config_sha256,
        "member_seed_order": [row.seed for row in decision.member_targets],
        "member_targets": [asdict(row) for row in decision.member_targets],
        "ensemble_target": list(decision.ensemble_target),
        "executed_target": list(decision.executed_target),
        "baseline_targets": {
            strategy: list(weights)
            for strategy, weights in sorted(decision.baseline_targets.items())
        },
        "execution_instructions": [
            {
                **asdict(instruction),
                "decision_date": instruction.decision_date.isoformat(),
                "execution_date": instruction.execution_date.isoformat(),
            }
            for instruction in decision.execution_instructions
        ],
        "performance_computed": False,
        "canonical_holdout_registered": False,
    }
    payload["recommendation_payload_sha256"] = logical_json_sha256(payload)
    envelope = {
        "recommendation": payload,
        "signature_record": create_signature_record(
            payload=payload,
            payload_path="recommendation.json",
            artifact_type="certification_recommendation",
            role="service_signing",
            principal=service_principal,
            namespace=CERTIFICATION_RECOMMENDATION_NAMESPACE,
            private_key_path=service_private_key_path,
            public_key_path=service_public_key_path,
            signed_at=signed_at,
        ),
    }
    destination = (
        output_root.resolve()
        / certification_id
        / "decisions"
        / decision.decision_date.isoformat()
        / "recommendation.json"
    )
    if destination.exists():
        if read_json(destination) != envelope:
            raise GovernanceError("signed certification target is immutable")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(canonical_json_bytes(envelope))
    return destination


def verify_certification_recommendation(
    *,
    path: Path,
    service_public_key_path: Path,
    expected_certification_id: str,
    expected_identity_sha256: str,
) -> dict[str, Any]:
    """Verify a certification target before one-close-delayed execution."""
    envelope = read_json(path)
    if set(envelope) != {"recommendation", "signature_record"}:
        raise GovernanceError("certification recommendation envelope mismatch")
    payload = envelope["recommendation"]
    if not isinstance(payload, dict):
        raise GovernanceError("certification recommendation is missing")
    unhashed = dict(payload)
    recorded = unhashed.pop("recommendation_payload_sha256", None)
    if logical_json_sha256(unhashed) != recorded:
        raise GovernanceError("certification recommendation hash mismatch")
    if payload.get("certification_id") != expected_certification_id:
        raise GovernanceError("certification recommendation ID mismatch")
    if payload.get("certification_identity_sha256") != expected_identity_sha256:
        raise GovernanceError("certification recommendation identity mismatch")
    if payload.get("canonical_holdout_registered") is not False:
        raise GovernanceError("certification artifact cannot register a holdout")
    verify_signature_record(
        payload=payload,
        record=envelope["signature_record"],
        public_key_path=service_public_key_path,
        expected_role="service_signing",
        expected_namespace=CERTIFICATION_RECOMMENDATION_NAMESPACE,
    )
    return payload


def executable_targets(
    payload: dict[str, Any], expected_asset_count: int
) -> dict[str, tuple[float, ...]]:
    """Extract the already signed targets without recomputing alpha at t+1."""
    candidate = tuple(float(value) for value in payload["executed_target"])
    baselines = payload.get("baseline_targets")
    if not isinstance(baselines, dict):
        raise GovernanceError("signed baseline targets are missing")
    targets = {
        "candidate": candidate,
        **{
            str(strategy): tuple(float(value) for value in values)
            for strategy, values in baselines.items()
        },
    }
    for strategy, values in targets.items():
        if len(values) != expected_asset_count:
            raise GovernanceError(f"signed target shape mismatch: {strategy}")
    return targets


def write_cycle_manifest(
    *,
    path: Path,
    certification_id: str,
    cycle_number: int,
    identity_sha256: str,
    decision_date: str,
    execution_date: str,
    artifact_bindings: dict[str, str],
    checks: dict[str, bool],
    official: bool,
    scheduled_decision_missed: bool,
) -> Path:
    """Record a complete cycle; partial checks cannot count as official."""
    if cycle_number not in {1, 2, 3, 4}:
        raise GovernanceError("certification cycle number must be between one and four")
    if set(checks) != set(REQUIRED_CYCLE_CHECKS):
        raise GovernanceError("certification cycle check inventory mismatch")
    if set(artifact_bindings) != set(REQUIRED_CYCLE_ARTIFACTS) or any(
        not _sha256(value) for value in artifact_bindings.values()
    ):
        raise GovernanceError("certification cycle artifact inventory mismatch")
    if official and (scheduled_decision_missed or not all(checks.values())):
        raise GovernanceError("failed certification work cannot count as official")
    payload = {
        "schema_version": 1,
        "certification_id": certification_id,
        "cycle_number": cycle_number,
        "identity_sha256": identity_sha256,
        "decision_date": decision_date,
        "execution_date": execution_date,
        "artifact_bindings": dict(sorted(artifact_bindings.items())),
        "checks": checks,
        "official": official,
        "scheduled_decision_missed": scheduled_decision_missed,
        "canonical_holdout_registered": False,
    }
    payload["cycle_manifest_sha256"] = logical_json_sha256(payload)
    if path.exists():
        if read_json(path) != payload:
            raise GovernanceError("certification cycle overwrite is forbidden")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(payload))
    return path


def _sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True

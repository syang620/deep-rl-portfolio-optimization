"""Official Phase 3B operational-certification governance."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any

from portfolio_rl.phase3b.execution import ExecutionConfig
from portfolio_rl.phase3b.governance import (
    ApprovalRole,
    GovernanceError,
    logical_json_sha256,
    read_json,
)
from portfolio_rl.phase3b.operational_metrics import OperationsConfig
from portfolio_rl.phase3b.signatures import verify_signature_record

CERTIFICATION_APPROVAL_NAMESPACE = "portfolio-rl-phase3b-certification-approval-v1"
REQUIRED_CYCLE_CHECKS = (
    "snapshot_validation",
    "candidate_and_baseline_generation",
    "signing_and_publication",
    "one_close_delayed_execution",
    "turnover_and_cost_reconciliation",
    "portfolio_state_roll_forward",
    "restart_replay_recovery",
    "incident_handling",
    "sealed_ledger_write",
    "operational_dashboard_filtering",
)
REQUIRED_CYCLE_ARTIFACTS = (
    "input_snapshot_sha256",
    "recommendation_sha256",
    "execution_price_snapshot_sha256",
    "execution_sha256",
    "operational_output_sha256",
    "sealed_ledger_entry_hash",
    "custodian_checkpoint_sha256",
    "restricted_state_sha256",
    "replay_verification_sha256",
    "incident_audit_sha256",
)


@dataclass(frozen=True)
class CertificationIdentity:
    """All approved identities whose change restarts official certification."""

    scaler_sha256: str
    service_signing_fingerprint: str
    container_image_digest: str
    runtime_git_commit: str
    feature_snapshot_schema_version: str
    execution_config_sha256: str
    asset_tier_cost_map_sha256: str
    operations_config_sha256: str

    @property
    def identity_sha256(self) -> str:
        return logical_json_sha256(asdict(self))


@dataclass(frozen=True)
class CertificationStatus:
    """Deterministically reconstructed certification progress."""

    official: bool
    valid: bool
    consecutive_completed_cycles: int
    required_cycles: int
    missed_scheduled_decisions: int
    identity_sha256: str
    next_cycle_number: int
    restart_reason: str | None


def build_certification_authorization_payload(
    *,
    certification_id: str,
    identity: CertificationIdentity,
    first_scheduled_decision_date: date,
) -> dict[str, Any]:
    """Build the single canonical claim independently approved by three roles."""
    return {
        "schema_version": 1,
        "authorization_type": "phase3b_official_certification",
        "certification_id": certification_id,
        "identity": asdict(identity),
        "identity_sha256": identity.identity_sha256,
        "first_scheduled_decision_date": first_scheduled_decision_date.isoformat(),
        "required_consecutive_cycles": 4,
        "canonical_holdout_registered": False,
    }


def verify_certification_authorization(
    *,
    payload: dict[str, Any],
    approval_records: dict[str, dict[str, Any]],
    approver_public_keys: dict[str, Path],
    execution_config: ExecutionConfig,
    operations_config: OperationsConfig,
) -> CertificationIdentity:
    """Require exact approved operational identities and independent signatures."""
    if execution_config.scaler_status != "approved_for_phase3b":
        raise GovernanceError("official certification cannot use a draft scaler")
    if operations_config.status != "approved":
        raise GovernanceError("official certification cannot use draft operations")
    if operations_config.sealing_approval_status != "approved_for_phase3b":
        raise GovernanceError("official certification requires approved ledger sealing")
    roles = tuple(role.value for role in ApprovalRole)
    if set(approval_records) != set(roles) or set(approver_public_keys) != set(roles):
        raise GovernanceError("certification requires all three approval roles")
    for role in roles:
        verify_signature_record(
            payload=payload,
            record=approval_records[role],
            public_key_path=approver_public_keys[role],
            expected_role=role,
            expected_namespace=CERTIFICATION_APPROVAL_NAMESPACE,
        )
    identity_payload = payload.get("identity")
    if not isinstance(identity_payload, dict):
        raise GovernanceError("certification identity is missing")
    try:
        identity = CertificationIdentity(**identity_payload)
    except TypeError as exc:
        raise GovernanceError("certification identity schema mismatch") from exc
    if payload.get("identity_sha256") != identity.identity_sha256:
        raise GovernanceError("certification identity hash mismatch")
    if identity.scaler_sha256 != execution_config.normalization_artifact_sha256:
        raise GovernanceError("certification scaler differs from execution config")
    if identity.service_signing_fingerprint != execution_config.signing.public_key_fingerprint:
        raise GovernanceError("certification signing identity differs from execution config")
    if identity.feature_snapshot_schema_version != execution_config.feature_payload_schema_version:
        raise GovernanceError("certification snapshot schema differs from execution config")
    if identity.execution_config_sha256 != execution_config.config_sha256:
        raise GovernanceError("certification execution config hash mismatch")
    if identity.operations_config_sha256 != operations_config.config_sha256:
        raise GovernanceError("certification operations config hash mismatch")
    if identity.asset_tier_cost_map_sha256 != logical_json_sha256(
        dict(execution_config.asset_cost_bps)
    ):
        raise GovernanceError("certification asset-tier cost map mismatch")
    if payload.get("canonical_holdout_registered") is not False:
        raise GovernanceError("PR 22 cannot register a canonical holdout")
    return identity


def reconstruct_certification_status(
    *,
    cycle_manifest_paths: list[Path],
    expected_identity_sha256: str,
    official: bool,
    required_cycles: int = 4,
) -> CertificationStatus:
    """Rebuild progress; an identity change resets the consecutive count."""
    consecutive = 0
    missed = 0
    restart_reason: str | None = None
    for path in sorted(cycle_manifest_paths):
        cycle = read_json(path)
        if cycle.get("official") is not official:
            raise GovernanceError("mixed development and official certification cycles")
        if cycle.get("scheduled_decision_missed") is True:
            missed += 1
            consecutive = 0
            restart_reason = "scheduled_decision_missed"
            continue
        if cycle.get("identity_sha256") != expected_identity_sha256:
            consecutive = 0
            restart_reason = "approved_identity_changed"
            continue
        checks = cycle.get("checks")
        if not isinstance(checks, dict) or set(checks) != set(REQUIRED_CYCLE_CHECKS):
            raise GovernanceError("certification cycle check inventory mismatch")
        if not all(value is True for value in checks.values()):
            consecutive = 0
            restart_reason = "cycle_check_failed"
            continue
        consecutive += 1
    valid = missed <= 2 and consecutive >= required_cycles
    return CertificationStatus(
        official=official,
        valid=valid,
        consecutive_completed_cycles=consecutive,
        required_cycles=required_cycles,
        missed_scheduled_decisions=missed,
        identity_sha256=expected_identity_sha256,
        next_cycle_number=consecutive + 1,
        restart_reason=restart_reason,
    )


def require_certification_complete(status: CertificationStatus) -> None:
    """Block registration unless four official cycles remain valid."""
    if not status.official or not status.valid:
        raise GovernanceError("four consecutive official certification cycles are required")

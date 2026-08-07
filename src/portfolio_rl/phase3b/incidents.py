"""Frozen incident classifications for the Phase 3B governance ledger."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from portfolio_rl.phase3b.governance import (
    GovernanceError,
    canonical_json_bytes,
    logical_json_sha256,
    read_json,
)


class IncidentSeverity(StrEnum):
    """Severity classes used by Phase 3B governance."""

    FATAL = "fatal"
    NONFATAL = "nonfatal"


FATAL_INCIDENTS = (
    "candidate_model_hash_mismatch",
    "wrong_seed_order",
    "wrong_partial_rebalance_alpha",
    "feature_spec_mismatch",
    "wrong_container_digest",
    "future_data_access",
    "recommendation_after_cutoff",
    "manual_recommendation_alteration",
    "candidate_baseline_snapshot_mismatch",
    "early_access_to_sealed_performance",
    "ledger_deletion_or_unverifiable_mutation",
    "unapproved_universe_change",
    "candidate_retraining_or_replacement",
    "more_than_two_missed_rebalances",
    "prolonged_required_data_failure",
)

NONFATAL_INCIDENTS = (
    "delayed_operational_dashboard_refresh",
    "deterministic_process_retry",
    "nonmaterial_vendor_correction",
    "one_isolated_missed_rebalance",
    "transient_infrastructure_outage_with_deterministic_recovery",
)


def incident_policy_payload() -> dict[str, object]:
    """Return the immutable incident-classification and approval contract."""
    return {
        "schema_version": 1,
        "fatal_incidents": list(FATAL_INCIDENTS),
        "nonfatal_incidents": list(NONFATAL_INCIDENTS),
        "approval_rules": {
            "holdout_registration": "unanimous_three_of_three",
            "pre_start_registration_change": "unanimous_three_of_three",
            "performance_unseal": "unanimous_three_of_three",
            "fatal_incident_disposition": "unanimous_three_of_three",
            "final_outcome": "unanimous_three_of_three",
            "routine_incident_disposition": {
                "minimum_approvals": 2,
                "independent_reviewer_required": True,
            },
        },
        "required_incident_fields": [
            "incident_id",
            "timestamp",
            "severity",
            "affected_decisions",
            "root_cause",
            "remediation",
            "hash_evidence",
            "approver_disposition",
        ],
    }


def record_no_trade_incident(
    *,
    path: Path,
    incident_id: str,
    incident_type: str,
    affected_decisions: list[str],
    root_cause: str,
    hash_evidence: dict[str, str],
    timestamp: datetime,
) -> dict[str, object]:
    """Append one immutable no-trade incident record."""
    if timestamp.tzinfo is None:
        raise GovernanceError("incident timestamp must be timezone-aware")
    severity = (
        IncidentSeverity.FATAL.value
        if incident_type in FATAL_INCIDENTS
        else IncidentSeverity.NONFATAL.value
    )
    payload: dict[str, object] = {
        "schema_version": 1,
        "incident_id": incident_id,
        "timestamp": timestamp.astimezone(UTC).isoformat(),
        "incident_type": incident_type,
        "severity": severity,
        "decision_action": "no_trade",
        "affected_decisions": affected_decisions,
        "root_cause": root_cause,
        "remediation": "pending_approved_disposition",
        "hash_evidence": hash_evidence,
        "approver_disposition": None,
    }
    payload["incident_sha256"] = logical_json_sha256(payload)
    if path.exists():
        if read_json(path) != payload:
            raise GovernanceError("incident record overwrite is forbidden")
        return payload
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(payload))
    return payload


def reject_and_log_unseal_attempt(
    *,
    audit_path: Path,
    requester: str,
    reason: str,
    timestamp: datetime,
) -> None:
    """Deny PR 22 performance access and leave an immutable audit record."""
    record_no_trade_incident(
        path=audit_path,
        incident_id=f"unauthorized-unseal-{timestamp.astimezone(UTC).isoformat()}",
        incident_type="early_access_to_sealed_performance",
        affected_decisions=[],
        root_cause=f"requester={requester}; reason={reason}",
        hash_evidence={},
        timestamp=timestamp,
    )
    raise GovernanceError("performance unsealing is not authorized in PR 22")

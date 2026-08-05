"""Frozen incident classifications for the Phase 3B governance ledger."""

from __future__ import annotations

from enum import StrEnum


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
                "independent_model_risk_required": True,
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

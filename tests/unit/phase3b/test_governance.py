from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta

import pytest

from portfolio_rl.phase3b.governance import (
    GovernanceError,
    logical_json_sha256,
    validate_certification,
    validate_container_identity,
    validate_schedule,
)
from portfolio_rl.phase3b.incidents import incident_policy_payload


def test_container_identity_requires_immutable_digest() -> None:
    payload = _container_identity()
    payload["image_digest"] = "phase3b:latest"

    with pytest.raises(GovernanceError, match="immutable OCI digest"):
        validate_container_identity(payload, "a" * 40)


def test_schedule_enforces_fixed_twelve_month_horizon() -> None:
    schedule, start, end = _schedule()

    summary = validate_schedule(
        schedule,
        start_decision_date=start,
        end_decision_date=end,
        minimum_decisions=50,
    )

    assert len(summary["holdout_decision_dates"]) >= 50
    assert summary["final_holding_period_end_date"] > end.isoformat()


def test_schedule_rejects_fewer_than_fifty_decisions() -> None:
    schedule, start, _ = _schedule()
    schedule["holdout_decision_dates"] = schedule["holdout_decision_dates"][:49]
    end = date.fromisoformat(schedule["holdout_decision_dates"][-1])
    sessions = [date.fromisoformat(item) for item in schedule["trading_sessions"]]
    end_position = sessions.index(end)
    schedule["final_holding_period_end_date"] = sessions[end_position + 5].isoformat()

    with pytest.raises(GovernanceError, match="fewer than 50"):
        validate_schedule(
            schedule,
            start_decision_date=start,
            end_decision_date=end,
            minimum_decisions=50,
        )


def test_certification_rejects_performance_computation() -> None:
    payload = _certification()
    payload["performance_metrics_computed"] = True
    payload["manifest_payload_sha256"] = logical_json_sha256(
        {
            key: value
            for key, value in payload.items()
            if key != "manifest_payload_sha256"
        }
    )

    with pytest.raises(GovernanceError, match="must not compute"):
        validate_certification(
            payload,
            candidate_manifest_sha256="b" * 64,
            container_digest=f"sha256:{'c' * 64}",
            git_commit="a" * 40,
            schedule_sha256="d" * 64,
            certification_dates=payload["completed_decision_dates"],
            frozen_config_hashes={
                "candidate_acceptance": "a" * 64,
                "execution": "e" * 64,
                "operations": "o" * 64,
            },
        )


def test_incident_policy_requires_independent_risk_for_routine_incidents() -> None:
    policy = incident_policy_payload()

    routine = policy["approval_rules"]["routine_incident_disposition"]
    assert routine == {
        "minimum_approvals": 2,
        "independent_model_risk_required": True,
    }
    assert "future_data_access" in policy["fatal_incidents"]


def _container_identity() -> dict[str, object]:
    return {
        "schema_version": 1,
        "image_reference": "registry.example/portfolio-rl@sha256:" + "c" * 64,
        "image_digest": "sha256:" + "c" * 64,
        "git_commit": "a" * 40,
        "input_schema_version": "phase3b-input-v1",
        "data_source_contract_version": "point-in-time-v1",
        "built_at": "2029-12-01T12:00:00+00:00",
    }


def _schedule() -> tuple[dict[str, object], date, date]:
    sessions = []
    cursor = date(2029, 12, 3)
    while len(sessions) < 330:
        if cursor.weekday() < 5:
            sessions.append(cursor)
        cursor += timedelta(days=1)
    certification = sessions[:20:5]
    start_position = 20
    start = sessions[start_position]
    anniversary = start.replace(year=start.year + 1)
    decision_positions = []
    position = start_position
    while sessions[position] <= anniversary:
        decision_positions.append(position)
        position += 5
    decisions = [sessions[index] for index in decision_positions]
    end = decisions[-1]
    final_end = sessions[decision_positions[-1] + 5]
    return (
        {
            "schema_version": 1,
            "schedule_id": "schedule-fixture",
            "timezone": "America/New_York",
            "trading_sessions": [item.isoformat() for item in sessions],
            "certification_decision_dates": [
                item.isoformat() for item in certification
            ],
            "holdout_decision_dates": [item.isoformat() for item in decisions],
            "final_holding_period_end_date": final_end.isoformat(),
            "final_holding_period_complete_at_utc": datetime(
                final_end.year, final_end.month, final_end.day, 22, tzinfo=UTC
            ).isoformat(),
        },
        start,
        end,
    )


def _certification() -> dict[str, object]:
    payload = {
        "schema_version": 1,
        "certification_id": "cert-fixture",
        "status": "passed",
        "cycle_count": 4,
        "completed_decision_dates": [
            "2029-12-03",
            "2029-12-10",
            "2029-12-17",
            "2029-12-24",
        ],
        "candidate_manifest_sha256": "b" * 64,
        "container_image_digest": f"sha256:{'c' * 64}",
        "git_commit": "a" * 40,
        "schedule_sha256": "d" * 64,
        "frozen_config_hashes": {
            "candidate_acceptance": "a" * 64,
            "execution": "e" * 64,
            "operations": "o" * 64,
        },
        "performance_metrics_computed": False,
        "certification_completed_at": "2029-12-28T12:00:00+00:00",
    }
    payload["manifest_payload_sha256"] = logical_json_sha256(payload)
    return json.loads(json.dumps(payload))

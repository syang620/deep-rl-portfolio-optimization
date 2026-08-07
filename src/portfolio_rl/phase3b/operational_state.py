"""Restricted, signed Phase 3B restart state."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from portfolio_rl.phase3b.close_processor import StrategyCloseState
from portfolio_rl.phase3b.governance import (
    GovernanceError,
    canonical_json_bytes,
    logical_json_sha256,
    read_json,
)
from portfolio_rl.phase3b.signatures import (
    create_signature_record,
    verify_signature_record,
)
from portfolio_rl.phase3b.snapshot_chain import STRATEGIES

STATE_SIGNATURE_NAMESPACE = "portfolio-rl-phase3b-operational-state-v1"


def write_restricted_state(
    *,
    path: Path,
    context_type: str,
    context_id: str,
    as_of_date: str,
    asset_order: tuple[str, ...],
    states: dict[str, StrategyCloseState],
    previous_result_sha256: str,
    ledger_tip_hash: str,
    service_private_key_path: Path,
    service_public_key_path: Path,
    service_principal: str,
    signed_at: datetime,
) -> Path:
    """Write a create-only state checkpoint excluded from operational views."""
    if set(states) != set(STRATEGIES):
        raise GovernanceError("restricted state strategy set is incomplete")
    payload = {
        "schema_version": 1,
        "visibility": "internal_restricted_not_dashboard_safe",
        "context_type": context_type,
        "context_id": context_id,
        "as_of_date": as_of_date,
        "asset_order": list(asset_order),
        "strategies": {
            strategy: {
                "weights": list(states[strategy].weights),
                "nav": states[strategy].nav,
                "peak_nav": states[strategy].peak_nav,
            }
            for strategy in STRATEGIES
        },
        "previous_result_sha256": previous_result_sha256,
        "ledger_tip_hash": ledger_tip_hash,
    }
    payload["state_payload_sha256"] = logical_json_sha256(payload)
    envelope = {
        "state": payload,
        "signature_record": create_signature_record(
            payload=payload,
            payload_path=path.name,
            artifact_type="restricted_operational_state",
            role="service_signing",
            principal=service_principal,
            namespace=STATE_SIGNATURE_NAMESPACE,
            private_key_path=service_private_key_path,
            public_key_path=service_public_key_path,
            signed_at=signed_at,
        ),
    }
    if path.exists():
        if read_json(path) != envelope:
            raise GovernanceError("restricted state overwrite is forbidden")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(envelope))
    path.chmod(0o600)
    return path


def load_restricted_state(
    *, path: Path, service_public_key_path: Path
) -> tuple[tuple[str, ...], dict[str, StrategyCloseState], dict[str, Any]]:
    """Verify and restore exact strategy state after restart."""
    envelope = read_json(path)
    if set(envelope) != {"state", "signature_record"}:
        raise GovernanceError("restricted state envelope mismatch")
    payload = envelope["state"]
    if not isinstance(payload, dict):
        raise GovernanceError("restricted state payload is missing")
    unhashed = dict(payload)
    recorded = unhashed.pop("state_payload_sha256", None)
    if logical_json_sha256(unhashed) != recorded:
        raise GovernanceError("restricted state payload hash mismatch")
    verify_signature_record(
        payload=payload,
        record=envelope["signature_record"],
        public_key_path=service_public_key_path,
        expected_role="service_signing",
        expected_namespace=STATE_SIGNATURE_NAMESPACE,
    )
    raw = payload.get("strategies")
    if not isinstance(raw, dict) or set(raw) != set(STRATEGIES):
        raise GovernanceError("restricted state strategy set is incomplete")
    states = {
        strategy: StrategyCloseState(
            weights=tuple(float(value) for value in raw[strategy]["weights"]),
            nav=float(raw[strategy]["nav"]),
            peak_nav=float(raw[strategy]["peak_nav"]),
        )
        for strategy in STRATEGIES
    }
    return tuple(payload["asset_order"]), states, payload

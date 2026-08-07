"""Phase 3B operations configuration and safe operational projections."""

from __future__ import annotations

import base64
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from portfolio_rl.config.loader import load_universe_config
from portfolio_rl.phase3b.governance import (
    GovernanceError,
    read_yaml,
    resolve_path,
    sha256_file,
)

SEALING_ALGORITHM = "pynacl_sealed_box_curve25519xsalsa20poly1305"


@dataclass(frozen=True)
class OperationsConfig:
    """Frozen operational, sealing, and visibility rules."""

    config_path: Path
    config_sha256: str
    status: str
    required_consecutive_cycles: int
    rebalance_frequency_trading_days: int
    maximum_missed_scheduled_decisions: int
    close_snapshot_schema_version: str
    price_field: str
    sealing_public_key_path: Path
    sealing_public_key_sha256: str
    sealing_key_fingerprint: str
    sealing_approval_status: str
    universe_config_path: Path
    universe_config_sha256: str
    asset_classes: dict[str, str]
    allowed_weight_types: tuple[str, ...]
    forbidden_field_tokens: tuple[str, ...]
    development_output_root: Path
    certification_output_root: Path
    holdout_output_root: Path


def load_operations_config(
    path: Path,
    *,
    repository_root: Path,
    require_approved: bool = True,
) -> OperationsConfig:
    """Load operations config and fail closed on unresolved official identities."""
    root = repository_root.resolve()
    config_path = resolve_path(root, path)
    payload = read_yaml(config_path)
    _keys(
        payload,
        {
            "schema_version",
            "status",
            "certification",
            "daily_close",
            "sealed_ledger",
            "universe",
            "operational_visibility",
            "output_roots",
        },
        "operations config",
    )
    if payload["schema_version"] != 1:
        raise GovernanceError("unsupported operations config schema")
    status = str(payload["status"])
    if require_approved and status != "approved":
        raise GovernanceError("Phase 3B operations config is still draft")
    if status not in {"draft", "approved"}:
        raise GovernanceError("invalid operations config status")

    certification = payload["certification"]
    _keys(
        certification,
        {
            "required_consecutive_cycles",
            "rebalance_frequency_trading_days",
            "maximum_missed_scheduled_decisions",
            "official_authorization_required",
        },
        "certification operations",
    )
    if certification != {
        "required_consecutive_cycles": 4,
        "rebalance_frequency_trading_days": 5,
        "maximum_missed_scheduled_decisions": 2,
        "official_authorization_required": True,
    }:
        raise GovernanceError("certification operations contract mismatch")

    close = payload["daily_close"]
    _keys(
        close,
        {
            "timezone",
            "price_field",
            "snapshot_schema_version",
            "require_service_signature",
        },
        "daily-close operations",
    )
    if close["timezone"] != "America/New_York":
        raise GovernanceError("daily-close timezone mismatch")
    if close["price_field"] != "adjusted_close_total_return_proxy":
        raise GovernanceError("daily-close price field mismatch")
    if close["require_service_signature"] is not True:
        raise GovernanceError("daily-close snapshots must be service-signed")

    sealing = payload["sealed_ledger"]
    _keys(
        sealing,
        {
            "schema_version",
            "algorithm",
            "recipient_public_key_path",
            "recipient_public_key_sha256",
            "recipient_key_fingerprint",
            "approval_status",
            "require_daily_custodian_checkpoint",
        },
        "sealed-ledger operations",
    )
    if sealing["algorithm"] != SEALING_ALGORITHM:
        raise GovernanceError("sealed-ledger algorithm mismatch")
    if sealing["require_daily_custodian_checkpoint"] is not True:
        raise GovernanceError("daily custodian checkpoints are required")
    if require_approved and sealing["approval_status"] != "approved_for_phase3b":
        raise GovernanceError("performance-sealing recipient is not approved")
    key_path = resolve_path(root, str(sealing["recipient_public_key_path"]))
    key_sha = _sha(sealing["recipient_public_key_sha256"], "sealing public key")
    if sha256_file(key_path) != key_sha:
        raise GovernanceError("performance-sealing public key hash mismatch")
    key_bytes = decode_sealing_public_key(key_path)
    fingerprint = sealing_key_fingerprint(key_bytes)
    if fingerprint != sealing["recipient_key_fingerprint"]:
        raise GovernanceError("performance-sealing key fingerprint mismatch")

    universe = payload["universe"]
    _keys(universe, {"config_path", "config_sha256"}, "operations universe")
    universe_path = resolve_path(root, str(universe["config_path"]))
    universe_sha = _sha(universe["config_sha256"], "universe config")
    if sha256_file(universe_path) != universe_sha:
        raise GovernanceError("operations universe config hash mismatch")
    universe_config = load_universe_config(universe_path)
    asset_classes = {
        asset.ticker: asset.asset_class for asset in universe_config.assets
    }

    visibility = payload["operational_visibility"]
    _keys(
        visibility,
        {"allowed_weight_types", "forbidden_field_tokens"},
        "operational visibility",
    )
    allowed = _string_tuple(visibility["allowed_weight_types"], "allowed weights")
    forbidden = _string_tuple(
        visibility["forbidden_field_tokens"], "forbidden fields"
    )
    outputs = payload["output_roots"]
    _keys(outputs, {"development", "certification", "holdout"}, "output roots")
    return OperationsConfig(
        config_path=config_path,
        config_sha256=sha256_file(config_path),
        status=status,
        required_consecutive_cycles=4,
        rebalance_frequency_trading_days=5,
        maximum_missed_scheduled_decisions=2,
        close_snapshot_schema_version=str(close["snapshot_schema_version"]),
        price_field=str(close["price_field"]),
        sealing_public_key_path=key_path,
        sealing_public_key_sha256=key_sha,
        sealing_key_fingerprint=fingerprint,
        sealing_approval_status=str(sealing["approval_status"]),
        universe_config_path=universe_path,
        universe_config_sha256=universe_sha,
        asset_classes=asset_classes,
        allowed_weight_types=allowed,
        forbidden_field_tokens=forbidden,
        development_output_root=resolve_path(root, str(outputs["development"])),
        certification_output_root=resolve_path(root, str(outputs["certification"])),
        holdout_output_root=resolve_path(root, str(outputs["holdout"])),
    )


def build_operational_snapshot(
    *,
    context_type: str,
    context_id: str,
    close_date: str,
    status: str,
    snapshot_timestamp: str,
    snapshot_sha256: str,
    identity_hash_status: dict[str, bool],
    targets: list[dict[str, object]],
    executions: list[dict[str, object]],
    incidents: list[dict[str, object]],
    asset_classes: dict[str, str],
    forbidden_field_tokens: tuple[str, ...],
) -> dict[str, object]:
    """Project execution data into a performance-free operational view."""
    exposure_rows = _exposures(executions, asset_classes)
    concentration = _concentration(executions)
    payload: dict[str, object] = {
        "schema_version": 1,
        "context_type": context_type,
        "context_id": context_id,
        "close_date": close_date,
        "decision_status": status,
        "snapshot_timestamp": snapshot_timestamp,
        "snapshot_sha256": snapshot_sha256,
        "identity_hash_status": identity_hash_status,
        "targets": targets,
        "executions": executions,
        "asset_class_exposures": exposure_rows,
        "concentration": concentration,
        "incident_status": incidents,
    }
    assert_operationally_safe(payload, forbidden_field_tokens)
    return payload


def assert_operationally_safe(
    payload: object, forbidden_field_tokens: tuple[str, ...]
) -> None:
    """Reject proxy-performance fields recursively before publication."""
    if isinstance(payload, dict):
        for key, value in payload.items():
            normalized = str(key).lower()
            if any(token in normalized for token in forbidden_field_tokens):
                raise GovernanceError(f"sealed field leaked into operations: {key}")
            assert_operationally_safe(value, forbidden_field_tokens)
    elif isinstance(payload, list):
        for value in payload:
            assert_operationally_safe(value, forbidden_field_tokens)
    elif isinstance(payload, str):
        normalized = payload.lower()
        if any(token in normalized for token in forbidden_field_tokens):
            raise GovernanceError("sealed performance label leaked into operations")


def decode_sealing_public_key(path: Path) -> bytes:
    """Decode one base64 Curve25519 public key file."""
    try:
        raw = base64.b64decode(path.read_text(encoding="utf-8").strip(), validate=True)
    except (OSError, ValueError) as exc:
        raise GovernanceError("cannot decode performance-sealing public key") from exc
    if len(raw) != 32:
        raise GovernanceError("performance-sealing public key must contain 32 bytes")
    return raw


def sealing_key_fingerprint(raw: bytes) -> str:
    """Return a stable, non-secret fingerprint for a sealing recipient."""
    digest = base64.b64encode(hashlib.sha256(raw).digest()).decode("ascii").rstrip("=")
    return f"SHA256:{digest}"


def _exposures(
    executions: list[dict[str, object]], asset_classes: dict[str, str]
) -> list[dict[str, object]]:
    totals: dict[tuple[str, str], float] = {}
    for row in executions:
        ticker = str(row["ticker"])
        key = (str(row["strategy"]), asset_classes[ticker])
        totals[key] = totals.get(key, 0.0) + float(row["execution_weight"])
    return [
        {"strategy": strategy, "asset_class": asset_class, "exposure": value}
        for (strategy, asset_class), value in sorted(totals.items())
    ]


def _concentration(executions: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[float]] = {}
    for row in executions:
        grouped.setdefault(str(row["strategy"]), []).append(
            float(row["execution_weight"])
        )
    return [
        {
            "strategy": strategy,
            "maximum_single_asset_weight": max(weights),
            "hhi": float(np.square(weights).sum()),
        }
        for strategy, weights in sorted(grouped.items())
    ]


def _keys(payload: Any, expected: set[str], label: str) -> None:
    if not isinstance(payload, dict) or set(payload) != expected:
        raise GovernanceError(f"{label} keys mismatch")


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise GovernanceError(f"{label} SHA-256 is unresolved")
    try:
        int(value, 16)
    except ValueError as exc:
        raise GovernanceError(f"{label} SHA-256 is invalid") from exc
    return value


def _string_tuple(value: Any, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value or not all(
        isinstance(item, str) and item for item in value
    ):
        raise GovernanceError(f"{label} must be a nonempty string list")
    if len(value) != len(set(value)):
        raise GovernanceError(f"{label} contains duplicates")
    return tuple(value)

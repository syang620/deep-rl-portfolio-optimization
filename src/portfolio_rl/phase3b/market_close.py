"""Signed point-in-time market-close snapshot validation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from portfolio_rl.phase3b.governance import (
    GovernanceError,
    logical_json_sha256,
    read_json,
    resolve_path,
    sha256_file,
)
from portfolio_rl.phase3b.operational_metrics import OperationsConfig
from portfolio_rl.phase3b.signatures import verify_signature_record
from portfolio_rl.phase3b.snapshot_chain import dataframe_logical_sha256

MARKET_CLOSE_SIGNATURE_NAMESPACE = "portfolio-rl-phase3b-market-close-v1"


@dataclass(frozen=True)
class MarketCloseSnapshot:
    """Verified same-close prices and returns in canonical asset order."""

    snapshot_id: str
    previous_close_date: date
    close_date: date
    generated_at: datetime
    asset_order: tuple[str, ...]
    log_returns: tuple[float, ...]
    snapshot_sha256: str
    manifest_path: Path
    prices_path: Path


def load_market_close_snapshot(
    *,
    manifest_path: Path,
    repository_root: Path,
    operations_config: OperationsConfig,
    expected_asset_order: tuple[str, ...],
    service_public_key_path: Path,
) -> MarketCloseSnapshot:
    """Verify source chronology, values, logical content, and service signature."""
    root = repository_root.resolve()
    resolved = resolve_path(root, manifest_path)
    envelope = read_json(resolved)
    if set(envelope) != {"snapshot", "signature_record"}:
        raise GovernanceError("market-close snapshot envelope schema mismatch")
    payload = envelope["snapshot"]
    if not isinstance(payload, dict):
        raise GovernanceError("market-close snapshot payload is missing")
    expected = {
        "schema_version",
        "snapshot_id",
        "previous_close_date",
        "close_date",
        "generated_at",
        "price_field",
        "asset_order",
        "prices_file",
        "snapshot_payload_sha256",
    }
    if set(payload) != expected:
        raise GovernanceError("market-close snapshot schema mismatch")
    if payload["schema_version"] != operations_config.close_snapshot_schema_version:
        raise GovernanceError("market-close snapshot version mismatch")
    if payload["price_field"] != operations_config.price_field:
        raise GovernanceError("market-close price field mismatch")
    unhashed = dict(payload)
    recorded_hash = unhashed.pop("snapshot_payload_sha256", None)
    if logical_json_sha256(unhashed) != recorded_hash:
        raise GovernanceError("market-close snapshot payload hash mismatch")
    verify_signature_record(
        payload=payload,
        record=envelope["signature_record"],
        public_key_path=service_public_key_path,
        expected_role="service_signing",
        expected_namespace=MARKET_CLOSE_SIGNATURE_NAMESPACE,
    )
    previous = _date(payload["previous_close_date"], "previous close")
    close = _date(payload["close_date"], "close")
    if close <= previous:
        raise GovernanceError("market-close snapshot chronology is invalid")
    generated_at = _datetime(payload["generated_at"])
    if generated_at.date() < close:
        raise GovernanceError("market-close snapshot was generated before its close")
    asset_order = tuple(payload["asset_order"])
    if asset_order != expected_asset_order:
        raise GovernanceError("market-close snapshot asset order mismatch")
    record = payload["prices_file"]
    if not isinstance(record, dict) or set(record) != {
        "path",
        "sha256",
        "logical_sha256",
    }:
        raise GovernanceError("market-close prices file record is malformed")
    prices_path = resolve_path(root, record["path"])
    if sha256_file(prices_path) != record["sha256"]:
        raise GovernanceError("market-close prices file hash mismatch")
    frame = pd.read_parquet(prices_path)
    if dataframe_logical_sha256(frame) != record["logical_sha256"]:
        raise GovernanceError("market-close prices logical hash mismatch")
    log_returns = _validate_prices(frame, asset_order, previous, close)
    return MarketCloseSnapshot(
        snapshot_id=str(payload["snapshot_id"]),
        previous_close_date=previous,
        close_date=close,
        generated_at=generated_at,
        asset_order=asset_order,
        log_returns=log_returns,
        snapshot_sha256=str(recorded_hash),
        manifest_path=resolved,
        prices_path=prices_path,
    )


def _validate_prices(
    frame: pd.DataFrame,
    asset_order: tuple[str, ...],
    previous: date,
    close: date,
) -> tuple[float, ...]:
    expected = [
        "ticker",
        "previous_close_date",
        "close_date",
        "previous_adjusted_close",
        "adjusted_close",
        "log_return",
    ]
    if list(frame.columns) != expected or tuple(frame["ticker"]) != asset_order:
        raise GovernanceError("market-close prices schema or asset order mismatch")
    if set(pd.to_datetime(frame["previous_close_date"]).dt.date) != {previous}:
        raise GovernanceError("market-close previous dates are inconsistent")
    if set(pd.to_datetime(frame["close_date"]).dt.date) != {close}:
        raise GovernanceError("market-close dates are inconsistent")
    old = frame["previous_adjusted_close"].to_numpy(dtype=np.float64)
    new = frame["adjusted_close"].to_numpy(dtype=np.float64)
    observed = frame["log_return"].to_numpy(dtype=np.float64)
    if (
        not np.isfinite(old).all()
        or not np.isfinite(new).all()
        or (old <= 0).any()
        or (new <= 0).any()
        or not np.allclose(observed, np.log(new / old), atol=1e-12, rtol=1e-12)
    ):
        raise GovernanceError("market-close returns do not reconcile to prices")
    return tuple(float(value) for value in observed)


def _date(value: Any, label: str) -> date:
    try:
        return date.fromisoformat(str(value))
    except ValueError as exc:
        raise GovernanceError(f"{label} date is invalid") from exc


def _datetime(value: Any) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError as exc:
        raise GovernanceError("market-close timestamp is invalid") from exc
    if parsed.tzinfo is None:
        raise GovernanceError("market-close timestamp must be timezone-aware")
    return parsed.astimezone(UTC)

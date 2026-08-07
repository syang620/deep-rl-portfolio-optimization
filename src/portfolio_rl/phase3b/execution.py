"""Frozen Phase 3B execution configuration and delayed instructions."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime, time
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np

from portfolio_rl.phase3b.governance import (
    EXPECTED_CANDIDATE_MANIFEST_SHA256,
    EXPECTED_MODEL_VERSION,
    GovernanceError,
    read_yaml,
    sha256_file,
    ssh_public_key_fingerprint,
)

EXPECTED_ASSET_COST_BPS = {
    "SPY": 5.0,
    "QQQ": 5.0,
    "IWM": 8.0,
    "EFA": 8.0,
    "EEM": 12.0,
    "TLT": 7.0,
    "IEF": 5.0,
    "SHY": 3.0,
    "LQD": 8.0,
    "HYG": 12.0,
    "GLD": 8.0,
    "DBC": 15.0,
    "VNQ": 10.0,
    "XLU": 8.0,
}
EXPECTED_SCALER_SHA256 = (
    "0f842facb61abc148dbdafc7c62410cbbae36e9df367478996702b5f8568691d"
)
EXPECTED_FEATURE_SPEC_SHA256 = (
    "718f08bd336d02805c36434c110f2b362f31383b8c975985b34d925e11a48520"
)
SCALER_APPROVAL_ROLES = (
    "portfolio_manager",
    "independent_reviewer",
    "data_operations_custodian",
)
RECOMMENDATION_SIGNATURE_NAMESPACE = "portfolio-rl-phase3b-recommendation-v1"


@dataclass(frozen=True)
class RecommendationSigningConfig:
    """Approved public identity for automated recommendation signing."""

    principal: str
    public_key_path: Path
    public_key_sha256: str
    public_key_fingerprint: str
    namespace: str


@dataclass(frozen=True)
class ExecutionConfig:
    """Frozen serving and execution contract for Phase 3B decisions."""

    config_path: Path
    config_sha256: str
    candidate_manifest_sha256: str
    normalization_artifact_path: Path
    normalization_artifact_sha256: str
    feature_payload_schema_version: str
    live_state_schema_version: str
    recommendation_cutoff_local: time
    timezone: str
    execution_delay_closes: int
    primary_transaction_cost_bps: float
    asset_cost_bps: Mapping[str, float]
    signing: RecommendationSigningConfig
    output_root: Path
    scaler_status: str = "draft_pending_registration"
    scaler_approved_by: tuple[str, ...] = ()


@dataclass(frozen=True)
class ExecutionInstruction:
    """A target awaiting execution at the next eligible close."""

    strategy: str
    decision_date: date
    execution_date: date
    current_decision_weights: tuple[float, ...]
    target_weights: tuple[float, ...]
    execution_delay_closes: int
    pre_execution_action: str
    target_effective: str
    primary_transaction_cost_bps: float
    turnover_definition: str
    turnover_basis: str
    turnover_status: str
    transaction_cost_status: str


def load_execution_config(
    path: Path,
    *,
    repository_root: Path,
    require_approved: bool = True,
) -> ExecutionConfig:
    """Load the strict execution contract and reject unresolved serving inputs."""
    root = repository_root.resolve()
    config_path = _resolved(root, str(path), "execution config")
    payload = read_yaml(config_path)
    _keys(
        payload,
        {
            "schema_version",
            "status",
            "candidate",
            "inference",
            "snapshot_contract",
            "decision_timing",
            "costs",
            "recommendation_signing",
            "output_root",
        },
        "execution config",
    )
    if payload["schema_version"] != 1:
        raise GovernanceError("unsupported execution config schema")
    if require_approved and payload["status"] != "approved":
        raise GovernanceError("Phase 3B execution config is still draft")
    _keys(
        payload["candidate"],
        {"model_version", "candidate_manifest_sha256"},
        "execution candidate",
    )
    if payload["candidate"] != {
        "model_version": EXPECTED_MODEL_VERSION,
        "candidate_manifest_sha256": EXPECTED_CANDIDATE_MANIFEST_SHA256,
    }:
        raise GovernanceError("execution candidate identity mismatch")
    _keys(payload["inference"], {"device", "deterministic"}, "inference")
    if payload["inference"] != {"device": "cpu", "deterministic": True}:
        raise GovernanceError("Phase 3B inference must be deterministic on CPU")

    snapshot = payload["snapshot_contract"]
    _keys(
        snapshot,
        {
            "feature_payload_schema_version",
            "live_state_schema_version",
            "market_feature_count",
            "trailing_return_rows",
            "scaler",
        },
        "snapshot contract",
    )
    if snapshot["market_feature_count"] != 302:
        raise GovernanceError("snapshot must contain exactly 302 market features")
    if snapshot["trailing_return_rows"] != 63:
        raise GovernanceError("snapshot must contain exactly 63 trailing return rows")
    normalization = snapshot["scaler"]
    _keys(
        normalization,
        {
            "path",
            "status",
            "configured_hash",
            "approved_hash",
            "approved_by",
            "feature_spec_hash",
        },
        "scaler artifact",
    )
    scaler_status = str(normalization["status"])
    configured_sha = _sha(normalization["configured_hash"], "configured scaler")
    if configured_sha != EXPECTED_SCALER_SHA256:
        raise GovernanceError("configured scaler is not the frozen training scaler")
    if normalization["feature_spec_hash"] != EXPECTED_FEATURE_SPEC_SHA256:
        raise GovernanceError("scaler feature specification hash mismatch")
    approved_by = normalization["approved_by"]
    if not isinstance(approved_by, list) or not all(
        isinstance(role, str) for role in approved_by
    ):
        raise GovernanceError("scaler approved_by must be a list of roles")
    if require_approved:
        if scaler_status != "approved_for_phase3b":
            raise GovernanceError("scaler is not approved for Phase 3B")
        if normalization["approved_hash"] != configured_sha:
            raise GovernanceError("approved scaler hash mismatch")
        if len(approved_by) != len(SCALER_APPROVAL_ROLES) or set(approved_by) != set(
            SCALER_APPROVAL_ROLES
        ):
            raise GovernanceError("scaler requires all three approval roles")
    elif scaler_status not in {"draft_pending_registration", "approved_for_phase3b"}:
        raise GovernanceError("unsupported scaler approval status")
    scaler_path = _resolved(root, normalization["path"], "scaler artifact")
    scaler_sha = configured_sha
    if sha256_file(scaler_path) != scaler_sha:
        raise GovernanceError("scaler artifact hash mismatch")

    timing = payload["decision_timing"]
    _keys(
        timing,
        {"timezone", "recommendation_cutoff_local", "execution_delay_closes"},
        "decision timing",
    )
    if timing["timezone"] != "America/New_York":
        raise GovernanceError("decision timezone must be America/New_York")
    if timing["execution_delay_closes"] != 1:
        raise GovernanceError("execution delay must equal one close")
    try:
        cutoff = time.fromisoformat(str(timing["recommendation_cutoff_local"]))
    except ValueError as exc:
        raise GovernanceError(
            "recommendation cutoff must be an ISO local time"
        ) from exc
    if cutoff != time(10, 0):
        raise GovernanceError("recommendation cutoff must equal 10:00 AM ET")

    costs = payload["costs"]
    _keys(
        costs,
        {
            "turnover_definition",
            "primary_flat_bps",
            "asset_tier_role",
            "asset_cost_bps",
        },
        "execution costs",
    )
    if costs["turnover_definition"] != "half_l1_one_way":
        raise GovernanceError("turnover definition must be half-L1 one-way")
    if float(costs["primary_flat_bps"]) != 10.0:
        raise GovernanceError("primary Phase 3B cost must equal 10 bps")
    if costs["asset_tier_role"] != "stress_advisory":
        raise GovernanceError("asset-tier costs must remain stress/advisory")
    asset_costs = {
        str(key): float(value) for key, value in costs["asset_cost_bps"].items()
    }
    if asset_costs != EXPECTED_ASSET_COST_BPS:
        raise GovernanceError("asset-tier cost map mismatch")

    signing = payload["recommendation_signing"]
    _keys(
        signing,
        {
            "namespace",
            "principal",
            "public_key_path",
            "public_key_sha256",
            "public_key_fingerprint",
            "approval_status",
        },
        "recommendation signing",
    )
    if require_approved and signing["approval_status"] != "approved":
        raise GovernanceError("recommendation signing identity is not approved")
    if signing["namespace"] != RECOMMENDATION_SIGNATURE_NAMESPACE:
        raise GovernanceError("recommendation signature namespace mismatch")
    principal = signing["principal"]
    if not isinstance(principal, str) or not principal.strip():
        raise GovernanceError("recommendation signing principal is unresolved")
    public_key_path = _resolved(root, signing["public_key_path"], "signing public key")
    public_key_sha = _sha(signing["public_key_sha256"], "signing public key")
    if sha256_file(public_key_path) != public_key_sha:
        raise GovernanceError("recommendation signing public key hash mismatch")
    fingerprint = signing["public_key_fingerprint"]
    if not isinstance(fingerprint, str) or not fingerprint.startswith("SHA256:"):
        raise GovernanceError("recommendation signing fingerprint is unresolved")
    if ssh_public_key_fingerprint(public_key_path) != fingerprint:
        raise GovernanceError("recommendation signing fingerprint mismatch")

    output_root = _resolved(root, payload["output_root"], "decision output root")
    return ExecutionConfig(
        config_path=config_path,
        config_sha256=sha256_file(config_path),
        candidate_manifest_sha256=EXPECTED_CANDIDATE_MANIFEST_SHA256,
        normalization_artifact_path=scaler_path,
        normalization_artifact_sha256=scaler_sha,
        feature_payload_schema_version=str(snapshot["feature_payload_schema_version"]),
        live_state_schema_version=str(snapshot["live_state_schema_version"]),
        recommendation_cutoff_local=cutoff,
        timezone=timing["timezone"],
        execution_delay_closes=1,
        primary_transaction_cost_bps=10.0,
        asset_cost_bps=asset_costs,
        signing=RecommendationSigningConfig(
            principal=principal,
            public_key_path=public_key_path,
            public_key_sha256=public_key_sha,
            public_key_fingerprint=fingerprint,
            namespace=RECOMMENDATION_SIGNATURE_NAMESPACE,
        ),
        output_root=output_root,
        scaler_status=scaler_status,
        scaler_approved_by=tuple(approved_by),
    )


def require_before_recommendation_cutoff(
    *, generated_at: datetime, execution_date: date, config: ExecutionConfig
) -> None:
    """Reject recommendations generated after the frozen local cutoff."""
    if generated_at.tzinfo is None:
        raise GovernanceError("recommendation timestamp must be timezone-aware")
    generated_utc = generated_at.astimezone(UTC)
    local_zone = ZoneInfo(config.timezone)
    cutoff_local = datetime.combine(
        execution_date,
        config.recommendation_cutoff_local,
        tzinfo=local_zone,
    )
    if generated_utc > cutoff_local.astimezone(UTC):
        raise GovernanceError("recommendation was generated after the frozen cutoff")


def build_execution_instructions(
    *,
    decision_date: date,
    execution_date: date,
    current_weights: Mapping[str, tuple[float, ...]],
    targets: Mapping[str, tuple[float, ...]],
    config: ExecutionConfig,
) -> tuple[ExecutionInstruction, ...]:
    """Build pending instructions without inventing close-t+1 turnover or cost."""
    if set(current_weights) != set(targets):
        raise GovernanceError("execution strategies and live-state strategies differ")
    rows = []
    for strategy in sorted(targets):
        current = np.asarray(current_weights[strategy], dtype=np.float64)
        target = np.asarray(targets[strategy], dtype=np.float64)
        _weights(current, "current execution weights")
        _weights(target, "execution target weights")
        if current.shape != target.shape:
            raise GovernanceError("execution current and target shapes differ")
        rows.append(
            ExecutionInstruction(
                strategy=strategy,
                decision_date=decision_date,
                execution_date=execution_date,
                current_decision_weights=tuple(float(value) for value in current),
                target_weights=tuple(float(value) for value in target),
                execution_delay_closes=config.execution_delay_closes,
                pre_execution_action="hold_live_portfolio_through_execution_close",
                target_effective="after_execution_close",
                primary_transaction_cost_bps=config.primary_transaction_cost_bps,
                turnover_definition="half_l1_one_way",
                turnover_basis="live_drifted_weights_at_execution_close",
                turnover_status="pending_live_drifted_weights_at_execution_close",
                transaction_cost_status="pending_execution_close",
            )
        )
    return tuple(rows)


def _weights(values: np.ndarray, label: str) -> None:
    if values.ndim != 1 or not np.isfinite(values).all() or (values < 0).any():
        raise GovernanceError(f"{label} must be finite nonnegative one-dimensional")
    if not np.isclose(values.sum(), 1.0):
        raise GovernanceError(f"{label} must sum to one")


def _keys(payload: Any, expected: set[str], label: str) -> None:
    if not isinstance(payload, dict) or set(payload) != expected:
        raise GovernanceError(f"{label} keys mismatch")


def _resolved(root: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise GovernanceError(f"{label} path is unresolved")
    path = (
        (root / value).resolve()
        if not Path(value).is_absolute()
        else Path(value).resolve()
    )
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise GovernanceError(f"{label} path escapes repository root") from exc
    return path


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise GovernanceError(f"{label} SHA-256 is unresolved")
    try:
        int(value, 16)
    except ValueError as exc:
        raise GovernanceError(f"{label} SHA-256 is invalid") from exc
    return value

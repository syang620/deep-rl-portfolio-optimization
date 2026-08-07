"""One-close-delayed execution and daily portfolio-state roll-forward."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np

from portfolio_rl.env.costs import (
    calculate_transaction_cost_fraction,
    calculate_turnover,
)
from portfolio_rl.env.drift import simulate_buy_and_hold_period
from portfolio_rl.evaluation.execution_stress import (
    calculate_asset_specific_cost_fraction,
)
from portfolio_rl.phase3b.governance import (
    GovernanceError,
    canonical_json_bytes,
    logical_json_sha256,
    read_json,
    ssh_public_key_fingerprint,
)
from portfolio_rl.phase3b.signatures import (
    create_signature_record,
    verify_signature_record,
)
from portfolio_rl.phase3b.snapshot_chain import STRATEGIES

EXECUTION_SIGNATURE_NAMESPACE = "portfolio-rl-phase3b-execution-v1"


@dataclass(frozen=True)
class StrategyCloseState:
    """One strategy's reconstructable state immediately before a daily close."""

    weights: tuple[float, ...]
    nav: float
    peak_nav: float


@dataclass(frozen=True)
class CloseProcessingResult:
    """Public execution facts plus data intended only for the sealed ledger."""

    executions: tuple[dict[str, Any], ...]
    next_states: dict[str, StrategyCloseState]
    sealed_performance: dict[str, Any]
    result_sha256: str


def process_market_close(
    *,
    previous_close_date: date,
    close_date: date,
    asset_order: tuple[str, ...],
    asset_log_returns: tuple[float, ...],
    states: dict[str, StrategyCloseState],
    signed_targets: dict[str, tuple[float, ...]] | None,
    recommendation_hash: str | None,
    expected_execution_date: date | None,
    flat_transaction_cost_bps: float,
    asset_cost_bps: dict[str, float],
) -> CloseProcessingResult:
    """Drift through the close, then execute an already signed target once."""
    if close_date <= previous_close_date:
        raise GovernanceError("market-close dates are not increasing")
    if set(states) != set(STRATEGIES):
        raise GovernanceError("close state must contain every frozen strategy")
    returns = np.asarray(asset_log_returns, dtype=np.float64)
    if returns.shape != (len(asset_order),) or not np.isfinite(returns).all():
        raise GovernanceError("market-close returns are invalid")
    executing = signed_targets is not None
    if executing != (recommendation_hash is not None):
        raise GovernanceError("signed targets and recommendation hash must coexist")
    if executing and expected_execution_date != close_date:
        raise GovernanceError("recommendation cannot execute on this close")
    if not executing and expected_execution_date is not None:
        raise GovernanceError("execution date exists without signed targets")
    if executing and set(signed_targets or {}) != set(STRATEGIES):
        raise GovernanceError("signed target set is incomplete")
    asset_cost_array = np.asarray(
        [asset_cost_bps[ticker] for ticker in asset_order], dtype=np.float64
    )

    executions: list[dict[str, Any]] = []
    sealed_rows: list[dict[str, Any]] = []
    next_states: dict[str, StrategyCloseState] = {}
    for strategy in STRATEGIES:
        state = states[strategy]
        start = _weights(state.weights, len(asset_order), f"{strategy} state")
        gross_factor, drifted, daily = simulate_buy_and_hold_period(
            start, returns.reshape(1, -1)
        )
        pre_trade = drifted.astype(np.float64)
        if executing and strategy != "buy_and_hold_equal_weight":
            target = _weights(
                (signed_targets or {})[strategy],
                len(asset_order),
                f"{strategy} signed target",
            )
        else:
            target = pre_trade
        turnover = calculate_turnover(pre_trade, target)
        cost = calculate_transaction_cost_fraction(
            turnover, flat_transaction_cost_bps
        )
        tier_cost = calculate_asset_specific_cost_fraction(
            pre_trade, target, asset_cost_array
        )
        if strategy == "buy_and_hold_equal_weight" and turnover != 0.0:
            raise GovernanceError("buy-and-hold strategy must never trade")
        gross_return = float(daily[0])
        net_return = float(gross_factor * (1.0 - cost) - 1.0)
        nav = float(state.nav * (1.0 + net_return))
        if not np.isfinite(nav) or nav <= 0.0:
            raise GovernanceError("strategy NAV must remain positive and finite")
        peak = max(float(state.peak_nav), nav)
        drawdown = nav / peak - 1.0
        next_states[strategy] = StrategyCloseState(
            weights=tuple(float(value) for value in target),
            nav=nav,
            peak_nav=peak,
        )
        executions.extend(
            {
                "strategy": strategy,
                "ticker": ticker,
                "execution_weight": float(weight),
                "turnover": turnover,
                "transaction_cost_fraction": cost,
                "asset_tier_cost_fraction": tier_cost,
                "recommendation_hash": recommendation_hash,
                "execution_date": close_date.isoformat(),
            }
            for ticker, weight in zip(asset_order, target, strict=True)
        )
        sealed_rows.append(
            {
                "strategy": strategy,
                "pre_trade_weights": [float(value) for value in pre_trade],
                "signed_target_weights": (
                    list((signed_targets or {})[strategy]) if executing else None
                ),
                "execution_weights": [float(value) for value in target],
                "turnover": turnover,
                "transaction_cost_fraction": cost,
                "asset_tier_cost_fraction": tier_cost,
                "gross_return": gross_return,
                "net_return": net_return,
                "nav": nav,
                "drawdown": drawdown,
            }
        )
    sealed = {
        "schema_version": 1,
        "previous_close_date": previous_close_date.isoformat(),
        "close_date": close_date.isoformat(),
        "asset_order": list(asset_order),
        "recommendation_hash": recommendation_hash,
        "strategies": sealed_rows,
    }
    result_hash = logical_json_sha256(
        {
            "executions": executions,
            "sealed_performance_sha256": logical_json_sha256(sealed),
        }
    )
    return CloseProcessingResult(
        executions=tuple(executions),
        next_states=next_states,
        sealed_performance=sealed,
        result_sha256=result_hash,
    )


def write_signed_execution_artifact(
    *,
    path: Path,
    result: CloseProcessingResult,
    context_type: str,
    context_id: str,
    decision_id: str,
    recommendation_hash: str,
    input_snapshot_hash: str,
    execution_price_snapshot_hash: str,
    candidate_hash: str,
    container_digest: str,
    service_signing_fingerprint: str,
    service_private_key_path: Path,
    service_public_key_path: Path,
    service_principal: str,
    signed_at: datetime,
) -> Path:
    """Bind one execution to every approved input and sign it create-only."""
    if ssh_public_key_fingerprint(service_public_key_path) != service_signing_fingerprint:
        raise GovernanceError("execution service-signing fingerprint mismatch")
    payload = {
        "schema_version": 1,
        "context_type": context_type,
        "context_id": context_id,
        "decision_id": decision_id,
        "recommendation_hash": recommendation_hash,
        "input_snapshot_hash": input_snapshot_hash,
        "execution_price_snapshot_hash": execution_price_snapshot_hash,
        "candidate_hash": candidate_hash,
        "container_digest": container_digest,
        "service_signing_fingerprint": service_signing_fingerprint,
        "result_sha256": result.result_sha256,
        "executions": list(result.executions),
    }
    payload["execution_payload_sha256"] = logical_json_sha256(payload)
    envelope = {
        "execution": payload,
        "signature_record": create_signature_record(
            payload=payload,
            payload_path=path.name,
            artifact_type="phase3b_execution",
            role="service_signing",
            principal=service_principal,
            namespace=EXECUTION_SIGNATURE_NAMESPACE,
            private_key_path=service_private_key_path,
            public_key_path=service_public_key_path,
            signed_at=signed_at,
        ),
    }
    if path.exists():
        if read_json(path) != envelope:
            raise GovernanceError("signed execution overwrite is forbidden")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(envelope))
    return path


def verify_signed_execution_artifact(
    *, path: Path, service_public_key_path: Path
) -> dict[str, Any]:
    """Verify execution content, binding hash, and service signature."""
    envelope = read_json(path)
    if set(envelope) != {"execution", "signature_record"}:
        raise GovernanceError("execution artifact envelope mismatch")
    payload = envelope["execution"]
    if not isinstance(payload, dict):
        raise GovernanceError("execution payload is missing")
    unhashed = dict(payload)
    recorded = unhashed.pop("execution_payload_sha256", None)
    if logical_json_sha256(unhashed) != recorded:
        raise GovernanceError("execution payload hash mismatch")
    verify_signature_record(
        payload=payload,
        record=envelope["signature_record"],
        public_key_path=service_public_key_path,
        expected_role="service_signing",
        expected_namespace=EXECUTION_SIGNATURE_NAMESPACE,
    )
    return payload


def _weights(values: tuple[float, ...], size: int, label: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if (
        array.shape != (size,)
        or not np.isfinite(array).all()
        or (array < 0.0).any()
        or not np.isclose(array.sum(), 1.0)
    ):
        raise GovernanceError(f"{label} is invalid")
    return array

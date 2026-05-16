"""Transaction cost mechanics for portfolio rebalancing."""

from __future__ import annotations

import numpy as np


def calculate_turnover(
    current_weights: np.ndarray,
    target_weights: np.ndarray,
) -> float:
    """Calculate one-way turnover between current and target weights."""
    current = np.asarray(current_weights, dtype=np.float64)
    target = np.asarray(target_weights, dtype=np.float64)
    _validate_weights(current, "current_weights")
    _validate_weights(target, "target_weights")
    if current.shape != target.shape:
        raise ValueError("current_weights and target_weights must have the same shape")

    return float(np.abs(target - current).sum())


def calculate_transaction_cost_fraction(
    turnover: float,
    transaction_cost_bps: float,
) -> float:
    """Convert turnover and basis-point cost into a NAV cost fraction."""
    _validate_nonnegative_finite(turnover, "turnover")
    _validate_nonnegative_finite(transaction_cost_bps, "transaction_cost_bps")
    return float(turnover * transaction_cost_bps / 10_000.0)


def _validate_weights(weights: np.ndarray, name: str) -> None:
    if weights.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array")
    if weights.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.isfinite(weights).all():
        raise ValueError(f"{name} values must be finite")
    if (weights < 0.0).any():
        raise ValueError(f"{name} values must be nonnegative")


def _validate_nonnegative_finite(value: float, name: str) -> None:
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if value < 0.0:
        raise ValueError(f"{name} must be nonnegative")

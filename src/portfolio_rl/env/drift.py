"""Weight drift and return compounding for portfolio holding periods."""

from __future__ import annotations

import numpy as np


def simulate_buy_and_hold_period(
    start_weights: np.ndarray,
    forward_log_returns: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Simulate one buy-and-hold period from daily asset log returns."""
    weights = np.asarray(start_weights, dtype=np.float64)
    returns = np.asarray(forward_log_returns, dtype=np.float64)
    _validate_start_weights(weights)
    _validate_forward_log_returns(returns, weights.size)

    drifted_weights = weights.copy()
    period_gross_return = 1.0
    daily_portfolio_simple_returns = []

    for daily_log_returns in returns:
        asset_gross_returns = np.exp(daily_log_returns)
        portfolio_gross_return = float(np.dot(drifted_weights, asset_gross_returns))
        if portfolio_gross_return <= 0.0 or not np.isfinite(portfolio_gross_return):
            raise ValueError("portfolio gross return must be positive and finite")

        period_gross_return *= portfolio_gross_return
        daily_portfolio_simple_returns.append(portfolio_gross_return - 1.0)
        drifted_weights = (
            drifted_weights * asset_gross_returns / portfolio_gross_return
        )

    return (
        float(period_gross_return),
        drifted_weights.astype(np.float32),
        np.asarray(daily_portfolio_simple_returns, dtype=np.float32),
    )


def _validate_start_weights(weights: np.ndarray) -> None:
    if weights.ndim != 1:
        raise ValueError("start_weights must be a one-dimensional array")
    if weights.size == 0:
        raise ValueError("start_weights must not be empty")
    if not np.isfinite(weights).all():
        raise ValueError("start_weights values must be finite")
    if (weights < 0.0).any():
        raise ValueError("start_weights values must be nonnegative")
    if weights.sum() <= 0.0:
        raise ValueError("start_weights must have positive total weight")
    if not np.isclose(weights.sum(), 1.0):
        raise ValueError("start_weights must sum to one")


def _validate_forward_log_returns(
    forward_log_returns: np.ndarray,
    n_assets: int,
) -> None:
    if forward_log_returns.ndim != 2:
        raise ValueError("forward_log_returns must be a two-dimensional array")
    if forward_log_returns.shape[0] == 0:
        raise ValueError("forward_log_returns must contain at least one row")
    if forward_log_returns.shape[1] != n_assets:
        raise ValueError("forward_log_returns asset dimension must match weights")
    if not np.isfinite(forward_log_returns).all():
        raise ValueError("forward_log_returns values must be finite")

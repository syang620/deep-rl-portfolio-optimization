"""Reward functions for portfolio environments."""

from __future__ import annotations

import numpy as np


def log_growth_reward(
    period_gross_return: float,
    transaction_cost_fraction: float,
    reward_scale: float = 100.0,
    bad_gross_penalty: float = -100.0,
) -> float:
    """Calculate scaled log-growth reward after transaction costs."""
    _validate_positive_finite(period_gross_return, "period_gross_return")
    _validate_nonnegative_finite(
        transaction_cost_fraction,
        "transaction_cost_fraction",
    )
    _validate_positive_finite(reward_scale, "reward_scale")
    _validate_finite(bad_gross_penalty, "bad_gross_penalty")

    net_gross = (1.0 - transaction_cost_fraction) * period_gross_return
    if net_gross <= 0.0:
        return float(bad_gross_penalty)
    return float(np.log(net_gross) * reward_scale)


def _validate_positive_finite(value: float, name: str) -> None:
    _validate_finite(value, name)
    if value <= 0.0:
        raise ValueError(f"{name} must be positive")


def _validate_nonnegative_finite(value: float, name: str) -> None:
    _validate_finite(value, name)
    if value < 0.0:
        raise ValueError(f"{name} must be nonnegative")


def _validate_finite(value: float, name: str) -> None:
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")

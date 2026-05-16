from __future__ import annotations

import numpy as np
import pytest

from portfolio_rl.env.costs import (
    calculate_transaction_cost_fraction,
    calculate_turnover,
)


def test_zero_turnover_has_zero_cost() -> None:
    turnover = calculate_turnover(
        np.array([0.25, 0.75]),
        np.array([0.25, 0.75]),
    )

    assert turnover == 0.0
    assert calculate_transaction_cost_fraction(turnover, 10.0) == 0.0


def test_full_rebalance_turnover_is_correct() -> None:
    turnover = calculate_turnover(
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
    )

    assert turnover == 2.0


def test_cost_fraction_bps_conversion() -> None:
    cost_fraction = calculate_transaction_cost_fraction(
        turnover=1.0,
        transaction_cost_bps=10.0,
    )

    assert cost_fraction == 0.001


def test_turnover_rejects_mismatched_weight_shapes() -> None:
    with pytest.raises(ValueError, match="same shape"):
        calculate_turnover(np.array([0.5, 0.5]), np.array([1.0]))


def test_turnover_rejects_multidimensional_weights() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        calculate_turnover(np.array([[0.5, 0.5]]), np.array([0.5, 0.5]))


def test_turnover_rejects_empty_weights() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        calculate_turnover(np.array([]), np.array([]))


def test_turnover_rejects_nonfinite_weights() -> None:
    with pytest.raises(ValueError, match="values must be finite"):
        calculate_turnover(np.array([0.5, np.nan]), np.array([0.5, 0.5]))


def test_turnover_rejects_negative_weights() -> None:
    with pytest.raises(ValueError, match="values must be nonnegative"):
        calculate_turnover(np.array([1.1, -0.1]), np.array([0.5, 0.5]))


@pytest.mark.parametrize("turnover", [-1.0, np.inf])
def test_cost_fraction_rejects_invalid_turnover(turnover: float) -> None:
    with pytest.raises(ValueError, match="turnover"):
        calculate_transaction_cost_fraction(turnover, transaction_cost_bps=10.0)


@pytest.mark.parametrize("transaction_cost_bps", [-1.0, np.nan])
def test_cost_fraction_rejects_invalid_bps(transaction_cost_bps: float) -> None:
    with pytest.raises(ValueError, match="transaction_cost_bps"):
        calculate_transaction_cost_fraction(
            turnover=1.0,
            transaction_cost_bps=transaction_cost_bps,
        )

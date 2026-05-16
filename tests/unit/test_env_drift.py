from __future__ import annotations

from math import log

import numpy as np
import pytest

from portfolio_rl.env.drift import simulate_buy_and_hold_period


def test_zero_returns_preserve_nav_and_weights() -> None:
    period_gross, end_weights, daily_returns = simulate_buy_and_hold_period(
        np.array([0.25, 0.75]),
        np.zeros((3, 2)),
    )

    assert period_gross == 1.0
    np.testing.assert_allclose(end_weights, [0.25, 0.75])
    np.testing.assert_allclose(daily_returns, [0.0, 0.0, 0.0])


def test_one_asset_case_compounds_exactly() -> None:
    period_gross, end_weights, daily_returns = simulate_buy_and_hold_period(
        np.array([1.0]),
        np.array([[log(1.01)], [log(0.99)], [log(1.02)]]),
    )

    np.testing.assert_allclose(period_gross, 1.01 * 0.99 * 1.02)
    np.testing.assert_allclose(end_weights, [1.0])
    np.testing.assert_allclose(daily_returns, [0.01, -0.01, 0.02])


def test_weights_drift_toward_outperforming_asset() -> None:
    period_gross, end_weights, daily_returns = simulate_buy_and_hold_period(
        np.array([0.5, 0.5]),
        np.array([[log(1.10), log(1.00)]]),
    )

    np.testing.assert_allclose(period_gross, 1.05)
    np.testing.assert_allclose(end_weights, [1.10 / 2.10, 1.00 / 2.10])
    np.testing.assert_allclose(daily_returns, [0.05])
    assert end_weights[0] > 0.5
    assert end_weights[1] < 0.5


def test_daily_portfolio_returns_have_one_value_per_horizon_row() -> None:
    _, _, daily_returns = simulate_buy_and_hold_period(
        np.array([0.5, 0.5]),
        np.array(
            [
                [log(1.01), log(1.00)],
                [log(1.00), log(1.02)],
                [log(0.99), log(1.01)],
            ]
        ),
    )

    assert daily_returns.shape == (3,)


def test_simulate_buy_and_hold_period_rejects_invalid_start_weights() -> None:
    with pytest.raises(ValueError, match="sum to one"):
        simulate_buy_and_hold_period(
            np.array([0.5, 0.6]),
            np.zeros((1, 2)),
        )


def test_simulate_buy_and_hold_period_rejects_negative_start_weights() -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        simulate_buy_and_hold_period(
            np.array([1.1, -0.1]),
            np.zeros((1, 2)),
        )


def test_simulate_buy_and_hold_period_rejects_invalid_return_shape() -> None:
    with pytest.raises(ValueError, match="two-dimensional"):
        simulate_buy_and_hold_period(
            np.array([1.0]),
            np.array([log(1.01)]),
        )


def test_simulate_buy_and_hold_period_rejects_asset_dimension_mismatch() -> None:
    with pytest.raises(ValueError, match="asset dimension"):
        simulate_buy_and_hold_period(
            np.array([0.5, 0.5]),
            np.zeros((1, 3)),
        )


def test_simulate_buy_and_hold_period_rejects_nonfinite_inputs() -> None:
    with pytest.raises(ValueError, match="forward_log_returns values must be finite"):
        simulate_buy_and_hold_period(
            np.array([1.0]),
            np.array([[np.nan]]),
        )

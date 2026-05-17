from __future__ import annotations

import numpy as np
import pytest

from portfolio_rl.policies.baseline_policies import (
    BuyAndHoldEqualWeightPolicy,
    EqualWeightWeeklyPolicy,
    InverseVolatilityPolicy,
    SingleAssetPolicy,
)


def test_equal_weight_weekly_policy_returns_equal_weights() -> None:
    policy = EqualWeightWeeklyPolicy(n_assets=4)

    weights = policy.target_weights(np.zeros(1), {})

    _assert_valid_weights(weights)
    np.testing.assert_allclose(weights, [0.25, 0.25, 0.25, 0.25])


def test_buy_and_hold_policy_returns_equal_weights_first() -> None:
    policy = BuyAndHoldEqualWeightPolicy(n_assets=2)

    weights = policy.target_weights(np.zeros(1), {})

    _assert_valid_weights(weights)
    np.testing.assert_allclose(weights, [0.5, 0.5])


def test_buy_and_hold_policy_targets_current_weights_after_first_call() -> None:
    policy = BuyAndHoldEqualWeightPolicy(n_assets=2)
    policy.target_weights(np.zeros(1), {})

    weights = policy.target_weights(
        np.zeros(1),
        {"current_weights": np.array([0.6, 0.4])},
    )

    _assert_valid_weights(weights)
    np.testing.assert_allclose(weights, [0.6, 0.4])


def test_buy_and_hold_policy_reset_starts_new_episode() -> None:
    policy = BuyAndHoldEqualWeightPolicy(n_assets=2)
    policy.target_weights(np.zeros(1), {})
    policy.reset()

    weights = policy.target_weights(np.zeros(1), {})

    np.testing.assert_allclose(weights, [0.5, 0.5])


def test_buy_and_hold_policy_requires_current_weights_after_first_call() -> None:
    policy = BuyAndHoldEqualWeightPolicy(n_assets=2)
    policy.target_weights(np.zeros(1), {})

    with pytest.raises(ValueError, match="current_weights"):
        policy.target_weights(np.zeros(1), {})


def test_single_asset_policy_allocates_to_requested_ticker() -> None:
    policy = SingleAssetPolicy(["SPY", "QQQ", "SHY"], ticker="SHY")

    weights = policy.target_weights(np.zeros(1), {})

    _assert_valid_weights(weights)
    np.testing.assert_allclose(weights, [0.0, 0.0, 1.0])


def test_single_asset_policy_normalizes_ticker_case() -> None:
    policy = SingleAssetPolicy(["SPY", "QQQ"], ticker="spy")

    weights = policy.target_weights(np.zeros(1), {})

    np.testing.assert_allclose(weights, [1.0, 0.0])


def test_single_asset_policy_rejects_unknown_ticker() -> None:
    with pytest.raises(ValueError, match="ticker is not in asset_order"):
        SingleAssetPolicy(["SPY", "QQQ"], ticker="SHY")


def test_inverse_volatility_policy_allocates_more_to_lower_vol_asset() -> None:
    policy = InverseVolatilityPolicy(n_assets=2)

    weights = policy.target_weights(
        np.zeros(1),
        {
            "trailing_log_returns": np.array(
                [
                    [0.01, 0.03],
                    [-0.01, -0.03],
                    [0.01, 0.03],
                    [-0.01, -0.03],
                ]
            )
        },
    )

    _assert_valid_weights(weights)
    assert weights[0] > weights[1]


def test_inverse_volatility_policy_handles_zero_volatility_with_floor() -> None:
    policy = InverseVolatilityPolicy(n_assets=2, volatility_floor=1e-6)

    weights = policy.target_weights(
        np.zeros(1),
        {"trailing_log_returns": np.zeros((3, 2))},
    )

    _assert_valid_weights(weights)
    np.testing.assert_allclose(weights, [0.5, 0.5])


def test_inverse_volatility_policy_requires_trailing_returns() -> None:
    policy = InverseVolatilityPolicy(n_assets=2)

    with pytest.raises(ValueError, match="trailing_log_returns"):
        policy.target_weights(np.zeros(1), {})


def test_inverse_volatility_policy_rejects_nonfinite_returns() -> None:
    policy = InverseVolatilityPolicy(n_assets=2)

    with pytest.raises(ValueError, match="values must be finite"):
        policy.target_weights(
            np.zeros(1),
            {"trailing_log_returns": np.array([[0.0, np.nan]])},
        )


def test_baseline_policies_reject_invalid_asset_counts() -> None:
    with pytest.raises(ValueError, match="n_assets must be positive"):
        EqualWeightWeeklyPolicy(n_assets=0)
    with pytest.raises(ValueError, match="n_assets must be positive"):
        BuyAndHoldEqualWeightPolicy(n_assets=-1)
    with pytest.raises(ValueError, match="n_assets must be positive"):
        InverseVolatilityPolicy(n_assets=0)


@pytest.mark.parametrize(
    "current_weights",
    [
        np.array([0.5, np.nan]),
        np.array([1.1, -0.1]),
        np.array([0.6, 0.6]),
        np.array([1.0]),
    ],
)
def test_buy_and_hold_policy_rejects_invalid_current_weights(
    current_weights: np.ndarray,
) -> None:
    policy = BuyAndHoldEqualWeightPolicy(n_assets=2)
    policy.target_weights(np.zeros(1), {})

    with pytest.raises(ValueError):
        policy.target_weights(np.zeros(1), {"current_weights": current_weights})


def _assert_valid_weights(weights: np.ndarray) -> None:
    assert weights.dtype == np.float32
    assert np.isfinite(weights).all()
    assert np.all(weights >= 0.0)
    assert np.isclose(weights.sum(), 1.0)

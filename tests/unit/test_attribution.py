from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio_rl.evaluation.attribution import (
    BuyAndHoldTargetPolicy,
    ConstantTargetPolicy,
    build_asset_contributions,
    build_exposure_paths,
)


def test_asset_contributions_and_cost_effect_reconcile_daily_return() -> None:
    positions = pd.DataFrame(
        {
            "fold_id": ["WF4"] * 4,
            "strategy": ["candidate", "candidate", "equal", "equal"],
            "date": pd.to_datetime(["2022-01-04"] * 4),
            "ticker": ["SPY", "SHY", "SPY", "SHY"],
            "pre_return_weight": [0.8, 0.2, 0.5, 0.5],
            "asset_simple_return": [0.01, 0.0, 0.01, 0.0],
            "gross_return_contribution": [0.008, 0.0, 0.005, 0.0],
        }
    )
    nav = pd.DataFrame(
        {
            "fold_id": ["WF4", "WF4"],
            "strategy": ["candidate", "equal"],
            "date": pd.to_datetime(["2022-01-04", "2022-01-04"]),
            "daily_return": [0.0079, 0.00495],
        }
    )
    result = build_asset_contributions(
        positions=positions, nav=nav, reference_strategy="equal"
    )
    candidate = result[result["strategy"] == "candidate"]
    assert candidate["transaction_cost_return_effect"].iloc[0] == pytest.approx(
        -0.0001
    )
    assert candidate["active_gross_contribution"].sum() == pytest.approx(0.003)


def test_exposure_groups_are_exhaustive_and_sum_to_one() -> None:
    positions = pd.DataFrame(
        {
            "fold_id": ["WF4", "WF4"],
            "strategy": ["candidate", "candidate"],
            "date": pd.to_datetime(["2022-01-04", "2022-01-04"]),
            "ticker": ["SPY", "SHY"],
            "pre_return_weight": [0.7, 0.3],
        }
    )
    result = build_exposure_paths(
        positions=positions,
        exposure_groups={"equity": ["SPY"], "cash_proxy": ["SHY"]},
    )
    assert result["exposure"].sum() == 1.0


def test_static_policies_distinguish_weekly_rebalance_and_buy_and_hold() -> None:
    target = np.array([0.7, 0.3])
    weekly = ConstantTargetPolicy(target)
    buy_hold = BuyAndHoldTargetPolicy(target)
    info = {"current_weights": np.array([0.6, 0.4])}
    np.testing.assert_array_equal(weekly.target_weights(np.array([]), info), target)
    np.testing.assert_array_equal(buy_hold.target_weights(np.array([]), info), target)
    np.testing.assert_array_equal(
        buy_hold.target_weights(np.array([]), info), info["current_weights"]
    )

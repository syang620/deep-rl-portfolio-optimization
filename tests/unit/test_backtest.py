from __future__ import annotations

from math import log

import numpy as np
import pandas as pd
import pytest

from portfolio_rl.data.dataset import build_portfolio_dataset
from portfolio_rl.data.feature_store import PortfolioFeatureStore
from portfolio_rl.evaluation.backtest import run_weight_policy_backtest
from portfolio_rl.features.feature_spec import FeatureSpec
from portfolio_rl.policies.baseline_policies import (
    BuyAndHoldEqualWeightPolicy,
    EqualWeightWeeklyPolicy,
    SingleAssetPolicy,
)


def test_equal_weight_backtest_produces_positive_finite_nav() -> None:
    result = run_weight_policy_backtest(
        feature_store=_feature_store(),
        policy=EqualWeightWeeklyPolicy(n_assets=2),
        strategy="equal_weight",
        transaction_cost_bps=0.0,
        max_steps=2,
    )

    assert not result.nav.empty
    assert np.isfinite(result.nav["nav"]).all()
    assert (result.nav["nav"] > 0.0).all()
    assert np.isfinite(result.nav["daily_return"]).all()
    assert np.isfinite(result.nav["drawdown"]).all()
    assert result.metrics["total_return"] is not None
    assert np.isfinite(result.metrics["total_return"])


def test_backtest_weights_are_long_only_and_sum_to_one() -> None:
    result = run_weight_policy_backtest(
        feature_store=_feature_store(),
        policy=SingleAssetPolicy(["SPY", "QQQ"], "SPY"),
        strategy="spy_only",
        transaction_cost_bps=0.0,
        max_steps=2,
    )

    _assert_grouped_weights_sum_to_one(
        result.weights_target,
        "target_weight",
    )
    _assert_grouped_weights_sum_to_one(
        result.weights_drifted,
        "drifted_weight",
    )


def test_backtest_trades_equal_target_minus_pre_trade_weights() -> None:
    result = run_weight_policy_backtest(
        feature_store=_feature_store(),
        policy=SingleAssetPolicy(["SPY", "QQQ"], "SPY"),
        strategy="spy_only",
        transaction_cost_bps=0.0,
        max_steps=1,
    )

    np.testing.assert_allclose(
        result.trades["trade_weight"],
        result.trades["target_weight"] - result.trades["pre_trade_weight"],
    )


def test_backtest_costs_are_nonnegative() -> None:
    result = run_weight_policy_backtest(
        feature_store=_feature_store(),
        policy=SingleAssetPolicy(["SPY", "QQQ"], "SPY"),
        strategy="spy_only",
        transaction_cost_bps=10.0,
        max_steps=2,
    )

    assert np.isfinite(result.costs["turnover"]).all()
    assert np.isfinite(result.costs["transaction_cost_fraction"]).all()
    assert (result.costs["turnover"] >= 0.0).all()
    assert (result.costs["transaction_cost_fraction"] >= 0.0).all()


def test_buy_and_hold_has_zero_turnover_after_initial_rebalance() -> None:
    result = run_weight_policy_backtest(
        feature_store=_feature_store(),
        policy=BuyAndHoldEqualWeightPolicy(n_assets=2),
        strategy="buy_and_hold",
        transaction_cost_bps=10.0,
        max_steps=3,
    )

    assert len(result.costs) == 3
    assert result.costs["turnover"].iloc[1:].eq(0.0).all()


def test_equal_weight_backtest_matches_manual_two_asset_case() -> None:
    result = run_weight_policy_backtest(
        feature_store=_feature_store(
            n_train_rows=2,
            return_overrides={1: (log(1.10), log(1.00))},
        ),
        policy=EqualWeightWeeklyPolicy(n_assets=2),
        strategy="equal_weight",
        rebalance_frequency_trading_days=1,
        transaction_cost_bps=0.0,
        max_steps=1,
    )

    assert result.nav["nav"].iloc[-1] == pytest.approx(1.05)
    np.testing.assert_allclose(
        result.weights_drifted["drifted_weight"],
        [1.10 / 2.10, 1.00 / 2.10],
    )


def _assert_grouped_weights_sum_to_one(frame: pd.DataFrame, column: str) -> None:
    assert np.isfinite(frame[column]).all()
    assert (frame[column] >= 0.0).all()
    grouped = frame.groupby(["date", "strategy"])[column].sum()
    np.testing.assert_allclose(grouped.to_numpy(), np.ones(len(grouped)))


def _feature_store(
    *,
    n_train_rows: int = 20,
    return_overrides: dict[int, tuple[float, float]] | None = None,
) -> PortfolioFeatureStore:
    return PortfolioFeatureStore(
        build_portfolio_dataset(
            _model_matrix(n_train_rows, return_overrides or {}),
            _feature_spec(),
        ),
        split="train",
    )


def _feature_spec() -> FeatureSpec:
    return FeatureSpec(
        feature_version="v1",
        asset_order=["SPY", "QQQ"],
        per_asset_features=["ret_1d"],
        global_features=["vix_z_21d"],
        current_weight_features=["weight_spy", "weight_qqq"],
        observation_dim=5,
        created_at="2026-05-07T00:00:00+00:00",
    )


def _model_matrix(
    n_train_rows: int,
    return_overrides: dict[int, tuple[float, float]],
) -> pd.DataFrame:
    dates = pd.date_range("2024-01-02", periods=n_train_rows + 2, freq="B")
    rows = []
    for index, date in enumerate(dates):
        spy_return, qqq_return = return_overrides.get(
            index,
            (0.0001 * np.sin(index), 0.0001 * np.cos(index)),
        )
        rows.append(
            {
                "date": date,
                "split": "train" if index < n_train_rows else "validation",
                "feature_version": "v1",
                "obs_000": 10.0 + index,
                "obs_001": 20.0 + index,
                "obs_002": 30.0 + index,
                "obs_003": 0.5,
                "obs_004": 0.5,
                "return_spy_1d": spy_return,
                "return_qqq_1d": qqq_return,
            }
        )
    return pd.DataFrame(rows)

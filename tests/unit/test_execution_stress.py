from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
import pytest

from portfolio_rl.data.dataset import PortfolioDataset
from portfolio_rl.data.feature_store import PortfolioFeatureStore
from portfolio_rl.evaluation.backtest import run_weight_policy_backtest
from portfolio_rl.evaluation.execution_stress import (
    ExecutionCostScenario,
    calculate_asset_specific_cost_fraction,
    run_execution_stress_backtest,
)


class CapturingPolicy:
    def __init__(self, target: np.ndarray) -> None:
        self.target = target
        self.current_weights: list[np.ndarray] = []

    def target_weights(
        self, observation: np.ndarray, info: Mapping[str, Any]
    ) -> np.ndarray:
        del observation
        self.current_weights.append(np.asarray(info["current_weights"]).copy())
        return self.target.copy()


def test_uniform_asset_cost_is_exactly_scalar_cost() -> None:
    current = np.array([0.2, 0.3, 0.5])
    target = np.array([0.4, 0.1, 0.5])
    bps = np.full(3, 25.0)
    expected = 0.5 * np.abs(target - current).sum() * 25.0 / 10_000.0
    assert calculate_asset_specific_cost_fraction(current, target, bps) == expected


def test_asset_cost_map_must_exactly_match_asset_order() -> None:
    scenario = ExecutionCostScenario("tiers", asset_bps={"SPY": 5.0})
    with pytest.raises(ValueError, match="exactly match"):
        scenario.validate(["SPY", "SHY"])


def test_no_delay_flat_cost_exactly_matches_standard_backtest() -> None:
    store = _store()
    standard = run_weight_policy_backtest(
        feature_store=store,
        policy=CapturingPolicy(np.array([0.8, 0.2])),
        strategy="candidate",
        transaction_cost_bps=10.0,
        inverse_vol_lookback_trading_days=2,
    )
    stressed = run_execution_stress_backtest(
        feature_store=store,
        policy=CapturingPolicy(np.array([0.8, 0.2])),
        strategy="candidate",
        cost_scenario=ExecutionCostScenario("flat_10_bps", flat_bps=10.0),
        execution_delay_closes=0,
        inverse_vol_lookback_trading_days=2,
    ).backtest
    pd.testing.assert_frame_equal(stressed.nav, standard.nav, check_exact=True)
    pd.testing.assert_frame_equal(stressed.trades, standard.trades, check_exact=True)
    pd.testing.assert_frame_equal(stressed.costs, standard.costs, check_exact=True)
    assert stressed.metrics == standard.metrics


def test_one_close_delay_earns_first_return_before_trading() -> None:
    store = _store()
    policy = CapturingPolicy(np.array([0.8, 0.2]))
    result = run_execution_stress_backtest(
        feature_store=store,
        policy=policy,
        strategy="candidate",
        cost_scenario=ExecutionCostScenario("flat_10_bps", flat_bps=10.0),
        execution_delay_closes=1,
        inverse_vol_lookback_trading_days=2,
        max_steps=1,
    )
    first_forward = store.get_forward_log_returns(0, 1)[0]
    gross = np.exp(first_forward)
    expected_pre_trade = np.array([0.5, 0.5]) * gross / np.dot(
        np.array([0.5, 0.5]), gross
    )
    trades = result.backtest.trades.pivot(
        index="date", columns="ticker", values="pre_trade_weight"
    )
    np.testing.assert_allclose(
        trades[["SPY", "SHY"]].iloc[0].to_numpy(), expected_pre_trade
    )
    assert result.execution_audit.iloc[0]["decision_date"] == store.date_at(0)
    assert result.execution_audit.iloc[0]["execution_date"] == store.date_at(1)
    expected_turnover = 0.5 * np.abs(np.array([0.8, 0.2]) - expected_pre_trade).sum()
    assert result.backtest.costs.iloc[0]["turnover"] == pytest.approx(expected_turnover)
    assert result.backtest.nav.iloc[0]["daily_return"] == pytest.approx(
        float(
            np.dot(np.array([0.5, 0.5]), gross)
            * (1 - expected_turnover * 10.0 / 10_000)
            - 1
        )
    )


def _store() -> PortfolioFeatureStore:
    dates = pd.date_range("2021-12-28", periods=13, freq="B")
    returns = np.zeros((13, 2), dtype=np.float32)
    returns[3, 0] = np.log(1.10)
    returns[4:, 0] = np.log(1.01)
    dataset = PortfolioDataset(
        dates=dates,
        splits=np.array(["inner_validation"] * 2 + ["outer_evaluation"] * 11),
        market_features=np.arange(13, dtype=np.float32).reshape(-1, 1),
        returns=returns,
        asset_order=["SPY", "SHY"],
        feature_version="test",
        observation_dim=3,
    )
    return PortfolioFeatureStore(dataset, "outer_evaluation")

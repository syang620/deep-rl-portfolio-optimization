"""Evaluation-only execution-delay and transaction-cost stresses."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from portfolio_rl.data.feature_store import PortfolioFeatureStore
from portfolio_rl.env.costs import calculate_turnover
from portfolio_rl.evaluation.backtest import BacktestResult, run_weight_policy_backtest
from portfolio_rl.evaluation.metrics import calculate_performance_metrics
from portfolio_rl.policies.baseline_policies import WeightPolicy


@dataclass(frozen=True)
class ExecutionCostScenario:
    """One mutually exclusive flat or ticker-specific cost assumption."""

    name: str
    flat_bps: float | None = None
    asset_bps: Mapping[str, float] | None = None

    def validate(self, asset_order: list[str]) -> None:
        """Validate scenario completeness and numerical assumptions."""
        if not self.name.strip():
            raise ValueError("cost scenario name must not be empty")
        if (self.flat_bps is None) == (self.asset_bps is None):
            raise ValueError("cost scenario must define exactly one cost convention")
        if self.flat_bps is not None:
            _validate_bps(self.flat_bps, "flat_bps")
            return
        assert self.asset_bps is not None
        expected = set(asset_order)
        observed = set(self.asset_bps)
        if observed != expected:
            raise ValueError(
                "asset_bps tickers must exactly match asset_order: "
                f"missing={sorted(expected - observed)}, "
                f"unexpected={sorted(observed - expected)}"
            )
        for ticker in asset_order:
            _validate_bps(self.asset_bps[ticker], f"asset_bps[{ticker}]")


@dataclass(frozen=True)
class ExecutionStressPath:
    """Backtest outputs plus daily holdings and execution chronology."""

    backtest: BacktestResult
    daily_positions: pd.DataFrame
    execution_audit: pd.DataFrame


def calculate_asset_specific_cost_fraction(
    current_weights: np.ndarray,
    target_weights: np.ndarray,
    asset_cost_bps: np.ndarray,
) -> float:
    """Calculate half-L1 one-way cost with per-asset basis-point rates."""
    current = np.asarray(current_weights, dtype=np.float64)
    target = np.asarray(target_weights, dtype=np.float64)
    bps = np.asarray(asset_cost_bps, dtype=np.float64)
    if current.ndim != 1 or target.shape != current.shape or bps.shape != current.shape:
        raise ValueError("current, target, and asset costs must share one-dimensional shape")
    if not np.isfinite(current).all() or not np.isfinite(target).all():
        raise ValueError("weights must be finite")
    if not np.isfinite(bps).all() or (bps < 0.0).any():
        raise ValueError("asset costs must be finite and nonnegative")
    return float(0.5 * np.sum(np.abs(target - current) * bps) / 10_000.0)


def run_execution_stress_backtest(
    *,
    feature_store: PortfolioFeatureStore,
    policy: WeightPolicy,
    strategy: str,
    cost_scenario: ExecutionCostScenario,
    execution_delay_closes: int,
    rebalance_frequency_trading_days: int = 5,
    initial_nav: float = 1.0,
    inverse_vol_lookback_trading_days: int = 63,
    max_steps: int | None = None,
) -> ExecutionStressPath:
    """Run a fresh closed-loop path under a zero- or one-close execution delay."""
    if execution_delay_closes not in (0, 1):
        raise ValueError("execution_delay_closes must be zero or one")
    if rebalance_frequency_trading_days <= execution_delay_closes:
        raise ValueError("rebalance period must exceed execution delay")
    if not np.isfinite(initial_nav) or initial_nav <= 0.0:
        raise ValueError("initial_nav must be positive and finite")
    if max_steps is not None and max_steps <= 0:
        raise ValueError("max_steps must be positive when provided")
    cost_scenario.validate(feature_store.asset_order)
    if execution_delay_closes == 0 and cost_scenario.flat_bps is not None:
        backtest = run_weight_policy_backtest(
            feature_store=feature_store,
            policy=policy,
            strategy=strategy,
            rebalance_frequency_trading_days=rebalance_frequency_trading_days,
            transaction_cost_bps=cost_scenario.flat_bps,
            initial_nav=initial_nav,
            max_steps=max_steps,
            inverse_vol_lookback_trading_days=inverse_vol_lookback_trading_days,
        )
        return ExecutionStressPath(
            backtest=backtest,
            daily_positions=reconstruct_daily_positions(
                feature_store=feature_store,
                backtest=backtest,
                strategy=strategy,
                rebalance_days=rebalance_frequency_trading_days,
            ),
            execution_audit=_standard_execution_audit(
                backtest=backtest,
                strategy=strategy,
                scenario=cost_scenario.name,
            ),
        )
    if hasattr(policy, "reset"):
        policy.reset()

    current_idx = 0
    current_weights = np.full(
        feature_store.n_assets,
        1.0 / feature_store.n_assets,
        dtype=np.float64,
    )
    portfolio_value = float(initial_nav)
    peak_nav = float(initial_nav)
    step = 0
    nav_rows: list[dict[str, Any]] = []
    target_rows: list[dict[str, Any]] = []
    drift_rows: list[dict[str, Any]] = []
    trade_rows: list[dict[str, Any]] = []
    cost_rows: list[dict[str, Any]] = []
    position_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []

    while max_steps is None or step < max_steps:
        try:
            forward = feature_store.get_forward_log_returns(
                current_idx,
                rebalance_frequency_trading_days,
            )
        except IndexError:
            break
        decision_date = feature_store.date_at(current_idx)
        observation = np.concatenate(
            [feature_store.get_market_features(current_idx), current_weights]
        ).astype(np.float32)
        info = {
            "date": decision_date,
            "portfolio_value": portfolio_value,
            "current_weights": current_weights.copy(),
            "asset_order": feature_store.asset_order,
            "trailing_log_returns": _available_trailing_returns(
                feature_store,
                current_idx,
                inverse_vol_lookback_trading_days,
            ),
        }
        target = np.asarray(policy.target_weights(observation, info), dtype=np.float64)
        _validate_weights(target, feature_store.n_assets)

        for ticker, weight in zip(feature_store.asset_order, target, strict=True):
            target_rows.append(
                {
                    "date": decision_date,
                    "strategy": strategy,
                    "ticker": ticker,
                    "target_weight": float(weight),
                }
            )

        delay = execution_delay_closes
        delayed_returns = forward[:delay]
        holding_returns = forward[delay:]
        if delay:
            nav_before_delay = portfolio_value
            peak_before_delay = peak_nav
            portfolio_value, peak_nav, current_weights = _apply_daily_returns(
                feature_store=feature_store,
                current_idx=current_idx,
                start_offset=1,
                log_returns=delayed_returns,
                start_weights=current_weights,
                strategy=strategy,
                portfolio_value=portfolio_value,
                peak_nav=peak_nav,
                cost_fraction_on_first_day=0.0,
                nav_rows=nav_rows,
                position_rows=position_rows,
            )
        pre_trade = current_weights.copy()
        execution_date = feature_store.date_at(current_idx + delay)
        turnover = calculate_turnover(pre_trade, target)
        cost_fraction = _cost_fraction(
            pre_trade,
            target,
            feature_store.asset_order,
            cost_scenario,
        )
        for ticker, pre_weight, target_weight in zip(
            feature_store.asset_order,
            pre_trade,
            target,
            strict=True,
        ):
            trade_rows.append(
                {
                    "date": execution_date,
                    "decision_date": decision_date,
                    "strategy": strategy,
                    "ticker": ticker,
                    "pre_trade_weight": float(pre_weight),
                    "target_weight": float(target_weight),
                    "trade_weight": float(target_weight - pre_weight),
                }
            )
        cost_rows.append(
            {
                "date": execution_date,
                "decision_date": decision_date,
                "strategy": strategy,
                "turnover": float(turnover),
                "transaction_cost_fraction": float(cost_fraction),
            }
        )
        audit_rows.append(
            {
                "decision_date": decision_date,
                "execution_date": execution_date,
                "decision_step": step,
                "strategy": strategy,
                "execution_delay_closes": delay,
                "cost_scenario": cost_scenario.name,
                "turnover": float(turnover),
                "transaction_cost_fraction": float(cost_fraction),
            }
        )

        if delay:
            # The close-t+1 market return has already accrued. Apply execution cost
            # at that close before the new target earns the next daily return.
            portfolio_value *= 1.0 - cost_fraction
            peak_nav = max(peak_before_delay, portfolio_value)
            nav_rows[-1]["nav"] = float(portfolio_value)
            nav_rows[-1]["daily_return"] = float(
                portfolio_value / nav_before_delay - 1.0
            )
            nav_rows[-1]["drawdown"] = float(portfolio_value / peak_nav - 1.0)
            first_holding_offset = delay + 1
            first_holding_cost = 0.0
        else:
            first_holding_offset = 1
            first_holding_cost = cost_fraction

        portfolio_value, peak_nav, current_weights = _apply_daily_returns(
            feature_store=feature_store,
            current_idx=current_idx,
            start_offset=first_holding_offset,
            log_returns=holding_returns,
            start_weights=target,
            strategy=strategy,
            portfolio_value=portfolio_value,
            peak_nav=peak_nav,
            cost_fraction_on_first_day=first_holding_cost,
            nav_rows=nav_rows,
            position_rows=position_rows,
        )
        current_weights = current_weights.astype(np.float32).astype(np.float64)
        end_date = feature_store.date_at(
            current_idx + rebalance_frequency_trading_days
        )
        for ticker, weight in zip(
            feature_store.asset_order,
            current_weights,
            strict=True,
        ):
            drift_rows.append(
                {
                    "date": end_date,
                    "strategy": strategy,
                    "ticker": ticker,
                    "drifted_weight": float(weight),
                }
            )
        current_idx += rebalance_frequency_trading_days
        step += 1

    nav = pd.DataFrame(nav_rows)
    costs = pd.DataFrame(cost_rows)
    result = BacktestResult(
        nav=nav,
        weights_target=pd.DataFrame(target_rows),
        weights_drifted=pd.DataFrame(drift_rows),
        trades=pd.DataFrame(trade_rows),
        costs=costs,
        metrics=calculate_performance_metrics(nav, costs) if not nav.empty else {},
    )
    return ExecutionStressPath(
        backtest=result,
        daily_positions=pd.DataFrame(position_rows),
        execution_audit=pd.DataFrame(audit_rows),
    )


def _apply_daily_returns(
    *,
    feature_store: PortfolioFeatureStore,
    current_idx: int,
    start_offset: int,
    log_returns: np.ndarray,
    start_weights: np.ndarray,
    strategy: str,
    portfolio_value: float,
    peak_nav: float,
    cost_fraction_on_first_day: float,
    nav_rows: list[dict[str, Any]],
    position_rows: list[dict[str, Any]],
) -> tuple[float, float, np.ndarray]:
    weights = np.asarray(start_weights, dtype=np.float64).copy()
    for local_offset, daily_log_returns in enumerate(log_returns):
        offset = start_offset + local_offset
        date = feature_store.date_at(current_idx + offset)
        asset_gross_returns = np.exp(daily_log_returns)
        asset_simple_returns = asset_gross_returns - 1.0
        portfolio_gross_return = float(np.dot(weights, asset_gross_returns))
        gross_return = np.float32(portfolio_gross_return - 1.0)
        for ticker, weight, asset_return in zip(
            feature_store.asset_order,
            weights,
            asset_simple_returns,
            strict=True,
        ):
            position_rows.append(
                {
                    "date": date,
                    "strategy": strategy,
                    "ticker": ticker,
                    "pre_return_weight": float(weight),
                    "asset_simple_return": float(asset_return),
                    "gross_return_contribution": float(weight * asset_return),
                }
            )
        previous_nav = portfolio_value
        multiplier = 1.0 + float(gross_return)
        if local_offset == 0:
            multiplier *= 1.0 - cost_fraction_on_first_day
        portfolio_value *= multiplier
        peak_nav = max(peak_nav, portfolio_value)
        nav_rows.append(
            {
                "date": date,
                "strategy": strategy,
                "nav": float(portfolio_value),
                "daily_return": float(portfolio_value / previous_nav - 1.0),
                "drawdown": float(portfolio_value / peak_nav - 1.0),
            }
        )
        weights = weights * asset_gross_returns / portfolio_gross_return
    return portfolio_value, peak_nav, weights


def _cost_fraction(
    current: np.ndarray,
    target: np.ndarray,
    asset_order: list[str],
    scenario: ExecutionCostScenario,
) -> float:
    if scenario.flat_bps is not None:
        return float(calculate_turnover(current, target) * scenario.flat_bps / 10_000.0)
    assert scenario.asset_bps is not None
    rates = np.asarray([scenario.asset_bps[ticker] for ticker in asset_order])
    return calculate_asset_specific_cost_fraction(current, target, rates)


def reconstruct_daily_positions(
    *,
    feature_store: PortfolioFeatureStore,
    backtest: BacktestResult,
    strategy: str,
    rebalance_days: int,
) -> pd.DataFrame:
    """Reconstruct exact pre-return holdings and gross asset contributions."""
    rows = []
    target_frame = backtest.weights_target.pivot(
        index="date", columns="ticker", values="target_weight"
    ).reindex(columns=feature_store.asset_order)
    current_idx = 0
    for _, target_row in target_frame.sort_index().iterrows():
        weights = target_row.to_numpy(dtype=np.float64)
        forward = feature_store.get_forward_log_returns(current_idx, rebalance_days)
        for offset, daily_log_returns in enumerate(forward, start=1):
            date = feature_store.date_at(current_idx + offset)
            gross = np.exp(daily_log_returns)
            simple = gross - 1.0
            for ticker, weight, asset_return in zip(
                feature_store.asset_order, weights, simple, strict=True
            ):
                rows.append(
                    {
                        "date": date,
                        "strategy": strategy,
                        "ticker": ticker,
                        "pre_return_weight": float(weight),
                        "asset_simple_return": float(asset_return),
                        "gross_return_contribution": float(weight * asset_return),
                    }
                )
            portfolio_gross = float(np.dot(weights, gross))
            weights = weights * gross / portfolio_gross
        current_idx += rebalance_days
    return pd.DataFrame(rows)


def _standard_execution_audit(
    *, backtest: BacktestResult, strategy: str, scenario: str
) -> pd.DataFrame:
    rows = []
    for step, row in backtest.costs.reset_index(drop=True).iterrows():
        rows.append(
            {
                "decision_date": row["date"],
                "execution_date": row["date"],
                "decision_step": step,
                "strategy": strategy,
                "execution_delay_closes": 0,
                "cost_scenario": scenario,
                "turnover": float(row["turnover"]),
                "transaction_cost_fraction": float(
                    row["transaction_cost_fraction"]
                ),
            }
        )
    return pd.DataFrame(rows)


def _available_trailing_returns(
    store: PortfolioFeatureStore,
    relative_idx: int,
    lookback: int,
) -> np.ndarray:
    within = min(relative_idx, lookback)
    before = lookback - within
    frames = []
    if before:
        frames.append(store.get_pre_window_log_returns(before))
    if within:
        frames.append(store.get_trailing_log_returns(relative_idx - 1, within))
    return np.vstack(frames)


def _validate_weights(weights: np.ndarray, n_assets: int) -> None:
    if weights.shape != (n_assets,):
        raise ValueError(f"target weights must have shape ({n_assets},)")
    if not np.isfinite(weights).all() or (weights < 0.0).any():
        raise ValueError("target weights must be finite and nonnegative")
    if not np.isclose(weights.sum(), 1.0):
        raise ValueError("target weights must sum to one")


def _validate_bps(value: float, name: str) -> None:
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")

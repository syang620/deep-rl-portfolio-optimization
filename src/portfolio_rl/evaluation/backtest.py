"""Deterministic in-memory backtest engine for target-weight policies."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from portfolio_rl.data.feature_store import PortfolioFeatureStore
from portfolio_rl.env.costs import (
    calculate_transaction_cost_fraction,
    calculate_turnover,
)
from portfolio_rl.env.drift import simulate_buy_and_hold_period
from portfolio_rl.evaluation.metrics import calculate_performance_metrics
from portfolio_rl.policies.baseline_policies import WeightPolicy


@dataclass(frozen=True)
class BacktestResult:
    """In-memory outputs from one deterministic policy backtest."""

    nav: pd.DataFrame
    weights_target: pd.DataFrame
    weights_drifted: pd.DataFrame
    trades: pd.DataFrame
    costs: pd.DataFrame
    metrics: dict[str, float | None]


BACKTEST_ARTIFACT_FILENAMES = {
    "nav": "nav.parquet",
    "weights_target": "weights_target.parquet",
    "weights_drifted": "weights_drifted.parquet",
    "trades": "trades.parquet",
    "costs": "costs.parquet",
    "metrics": "metrics.json",
    "report": "report.md",
}


def run_weight_policy_backtest(
    *,
    feature_store: PortfolioFeatureStore,
    policy: WeightPolicy,
    strategy: str,
    rebalance_frequency_trading_days: int = 5,
    transaction_cost_bps: float = 10.0,
    initial_nav: float = 1.0,
    max_steps: int | None = None,
    inverse_vol_lookback_trading_days: int = 21,
) -> BacktestResult:
    """Run a deterministic target-weight policy through portfolio mechanics."""
    if rebalance_frequency_trading_days <= 0:
        raise ValueError("rebalance_frequency_trading_days must be positive")
    if not np.isfinite(transaction_cost_bps) or transaction_cost_bps < 0.0:
        raise ValueError("transaction_cost_bps must be nonnegative and finite")
    if not np.isfinite(initial_nav) or initial_nav <= 0.0:
        raise ValueError("initial_nav must be positive and finite")
    if max_steps is not None and max_steps <= 0:
        raise ValueError("max_steps must be positive when provided")
    if inverse_vol_lookback_trading_days <= 0:
        raise ValueError("inverse_vol_lookback_trading_days must be positive")
    if hasattr(policy, "reset"):
        policy.reset()

    current_idx = 0
    current_weights = _equal_weight_vector(feature_store.n_assets)
    portfolio_value = float(initial_nav)
    peak_nav = float(initial_nav)
    step_count = 0

    nav_rows: list[dict[str, Any]] = []
    target_weight_rows: list[dict[str, Any]] = []
    drifted_weight_rows: list[dict[str, Any]] = []
    trade_rows: list[dict[str, Any]] = []
    cost_rows: list[dict[str, Any]] = []

    while max_steps is None or step_count < max_steps:
        try:
            forward_log_returns = feature_store.get_forward_log_returns(
                current_idx,
                rebalance_frequency_trading_days,
            )
        except IndexError:
            break

        observation = _build_observation(feature_store, current_idx, current_weights)
        decision_date = feature_store.date_at(current_idx)
        policy_info = {
            "date": decision_date,
            "portfolio_value": portfolio_value,
            "current_weights": current_weights.copy(),
            "asset_order": feature_store.asset_order,
            "trailing_log_returns": _get_available_trailing_log_returns(
                feature_store,
                current_idx,
                inverse_vol_lookback_trading_days,
            ),
        }
        target_weights = np.asarray(
            policy.target_weights(observation, policy_info),
            dtype=np.float64,
        )
        _validate_target_weights(target_weights, feature_store.n_assets)

        pre_trade_weights = current_weights.copy()
        turnover = calculate_turnover(pre_trade_weights, target_weights)
        cost_fraction = calculate_transaction_cost_fraction(
            turnover,
            transaction_cost_bps,
        )
        period_gross_return, drifted_weights, daily_portfolio_returns = (
            simulate_buy_and_hold_period(target_weights, forward_log_returns)
        )
        del period_gross_return

        for ticker, pre_weight, target_weight in zip(
            feature_store.asset_order,
            pre_trade_weights,
            target_weights,
            strict=True,
        ):
            target_weight_rows.append(
                {
                    "date": decision_date,
                    "strategy": strategy,
                    "ticker": ticker,
                    "target_weight": float(target_weight),
                }
            )
            trade_rows.append(
                {
                    "date": decision_date,
                    "strategy": strategy,
                    "ticker": ticker,
                    "pre_trade_weight": float(pre_weight),
                    "target_weight": float(target_weight),
                    "trade_weight": float(target_weight - pre_weight),
                }
            )

        cost_rows.append(
            {
                "date": decision_date,
                "strategy": strategy,
                "turnover": float(turnover),
                "transaction_cost_fraction": float(cost_fraction),
            }
        )

        for offset, daily_return in enumerate(daily_portfolio_returns, start=1):
            previous_nav = portfolio_value
            day_multiplier = 1.0 + float(daily_return)
            if offset == 1:
                day_multiplier *= 1.0 - cost_fraction
            portfolio_value *= day_multiplier
            peak_nav = max(peak_nav, portfolio_value)
            nav_rows.append(
                {
                    "date": feature_store.date_at(current_idx + offset),
                    "strategy": strategy,
                    "nav": float(portfolio_value),
                    "daily_return": float(portfolio_value / previous_nav - 1.0),
                    "drawdown": float(portfolio_value / peak_nav - 1.0),
                }
            )

        end_date = feature_store.date_at(
            current_idx + rebalance_frequency_trading_days,
        )
        for ticker, drifted_weight in zip(
            feature_store.asset_order,
            drifted_weights,
            strict=True,
        ):
            drifted_weight_rows.append(
                {
                    "date": end_date,
                    "strategy": strategy,
                    "ticker": ticker,
                    "drifted_weight": float(drifted_weight),
                }
            )

        current_weights = drifted_weights.astype(np.float64)
        current_idx += rebalance_frequency_trading_days
        step_count += 1

    nav = pd.DataFrame(
        nav_rows,
        columns=["date", "strategy", "nav", "daily_return", "drawdown"],
    )
    weights_target = pd.DataFrame(
        target_weight_rows,
        columns=["date", "strategy", "ticker", "target_weight"],
    )
    weights_drifted = pd.DataFrame(
        drifted_weight_rows,
        columns=["date", "strategy", "ticker", "drifted_weight"],
    )
    trades = pd.DataFrame(
        trade_rows,
        columns=[
            "date",
            "strategy",
            "ticker",
            "pre_trade_weight",
            "target_weight",
            "trade_weight",
        ],
    )
    costs = pd.DataFrame(
        cost_rows,
        columns=[
            "date",
            "strategy",
            "turnover",
            "transaction_cost_fraction",
        ],
    )
    metrics = calculate_performance_metrics(nav, costs) if not nav.empty else {}

    return BacktestResult(
        nav=nav,
        weights_target=weights_target,
        weights_drifted=weights_drifted,
        trades=trades,
        costs=costs,
        metrics=metrics,
    )


def write_backtest_artifacts(
    result: BacktestResult,
    output_dir: str | Path,
) -> None:
    """Write required Milestone 6 backtest artifacts to disk."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    result.nav.to_parquet(output_path / BACKTEST_ARTIFACT_FILENAMES["nav"], index=False)
    result.weights_target.to_parquet(
        output_path / BACKTEST_ARTIFACT_FILENAMES["weights_target"],
        index=False,
    )
    result.weights_drifted.to_parquet(
        output_path / BACKTEST_ARTIFACT_FILENAMES["weights_drifted"],
        index=False,
    )
    result.trades.to_parquet(
        output_path / BACKTEST_ARTIFACT_FILENAMES["trades"],
        index=False,
    )
    result.costs.to_parquet(
        output_path / BACKTEST_ARTIFACT_FILENAMES["costs"],
        index=False,
    )
    (output_path / BACKTEST_ARTIFACT_FILENAMES["metrics"]).write_text(
        json.dumps(result.metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_path / BACKTEST_ARTIFACT_FILENAMES["report"]).write_text(
        _build_backtest_report(result),
        encoding="utf-8",
    )


def _build_observation(
    feature_store: PortfolioFeatureStore,
    relative_idx: int,
    current_weights: np.ndarray,
) -> np.ndarray:
    return np.concatenate(
        [feature_store.get_market_features(relative_idx), current_weights],
    ).astype(np.float32)


def _get_available_trailing_log_returns(
    feature_store: PortfolioFeatureStore,
    relative_idx: int,
    lookback: int,
) -> np.ndarray:
    available_lookback = min(lookback, relative_idx + 1)
    return feature_store.get_trailing_log_returns(relative_idx, available_lookback)


def _build_backtest_report(result: BacktestResult) -> str:
    strategy = (
        str(result.nav["strategy"].iloc[0])
        if not result.nav.empty and "strategy" in result.nav
        else "unknown"
    )
    final_nav = (
        float(result.nav["nav"].iloc[-1])
        if not result.nav.empty and "nav" in result.nav
        else None
    )
    metric_labels = [
        ("Final NAV", final_nav),
        ("Total Return", result.metrics.get("total_return")),
        ("CAGR", result.metrics.get("cagr")),
        ("Sharpe Ratio", result.metrics.get("sharpe_ratio")),
        ("Max Drawdown", result.metrics.get("max_drawdown")),
        ("Average Weekly Turnover", result.metrics.get("average_weekly_turnover")),
        ("Transaction Cost Drag", result.metrics.get("transaction_cost_drag")),
    ]
    lines = [
        f"# Backtest Report: {strategy}",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    lines.extend(
        f"| {label} | {_format_report_value(value)} |"
        for label, value in metric_labels
    )
    lines.append("")
    return "\n".join(lines)


def _format_report_value(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.6f}"


def _equal_weight_vector(n_assets: int) -> np.ndarray:
    return np.full(n_assets, 1.0 / n_assets, dtype=np.float64)


def _validate_target_weights(weights: np.ndarray, n_assets: int) -> None:
    if weights.ndim != 1:
        raise ValueError("target_weights must be a one-dimensional array")
    if weights.shape != (n_assets,):
        raise ValueError(f"target_weights must have shape ({n_assets},)")
    if not np.isfinite(weights).all():
        raise ValueError("target_weights values must be finite")
    if (weights < 0.0).any():
        raise ValueError("target_weights values must be nonnegative")
    if not np.isclose(weights.sum(), 1.0):
        raise ValueError("target_weights must sum to one")

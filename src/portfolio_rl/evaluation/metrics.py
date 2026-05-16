"""Performance metrics for daily NAV backtests."""

from __future__ import annotations

import numpy as np
import pandas as pd


TRADING_DAYS_PER_YEAR = 252
WEEKLY_REBALANCES_PER_YEAR = 52


def calculate_performance_metrics(
    nav: pd.DataFrame,
    costs: pd.DataFrame,
) -> dict[str, float | None]:
    """Calculate Phase 2 backtest metrics from daily NAV and weekly costs."""
    _require_columns(nav, ["date", "nav", "daily_return", "drawdown"], "nav")
    _require_columns(
        costs,
        ["turnover", "transaction_cost_fraction"],
        "costs",
    )
    if nav.empty:
        raise ValueError("nav must not be empty")

    nav_sorted = nav.sort_values("date").reset_index(drop=True)
    nav_values = nav_sorted["nav"].to_numpy(dtype=np.float64)
    daily_returns = nav_sorted["daily_return"].to_numpy(dtype=np.float64)
    drawdowns = nav_sorted["drawdown"].to_numpy(dtype=np.float64)
    _assert_finite(nav_values, "nav values")
    _assert_finite(daily_returns, "daily_return values")
    _assert_finite(drawdowns, "drawdown values")
    if (nav_values <= 0.0).any():
        raise ValueError("nav values must be positive")
    if (daily_returns <= -1.0).any():
        raise ValueError("daily_return values must be greater than -1")

    initial_nav = _infer_initial_nav(nav_values[0], daily_returns[0])
    final_nav = float(nav_values[-1])
    total_return = final_nav / initial_nav - 1.0
    n_days = len(nav_sorted)
    cagr = (final_nav / initial_nav) ** (TRADING_DAYS_PER_YEAR / n_days) - 1.0
    annualized_volatility = _annualized_volatility(daily_returns)
    sharpe_ratio = _annualized_ratio(daily_returns, annualized_volatility)
    sortino_ratio = _sortino_ratio(daily_returns)
    max_drawdown = float(np.min(drawdowns))
    calmar_ratio = _calmar_ratio(cagr, max_drawdown)

    turnovers = costs["turnover"].to_numpy(dtype=np.float64)
    cost_fractions = costs["transaction_cost_fraction"].to_numpy(dtype=np.float64)
    _assert_finite(turnovers, "turnover values")
    _assert_finite(cost_fractions, "transaction_cost_fraction values")
    if (turnovers < 0.0).any():
        raise ValueError("turnover values must be nonnegative")
    if (cost_fractions < 0.0).any():
        raise ValueError("transaction_cost_fraction values must be nonnegative")

    average_weekly_turnover = (
        float(np.mean(turnovers)) if len(turnovers) > 0 else None
    )
    annualized_turnover = (
        average_weekly_turnover * WEEKLY_REBALANCES_PER_YEAR
        if average_weekly_turnover is not None
        else None
    )
    transaction_cost_drag = float(np.sum(cost_fractions))
    monthly_returns = _monthly_returns(nav_sorted, initial_nav)

    return {
        "total_return": float(total_return),
        "cagr": float(cagr),
        "annualized_volatility": annualized_volatility,
        "sharpe_ratio": sharpe_ratio,
        "sortino_ratio": sortino_ratio,
        "max_drawdown": max_drawdown,
        "calmar_ratio": calmar_ratio,
        "average_weekly_turnover": average_weekly_turnover,
        "annualized_turnover": annualized_turnover,
        "transaction_cost_drag": transaction_cost_drag,
        "hit_rate": float(np.mean(daily_returns > 0.0)),
        "best_month": (
            float(monthly_returns.max()) if not monthly_returns.empty else None
        ),
        "worst_month": (
            float(monthly_returns.min()) if not monthly_returns.empty else None
        ),
    }


def _infer_initial_nav(first_nav: float, first_daily_return: float) -> float:
    initial_nav = first_nav / (1.0 + first_daily_return)
    if not np.isfinite(initial_nav) or initial_nav <= 0.0:
        raise ValueError("inferred initial NAV must be positive and finite")
    return float(initial_nav)


def _annualized_volatility(daily_returns: np.ndarray) -> float | None:
    if len(daily_returns) < 2:
        return None
    volatility = float(np.std(daily_returns, ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))
    return volatility if volatility > 0.0 else None


def _annualized_ratio(
    daily_returns: np.ndarray,
    annualized_volatility: float | None,
) -> float | None:
    if annualized_volatility is None:
        return None
    annualized_return = float(np.mean(daily_returns) * TRADING_DAYS_PER_YEAR)
    return annualized_return / annualized_volatility


def _sortino_ratio(daily_returns: np.ndarray) -> float | None:
    downside_returns = daily_returns[daily_returns < 0.0]
    if len(downside_returns) < 2:
        return None
    downside_deviation = float(
        np.std(downside_returns, ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)
    )
    if downside_deviation <= 0.0:
        return None
    annualized_return = float(np.mean(daily_returns) * TRADING_DAYS_PER_YEAR)
    return annualized_return / downside_deviation


def _calmar_ratio(cagr: float, max_drawdown: float) -> float | None:
    if max_drawdown >= 0.0:
        return None
    return cagr / abs(max_drawdown)


def _monthly_returns(nav: pd.DataFrame, initial_nav: float) -> pd.Series:
    nav_by_date = nav.assign(date=pd.to_datetime(nav["date"])).set_index("date")
    month_end_nav = nav_by_date["nav"].resample("ME").last().dropna()
    if month_end_nav.empty:
        return pd.Series(dtype=np.float64)
    previous_nav = month_end_nav.shift(1)
    previous_nav.iloc[0] = initial_nav
    return month_end_nav / previous_nav - 1.0


def _require_columns(
    frame: pd.DataFrame,
    columns: list[str],
    frame_name: str,
) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{frame_name} missing required columns: {missing}")


def _assert_finite(values: np.ndarray, name: str) -> None:
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must be finite")

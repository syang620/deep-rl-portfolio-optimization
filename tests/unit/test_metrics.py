from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio_rl.evaluation.metrics import calculate_performance_metrics


def test_metrics_total_return_and_max_drawdown() -> None:
    metrics = calculate_performance_metrics(_nav_frame(), _costs_frame())

    assert metrics["total_return"] == pytest.approx(0.188)
    assert metrics["max_drawdown"] == pytest.approx(-0.1)
    assert metrics["calmar_ratio"] == pytest.approx(
        metrics["cagr"] / abs(metrics["max_drawdown"])
    )


def test_metrics_annualized_volatility_and_sharpe() -> None:
    nav = _nav_frame(daily_returns=[0.01, -0.02, 0.03, 0.0])

    metrics = calculate_performance_metrics(nav, _costs_frame())

    expected_volatility = np.std([0.01, -0.02, 0.03, 0.0], ddof=1) * np.sqrt(252)
    expected_sharpe = np.mean([0.01, -0.02, 0.03, 0.0]) * 252 / expected_volatility
    assert metrics["annualized_volatility"] == pytest.approx(expected_volatility)
    assert metrics["sharpe_ratio"] == pytest.approx(expected_sharpe)


def test_metrics_sortino_uses_negative_returns() -> None:
    nav = _nav_frame(daily_returns=[0.01, -0.02, 0.03, -0.01])

    metrics = calculate_performance_metrics(nav, _costs_frame())

    downside_deviation = np.std([-0.02, -0.01], ddof=1) * np.sqrt(252)
    expected_sortino = np.mean([0.01, -0.02, 0.03, -0.01]) * 252 / downside_deviation
    assert metrics["sortino_ratio"] == pytest.approx(expected_sortino)


def test_metrics_turnover_and_cost_drag() -> None:
    metrics = calculate_performance_metrics(
        _nav_frame(),
        _costs_frame(turnovers=[0.2, 0.4], costs=[0.001, 0.002]),
    )

    assert metrics["average_weekly_turnover"] == pytest.approx(0.3)
    assert metrics["annualized_turnover"] == pytest.approx(0.3 * 52)
    assert metrics["transaction_cost_drag"] == pytest.approx(0.003)


def test_metrics_best_and_worst_month() -> None:
    nav = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2024-01-31", "2024-02-29", "2024-03-29"]
            ),
            "strategy": ["test", "test", "test"],
            "nav": [1.10, 0.99, 1.188],
            "daily_return": [0.10, -0.10, 0.20],
            "drawdown": [0.0, -0.10, 0.0],
        }
    )

    metrics = calculate_performance_metrics(nav, _costs_frame())

    assert metrics["best_month"] == pytest.approx(0.20)
    assert metrics["worst_month"] == pytest.approx(-0.10)


def test_metrics_reject_empty_nav() -> None:
    with pytest.raises(ValueError, match="nav must not be empty"):
        calculate_performance_metrics(
            pd.DataFrame(columns=["date", "nav", "daily_return", "drawdown"]),
            _costs_frame(),
        )


def test_metrics_returns_none_for_undefined_ratios() -> None:
    nav = _nav_frame(daily_returns=[0.0])

    metrics = calculate_performance_metrics(nav, _costs_frame())

    assert metrics["annualized_volatility"] is None
    assert metrics["sharpe_ratio"] is None
    assert metrics["sortino_ratio"] is None
    assert metrics["calmar_ratio"] is None


def _nav_frame(daily_returns: list[float] | None = None) -> pd.DataFrame:
    returns = daily_returns or [0.10, -0.10, 0.20]
    nav = 1.0
    rows = []
    peak = 1.0
    for index, daily_return in enumerate(returns):
        nav *= 1.0 + daily_return
        peak = max(peak, nav)
        rows.append(
            {
                "date": pd.Timestamp("2024-01-02") + pd.offsets.BDay(index),
                "strategy": "test",
                "nav": nav,
                "daily_return": daily_return,
                "drawdown": nav / peak - 1.0,
            }
        )
    return pd.DataFrame(rows)


def _costs_frame(
    turnovers: list[float] | None = None,
    costs: list[float] | None = None,
) -> pd.DataFrame:
    turnover_values = turnovers or [0.2]
    cost_values = costs or [0.001]
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-02", periods=len(turnover_values), freq="B"),
            "strategy": ["test"] * len(turnover_values),
            "turnover": turnover_values,
            "transaction_cost_fraction": cost_values,
        }
    )

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from portfolio_rl.config.schemas import PolicyBehaviorConfig
from portfolio_rl.evaluation.backtest import BacktestResult
from portfolio_rl.evaluation.diagnostics import (
    PolicyBehaviorResult,
    _daily_nav_frame,
    _reconcile_validation_metrics,
    build_policy_behavior_summary,
    calculate_concentration_metrics,
    calculate_drawdown_periods,
    calculate_monthly_returns,
    calculate_turnover_distribution,
    format_policy_behavior_report,
)
from scripts import analyze_policy_behavior as diagnostics_script

METADATA = {
    "run_id": "run_7",
    "seed": 7,
    "selection_checkpoint": "best_checkpoint",
    "regime_name": "validation_2024",
    "split": "validation",
    "in_sample": False,
}


def test_concentration_metrics_include_cash_and_equity_exposure() -> None:
    allocation = pd.DataFrame(
        {
            **_repeated_metadata(6),
            "date": pd.to_datetime(
                [
                    "2024-01-02",
                    "2024-01-02",
                    "2024-01-02",
                    "2024-01-09",
                    "2024-01-09",
                    "2024-01-09",
                ]
            ),
            "ticker": ["SPY", "SHY", "GLD", "SPY", "SHY", "GLD"],
            "target_weight": [0.8, 0.1, 0.1, 0.4, 0.5, 0.1],
            "high_volatility": [False, False, False, True, True, True],
        }
    )

    result = calculate_concentration_metrics(
        allocation,
        asset_classes={
            "SPY": "us_large_cap_equity",
            "SHY": "short_treasury_cash_proxy",
            "GLD": "gold",
        },
        active_weight_threshold=0.05,
    )

    assert result.loc[0, "max_ticker"] == "SPY"
    assert result.loc[0, "hhi"] == pytest.approx(0.66)
    assert result.loc[0, "effective_asset_count"] == pytest.approx(1 / 0.66)
    assert result.loc[0, "active_asset_count"] == 3
    assert result.loc[0, "shy_weight"] == pytest.approx(0.1)
    assert result.loc[0, "equity_like_weight"] == pytest.approx(0.8)


def test_concentration_metrics_reject_invalid_weight_sum() -> None:
    allocation = pd.DataFrame(
        {
            **_repeated_metadata(2),
            "date": pd.to_datetime(["2024-01-02", "2024-01-02"]),
            "ticker": ["SPY", "SHY"],
            "target_weight": [0.8, 0.3],
            "high_volatility": [False, False],
        }
    )

    with pytest.raises(ValueError, match="sum to one"):
        calculate_concentration_metrics(
            allocation,
            asset_classes={
                "SPY": "us_large_cap_equity",
                "SHY": "short_treasury_cash_proxy",
            },
            active_weight_threshold=0.05,
        )


def test_turnover_distribution_uses_half_l1_target_change() -> None:
    dates = pd.to_datetime(["2024-01-02", "2024-01-09"])
    allocation = pd.DataFrame(
        {
            **_repeated_metadata(4),
            "date": [dates[0], dates[0], dates[1], dates[1]],
            "ticker": ["SPY", "SHY", "SPY", "SHY"],
            "target_weight": [0.5, 0.5, 0.75, 0.25],
            "high_volatility": [False, False, True, True],
        }
    )
    backtest = _backtest(
        costs=pd.DataFrame(
            {
                "date": dates,
                "turnover": [0.1, 0.25],
                "transaction_cost_fraction": [0.0001, 0.00025],
            }
        )
    )

    result = calculate_turnover_distribution(
        backtest=backtest,
        allocation=allocation,
        metadata=METADATA,
        spike_quantile=0.95,
    )

    assert result["target_change_turnover"].tolist() == pytest.approx([0.0, 0.25])
    assert result["is_turnover_spike"].tolist() == [False, True]


def test_monthly_returns_and_drawdown_periods() -> None:
    nav = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2024-01-30",
                    "2024-01-31",
                    "2024-02-01",
                    "2024-02-02",
                ]
            ),
            "daily_return": [0.10, -0.05, -0.02, 0.03],
            "drawdown": [0.0, -0.05, -0.069, 0.0],
        }
    )

    monthly = calculate_monthly_returns(nav, metadata=METADATA)
    drawdowns = calculate_drawdown_periods(nav, metadata=METADATA)

    assert monthly["month"].tolist() == ["2024-01", "2024-02"]
    assert monthly["monthly_return"].tolist() == pytest.approx([0.045, 0.0094])
    assert monthly["is_best_month"].tolist() == [True, False]
    assert monthly["total_return_excluding_best_month"].iloc[0] == pytest.approx(0.0094)
    assert len(drawdowns) == 1
    assert drawdowns.loc[0, "start_date"] == pd.Timestamp("2024-01-31")
    assert drawdowns.loc[0, "trough_date"] == pd.Timestamp("2024-02-01")
    assert drawdowns.loc[0, "recovery_date"] == pd.Timestamp("2024-02-02")
    assert bool(drawdowns.loc[0, "recovered"]) is True


def test_daily_nav_frame_attaches_checkpoint_metadata() -> None:
    nav = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "strategy": ["ppo", "ppo"],
            "nav": [1.01, 1.02],
            "daily_return": [0.01, 1.02 / 1.01 - 1.0],
            "drawdown": [0.0, 0.0],
        }
    )

    result = _daily_nav_frame(nav, metadata=METADATA)

    assert result["selection_checkpoint"].tolist() == ["best_checkpoint"] * 2
    assert result["regime_name"].tolist() == ["validation_2024"] * 2
    assert result["date"].is_monotonic_increasing
    assert result["nav"].tolist() == pytest.approx([1.01, 1.02])


def test_daily_nav_frame_rejects_duplicate_dates() -> None:
    nav = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-02"]),
            "strategy": ["ppo", "ppo"],
            "nav": [1.0, 1.01],
            "daily_return": [0.0, 0.01],
            "drawdown": [0.0, 0.0],
        }
    )

    with pytest.raises(ValueError, match="duplicate dates"):
        _daily_nav_frame(nav, metadata=METADATA)


def test_summary_flags_collapse_static_and_lucky_month() -> None:
    concentration = pd.DataFrame(
        {
            **_repeated_metadata(4),
            "date": pd.to_datetime(
                ["2024-01-02", "2024-01-09", "2024-01-16", "2024-01-23"]
            ),
            "high_volatility": [False, False, False, True],
            "max_weight": [0.9, 0.9, 0.9, 0.9],
            "max_ticker": ["SPY"] * 4,
            "hhi": [0.82] * 4,
            "effective_asset_count": [1 / 0.82] * 4,
            "active_asset_count": [2] * 4,
            "shy_weight": [0.1] * 4,
            "equity_like_weight": [0.9] * 4,
        }
    )
    turnover = pd.DataFrame(
        {
            **_repeated_metadata(4),
            "date": concentration["date"],
            "turnover": [0.0, 0.0, 0.0, 0.0],
            "target_change_turnover": [0.0, 0.0, 0.0, 0.0],
        }
    )
    monthly = pd.DataFrame(
        {
            **_repeated_metadata(2),
            "month": ["2024-01", "2024-02"],
            "monthly_return": [0.2, -0.05],
            "is_best_month": [True, False],
            "best_month_positive_return_share": [1.0, 1.0],
            "total_return_excluding_best_month": [-0.05, -0.05],
        }
    )

    summary = build_policy_behavior_summary(
        selected={
            "configuration_id": "selected",
            "experiment_name": "seed_sweep",
        },
        selected_runs=[
            {
                "run_id": "run_7",
                "seed": 7,
                "selection_checkpoint": "best_checkpoint",
            }
        ],
        selected_models=[],
        allocations=pd.DataFrame(
            {
                **_repeated_metadata(8),
                "date": list(concentration["date"].repeat(2)),
                "ticker": ["SPY", "SHY"] * 4,
                "target_weight": [0.9, 0.1] * 4,
                "pre_trade_weight": [0.9, 0.1] * 4,
                "trade_weight": [0.0, 0.0] * 4,
            }
        ),
        concentration=concentration,
        turnover=turnover,
        monthly_returns=monthly,
        config=_behavior_config(),
        max_median_turnover=0.5,
        reconciliation=[{"all_metrics_match": True}],
        regime_windows=[],
        skipped_regimes=[],
        sources={},
    )

    warnings = summary["groups"][0]["warnings"]
    assert warnings["spy_only_collapse"] is True
    assert warnings["shy_only_collapse"] is False
    assert warnings["excessive_concentration"] is True
    assert warnings["static_target_policy"] is True
    assert warnings["lucky_month_dependence"] is True
    assert summary["campaign_warnings"]["spy_only_collapse"]["triggered"] is True
    assert "# Policy Behavior Diagnostics" in format_policy_behavior_report(summary)


def test_validation_reconciliation_rejects_wrong_checkpoint_metrics() -> None:
    registry_row = {
        "seed": 7,
        "selection_validation_total_return": 0.1,
        "selection_validation_sharpe_ratio": 1.0,
        "selection_validation_max_drawdown": -0.05,
        "selection_validation_average_weekly_turnover": 0.2,
        "selection_validation_transaction_cost_drag": 0.01,
    }
    computed = {
        "total_return": 0.2,
        "sharpe_ratio": 1.0,
        "max_drawdown": -0.05,
        "average_weekly_turnover": 0.2,
        "transaction_cost_drag": 0.01,
    }

    with pytest.raises(ValueError, match="do not reconcile"):
        _reconcile_validation_metrics(
            computed,
            registry_row=registry_row,
            run_id="run_7",
        )


def test_policy_behavior_cli_prints_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    report = tmp_path / "diagnostics_report.md"
    fake = PolicyBehaviorResult(
        outputs={"report": report},
        allocation_by_regime=pd.DataFrame(),
        nav_by_regime=pd.DataFrame(),
        concentration_metrics=pd.DataFrame(),
        turnover_distribution=pd.DataFrame(),
        monthly_returns=pd.DataFrame(),
        drawdown_periods=pd.DataFrame(),
        summary={},
    )
    monkeypatch.setattr(
        diagnostics_script,
        "run_policy_behavior_diagnostics",
        lambda **_kwargs: fake,
    )

    diagnostics_script.main(
        [
            "--selected-configuration",
            "selected.json",
            "--registry",
            "registry.csv",
            "--output-dir",
            "diagnostics",
        ]
    )

    assert capsys.readouterr().out == f"report: {report}\n"


def _behavior_config() -> PolicyBehaviorConfig:
    return PolicyBehaviorConfig(
        dominance_weight_threshold=0.8,
        dominance_date_fraction=0.8,
        active_weight_threshold=0.05,
        concentration_hhi_threshold=0.25,
        turnover_spike_quantile=0.95,
        high_volatility_quantile=0.75,
        static_target_change_threshold=0.01,
        lucky_month_positive_return_share=0.5,
        sensitivity_low_quantile=0.25,
        sensitivity_high_quantile=0.75,
        sensitivity_material_weight_shift=0.01,
    )


def _repeated_metadata(size: int) -> dict[str, list[object]]:
    return {
        column: [value] * size
        for column, value in METADATA.items()
        if column != "selection_checkpoint"
    }


def _backtest(*, costs: pd.DataFrame) -> BacktestResult:
    empty = pd.DataFrame()
    return BacktestResult(
        nav=empty,
        weights_target=empty,
        weights_drifted=empty,
        trades=empty,
        costs=costs,
        metrics={},
    )

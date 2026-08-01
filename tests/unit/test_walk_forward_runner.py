from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from portfolio_rl.evaluation import walk_forward_report
from portfolio_rl.evaluation.backtest import BacktestResult
from portfolio_rl.training.walk_forward_runner import (
    SelectionResult,
    load_walk_forward_campaign_config,
)


def test_canonical_walk_forward_campaign_is_frozen() -> None:
    config = load_walk_forward_campaign_config(
        "configs/experiments/ppo_walk_forward.yaml"
    )

    assert config.folds == ("WF1", "WF2", "WF3", "WF4")
    assert config.seeds == (7, 42, 101, 202, 999)
    assert config.total_timesteps == 500_000
    assert config.eval_freq_timesteps == 25_000
    assert (config.pilot_fold, config.pilot_seed, config.pilot_timesteps) == (
        "WF1",
        42,
        50_000,
    )
    assert config.alphas == (0.25, 0.5, 0.75, 1.0)


def test_outer_loader_runs_only_after_selection_freeze_verification(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = load_walk_forward_campaign_config(
        "configs/experiments/ppo_walk_forward.yaml"
    )
    result = SelectionResult(
        fold_id="WF1",
        seed=42,
        run_id="pilot",
        output_dir=tmp_path,
        freeze_path=tmp_path / "selection_freeze.json",
        selected_model_path=tmp_path / "selected_model.zip",
    )
    calls = []

    def verify(output_dir, *, config):
        del output_dir, config
        calls.append("verify_freeze")
        return result

    def load_outer(fold_dir):
        del fold_dir
        calls.append("load_outer")
        raise RuntimeError("stop after ordering assertion")

    monkeypatch.setattr(walk_forward_report, "verify_selection_freeze", verify)
    monkeypatch.setattr(
        walk_forward_report,
        "load_outer_evaluation_dataset",
        load_outer,
    )

    with pytest.raises(RuntimeError, match="ordering assertion"):
        walk_forward_report.evaluate_frozen_selection(
            config=config,
            fold_id="WF1",
            selection_results=[result],
            output_dir=tmp_path / "outer",
            pilot=True,
        )

    assert calls == ["verify_freeze", "load_outer"]


def test_same_path_gross_return_removes_multiplicative_costs() -> None:
    empty = pd.DataFrame()
    result = BacktestResult(
        nav=empty,
        weights_target=empty,
        weights_drifted=empty,
        trades=empty,
        costs=pd.DataFrame(
            {"transaction_cost_fraction": [0.01, 0.01]}
        ),
        metrics={"total_return": 0.08},
    )

    gross = walk_forward_report._same_path_gross_total_return(result)

    assert gross == pytest.approx(1.08 / (0.99 * 0.99) - 1.0)


def test_alpha_025_comparison_is_fold_level_and_benchmark_specific() -> None:
    rows = []
    strategies = (
        "ensemble_alpha_0.25",
        "equal_weight_weekly",
        "inverse_volatility",
        "momentum_63d_top3_equal_weight",
        "buy_and_hold_equal_weight",
    )
    for fold_index, fold_id in enumerate(("WF1", "WF2", "WF3", "WF4")):
        for strategy_index, strategy in enumerate(strategies):
            value = 0.01 * (fold_index - strategy_index)
            rows.append(
                {
                    "fold_id": fold_id,
                    "strategy": strategy,
                    "total_return": value,
                    "gross_total_return": value + 0.001,
                    "sharpe_ratio": value,
                    "max_drawdown": -0.1 + value,
                    "transaction_cost_drag": 0.001,
                }
            )

    by_fold, summary = walk_forward_report._alpha_025_comparisons(
        pd.DataFrame(rows)
    )

    assert len(by_fold) == 16
    assert len(summary) == 4
    assert set(by_fold["fold_id"]) == {"WF1", "WF2", "WF3", "WF4"}
    assert set(summary["benchmark"]) == set(strategies[1:])

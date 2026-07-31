from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from portfolio_rl.data.dataset import PortfolioDataset
from portfolio_rl.data.feature_store import PortfolioFeatureStore
from portfolio_rl.evaluation.turnover_overlay import (
    evaluate_turnover_overlay_study,
    write_turnover_overlay_artifacts,
)
from portfolio_rl.policies.baseline_policies import EqualWeightWeeklyPolicy


class FixedPolicy:
    def __init__(self, weights: list[float]) -> None:
        self._weights = np.asarray(weights, dtype=np.float64)

    def target_weights(
        self,
        observation: np.ndarray,
        info: Mapping[str, Any],
    ) -> np.ndarray:
        del observation, info
        return self._weights.copy()


def test_turnover_overlay_executes_frontier_and_writes_audit(
    tmp_path: Path,
) -> None:
    result = evaluate_turnover_overlay_study(
        feature_store=_store(),
        candidate_policy_factories={
            "five_seed_mean_weight_ensemble": lambda: FixedPolicy([0.9, 0.1]),
            "seed_42": lambda: FixedPolicy([0.8, 0.2]),
        },
        alphas=[0.25, 0.5, 0.75, 1.0],
        hurdle_policy_factory=lambda: EqualWeightWeeklyPolicy(2),
        configured_test_start_date="2025-01-01",
    )

    assert len(result.overlay_results) == 8
    assert set(result.overlay_results["alpha"]) == {0.25, 0.5, 0.75, 1.0}
    ensemble = result.overlay_results[
        result.overlay_results["candidate"]
        == "five_seed_mean_weight_ensemble"
    ].set_index("alpha")
    assert ensemble.loc[1.0, "return_advantage_retention"] == pytest.approx(1.0)
    assert ensemble.loc[1.0, "turnover_reduction_vs_alpha_1"] == pytest.approx(
        0.0
    )
    assert ensemble.loc[0.25, "average_weekly_turnover"] < ensemble.loc[
        1.0, "average_weekly_turnover"
    ]

    audit = result.target_audit[
        result.target_audit["candidate"] == "five_seed_mean_weight_ensemble"
    ]
    for alpha, frame in audit.groupby("alpha"):
        assert np.isfinite(frame["executed_target"]).all()
        assert frame["executed_target"].ge(0.0).all()
        executed_sums = frame.groupby(["date", "decision_step"])[
            "executed_target"
        ].sum()
        np.testing.assert_allclose(executed_sums, 1.0, atol=1e-7, rtol=0.0)
        np.testing.assert_allclose(
            frame["executed_target"],
            frame["current_weight"]
            + alpha * (frame["raw_policy_target"] - frame["current_weight"]),
            atol=1e-12,
            rtol=0.0,
        )
        calculated_turnover = (
            0.5
            * frame.assign(
                absolute_trade=(
                    frame["executed_target"] - frame["current_weight"]
                ).abs()
            )
            .groupby(["date", "decision_step"])["absolute_trade"]
            .sum()
        )
        recorded_turnover = (
            frame.groupby(["date", "decision_step"])[
                "executed_half_l1_turnover"
            ].first()
        )
        np.testing.assert_allclose(
            calculated_turnover,
            recorded_turnover,
            atol=1e-12,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            frame["executed_half_l1_turnover"],
            alpha * frame["raw_half_l1_turnover"],
            atol=1e-12,
            rtol=0.0,
        )
        backtest = result.backtests[
            ("five_seed_mean_weight_ensemble", float(alpha))
        ]
        trades = backtest.trades.rename(
            columns={
                "pre_trade_weight": "backtest_current_weight",
                "target_weight": "backtest_executed_target",
            }
        )
        reconciled = frame.merge(
            trades[
                [
                    "date",
                    "ticker",
                    "backtest_current_weight",
                    "backtest_executed_target",
                ]
            ],
            on=["date", "ticker"],
            validate="one_to_one",
        )
        np.testing.assert_allclose(
            reconciled["current_weight"],
            reconciled["backtest_current_weight"],
            atol=1e-12,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            reconciled["executed_target"],
            reconciled["backtest_executed_target"],
            atol=1e-12,
            rtol=0.0,
        )

    alpha_one = audit[np.isclose(audit["alpha"], 1.0)]
    np.testing.assert_allclose(
        alpha_one["raw_policy_target"],
        alpha_one["executed_target"],
        atol=1e-12,
        rtol=0.0,
    )
    alpha_quarter = audit[np.isclose(audit["alpha"], 0.25)]
    assert not np.allclose(
        alpha_quarter["raw_policy_target"],
        alpha_quarter["executed_target"],
    )

    output = tmp_path / "turnover_overlay"
    outputs = write_turnover_overlay_artifacts(
        result=result,
        output_dir=output,
        manifest={
            "pr13_manifest_sha256": "abc123",
            "test_accessed": False,
            "alpha_selected": None,
        },
    )
    assert all(path.exists() for path in outputs.values())
    report = (output / "turnover_overlay_report.md").read_text(encoding="utf-8")
    assert "2024 is consumed development/selection data" in report
    assert "No alpha is selected in PR 14" in report
    assert "live drifted weights" in report
    assert "recomputed from each overlay path" in report
    assert (output / "backtest/equal_weight_weekly/metrics.json").exists()
    assert (
        output
        / "backtest/five_seed_mean_weight_ensemble/alpha_0_25/metrics.json"
    ).exists()

    with pytest.raises(FileExistsError):
        write_turnover_overlay_artifacts(
            result=result,
            output_dir=output,
            manifest={"pr13_manifest_sha256": "abc123"},
        )


@pytest.mark.parametrize(
    ("split", "start_date"),
    [("test", "2024-01-02"), ("validation", "2025-01-02")],
)
def test_turnover_overlay_rejects_final_test_access(
    split: str,
    start_date: str,
) -> None:
    with pytest.raises(ValueError, match="must not access"):
        evaluate_turnover_overlay_study(
            feature_store=_store(split=split, start_date=start_date),
            candidate_policy_factories={
                "five_seed_mean_weight_ensemble": lambda: FixedPolicy([0.9, 0.1])
            },
            alphas=[0.25, 0.5, 0.75, 1.0],
            hurdle_policy_factory=lambda: EqualWeightWeeklyPolicy(2),
            configured_test_start_date="2025-01-01",
        )


def _store(
    *,
    split: str = "validation",
    start_date: str = "2024-01-02",
) -> PortfolioFeatureStore:
    dates = pd.date_range(start_date, periods=70, freq="B")
    returns = np.column_stack(
        [
            np.full(70, 0.002),
            np.full(70, -0.0002),
        ]
    ).astype(np.float32)
    dataset = PortfolioDataset(
        dates=dates,
        splits=np.array([split] * 70),
        market_features=np.zeros((70, 2), dtype=np.float32),
        returns=returns,
        asset_order=["SPY", "SHY"],
        feature_version="v1",
        observation_dim=4,
    )
    return PortfolioFeatureStore(dataset, split)

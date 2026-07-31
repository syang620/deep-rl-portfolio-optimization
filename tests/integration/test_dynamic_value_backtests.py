from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from portfolio_rl.data.dataset import PortfolioDataset
from portfolio_rl.data.feature_store import PortfolioFeatureStore
from portfolio_rl.evaluation.dynamic_value import (
    evaluate_dynamic_value_checks,
    write_dynamic_value_artifacts,
)
from portfolio_rl.policies.baseline_policies import EqualWeightWeeklyPolicy


class CurrentWeightResponsivePolicy:
    def target_weights(
        self,
        observation: np.ndarray,
        info: Mapping[str, Any],
    ) -> np.ndarray:
        current = np.asarray(info["current_weights"], dtype=np.float64)
        np.testing.assert_allclose(observation[-2:], current, atol=1e-7)
        spy_weight = 0.7 + 0.1 * (current[0] - 0.5)
        return np.array([spy_weight, 1.0 - spy_weight])


def test_dynamic_value_checks_execute_live_and_control_paths(
    tmp_path: Path,
) -> None:
    prior, evaluation = _stores()
    result = evaluate_dynamic_value_checks(
        prior_feature_store=prior,
        evaluation_feature_store=evaluation,
        ensemble_policy_factory=CurrentWeightResponsivePolicy,
        hurdle_policy_factory=lambda: EqualWeightWeeklyPolicy(2),
        alphas=[0.25, 0.5, 0.75, 1.0],
        circular_shifts=[4, 13, 26],
        configured_test_start_date="2025-01-01",
    )

    assert len(result.diagnostic_results) == 35
    assert set(result.diagnostic_results["diagnostic"]) == {
        "dynamic_live",
        "ex_ante_static_2023",
        "oracle_static_2024",
        "lag_1_decision",
        "circular_shift_4",
        "circular_shift_13",
        "circular_shift_26",
    }
    assert len(result.active_return_decomposition) == 30
    direct = result.backtests[("ensemble_unmodified", "dynamic_live")]
    alpha_one = result.backtests[("ensemble_alpha_1.00", "dynamic_live")]
    pd.testing.assert_frame_equal(
        direct.nav.drop(columns="strategy"),
        alpha_one.nav.drop(columns="strategy"),
    )

    audit = result.target_sequences
    lagged = audit[
        (audit["candidate"] == "ensemble_alpha_0.50")
        & (audit["diagnostic"] == "lag_1_decision")
    ]
    first_lag = lagged[lagged["decision_step"] == 0].sort_values("ticker")
    np.testing.assert_allclose(first_lag["target_weight"], [0.5, 0.5])
    dynamic = audit[
        (audit["candidate"] == "ensemble_alpha_0.50")
        & (audit["diagnostic"] == "dynamic_live")
    ]
    assert dynamic["dynamic_raw_target"].ne(
        dynamic["dynamic_executed_target"]
    ).any()
    assert (
        audit[audit["diagnostic"].str.startswith("circular_shift_")][
            "deployability"
        ]
        == "non_deployable_sequence_attribution"
    ).all()

    output = tmp_path / "dynamic_value"
    outputs = write_dynamic_value_artifacts(
        result=result,
        output_dir=output,
        manifest={
            "pr14_manifest_sha256": "abc123",
            "test_accessed": False,
            "alpha_selected": None,
        },
    )
    assert all(path.exists() for path in outputs.values())
    report = (output / "dynamic_value_report.md").read_text(encoding="utf-8")
    assert "2024 is consumed development/selection data" in report
    assert "No alpha is selected or eliminated" in report
    assert "own live drifted portfolio" in report
    assert "Oracle static and circular-shift controls are non-deployable" in report
    with pytest.raises(FileExistsError):
        write_dynamic_value_artifacts(
            result=result,
            output_dir=output,
            manifest={"pr14_manifest_sha256": "abc123"},
        )


@pytest.mark.parametrize(
    ("prior_split", "evaluation_split", "evaluation_start"),
    [
        ("train", "test", "2024-01-02"),
        ("train", "validation", "2025-01-02"),
    ],
)
def test_dynamic_value_checks_reject_final_test_access(
    prior_split: str,
    evaluation_split: str,
    evaluation_start: str,
) -> None:
    prior, evaluation = _stores(
        prior_split=prior_split,
        evaluation_split=evaluation_split,
        evaluation_start=evaluation_start,
    )

    with pytest.raises(ValueError, match="must not access"):
        evaluate_dynamic_value_checks(
            prior_feature_store=prior,
            evaluation_feature_store=evaluation,
            ensemble_policy_factory=CurrentWeightResponsivePolicy,
            hurdle_policy_factory=lambda: EqualWeightWeeklyPolicy(2),
            alphas=[0.25, 0.5, 0.75, 1.0],
            circular_shifts=[4, 13, 26],
            configured_test_start_date="2025-01-01",
        )


def _stores(
    *,
    prior_split: str = "train",
    evaluation_split: str = "validation",
    evaluation_start: str = "2024-01-02",
) -> tuple[PortfolioFeatureStore, PortfolioFeatureStore]:
    prior_dates = pd.date_range("2023-01-02", periods=140, freq="B")
    evaluation_dates = pd.date_range(evaluation_start, periods=140, freq="B")
    dates = prior_dates.append(evaluation_dates)
    returns = np.column_stack(
        [
            0.001 + 0.0005 * np.sin(np.arange(280)),
            0.0002 + 0.0003 * np.cos(np.arange(280)),
        ]
    ).astype(np.float32)
    dataset = PortfolioDataset(
        dates=dates,
        splits=np.array([prior_split] * 140 + [evaluation_split] * 140),
        market_features=np.zeros((280, 2), dtype=np.float32),
        returns=returns,
        asset_order=["SPY", "SHY"],
        feature_version="v1",
        observation_dim=4,
    )
    return (
        PortfolioFeatureStore(dataset, prior_split),
        PortfolioFeatureStore(dataset, evaluation_split),
    )

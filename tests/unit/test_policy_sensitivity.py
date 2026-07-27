from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from portfolio_rl.data.dataset import PortfolioDataset
from portfolio_rl.evaluation.policy_sensitivity import (
    PolicySensitivityResult,
    aggregate_sensitivity_results,
    build_counterfactual_observations,
    global_feature_indices,
    interpret_sensitivity_results,
    train_feature_quantiles,
)
from portfolio_rl.features.feature_spec import FeatureSpec
from scripts import run_policy_sensitivity as sensitivity_script


def test_global_feature_indices_follow_flattened_feature_contract() -> None:
    spec = _feature_spec()

    indices = global_feature_indices(
        spec,
        ["vix_z_21d", "spy_vol_21d", "spy_drawdown_63d"],
    )

    assert indices == {
        "vix_z_21d": 4,
        "spy_vol_21d": 7,
        "spy_drawdown_63d": 8,
    }


def test_train_feature_quantiles_exclude_test_rows() -> None:
    dataset = _dataset()

    quantiles = train_feature_quantiles(
        dataset,
        feature_indices={"spy_vol_21d": 7},
        low_quantile=0.25,
        high_quantile=0.75,
    )

    assert quantiles["spy_vol_21d"]["low"] == pytest.approx(1.5)
    assert quantiles["spy_vol_21d"]["high"] == pytest.approx(2.5)


def test_counterfactual_observations_change_only_requested_features() -> None:
    observation = np.arange(11, dtype=np.float32)
    indices = {
        "vix_z_21d": 4,
        "vix_z_63d": 5,
        "credit_spread_z_63d": 6,
        "spy_vol_21d": 7,
        "spy_drawdown_63d": 8,
    }
    quantiles = {feature: {"low": -1.0, "high": 1.0} for feature in indices}

    scenarios = build_counterfactual_observations(
        observation,
        feature_indices=indices,
        quantiles=quantiles,
    )

    spy_high = scenarios[("spy_volatility", "high_risk")]["observation"]
    changed = np.flatnonzero(spy_high != observation)
    assert changed.tolist() == [7]
    assert spy_high[7] == 1.0

    global_high = scenarios[("global_risk", "high_risk")]["observation"]
    assert global_high[4:8].tolist() == [1.0, 1.0, 1.0, 1.0]
    assert global_high[8] == -1.0
    np.testing.assert_array_equal(global_high[9:], observation[9:])


def test_sensitivity_aggregation_and_interpretation_flag_pro_risk_response() -> None:
    allocations = _sensitivity_allocations()

    summary = aggregate_sensitivity_results(
        allocations,
        material_weight_shift=0.01,
    )
    interpretation = interpret_sensitivity_results(summary)

    assert len(summary) == 2
    assert summary["median_equity_like_weight_delta"].tolist() == pytest.approx(
        [0.2, 0.2]
    )
    assert summary["median_shy_weight_delta"].tolist() == pytest.approx([-0.2, -0.2])
    assert summary["median_one_way_allocation_distance"].tolist() == pytest.approx(
        [0.2, 0.2]
    )
    assert summary["material_pro_risk_response"].all()
    assert interpretation["stop_before_packaging"] is True
    assert interpretation["stop_regimes"] == ["validation_2024"]


def test_policy_sensitivity_cli_prints_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    report = tmp_path / "sensitivity_report.md"
    fake = PolicySensitivityResult(
        outputs={"report": report},
        allocations=pd.DataFrame(),
        summary=pd.DataFrame(),
        manifest={},
    )
    monkeypatch.setattr(
        sensitivity_script,
        "run_policy_sensitivity",
        lambda **_kwargs: fake,
    )

    sensitivity_script.main(
        [
            "--selected-configuration",
            "selected.json",
            "--registry",
            "registry.csv",
            "--diagnostics-dir",
            "diagnostics",
            "--output-dir",
            "sensitivity",
        ]
    )

    assert capsys.readouterr().out == f"report: {report}\n"


def _feature_spec() -> FeatureSpec:
    return FeatureSpec(
        feature_version="v1",
        asset_order=["SPY", "SHY"],
        per_asset_features=["ret_21d", "vol_21d"],
        global_features=[
            "vix_z_21d",
            "vix_z_63d",
            "credit_spread_z_63d",
            "spy_vol_21d",
            "spy_drawdown_63d",
        ],
        current_weight_features=["weight_spy", "weight_shy"],
        observation_dim=11,
        created_at="2026-01-01T00:00:00+00:00",
    )


def _dataset() -> PortfolioDataset:
    market_features = np.zeros((3, 9), dtype=np.float32)
    market_features[:, 7] = [1.0, 3.0, 100.0]
    return PortfolioDataset(
        dates=pd.DatetimeIndex(["2024-01-01", "2024-01-02", "2025-01-02"]),
        splits=np.array(["train", "train", "test"]),
        market_features=market_features,
        returns=np.zeros((3, 2), dtype=np.float32),
        asset_order=["SPY", "SHY"],
        feature_version="v1",
        observation_dim=11,
    )


def _sensitivity_allocations() -> pd.DataFrame:
    rows = []
    for probe in ["global_risk", "spy_volatility"]:
        for date in pd.to_datetime(["2024-01-02", "2024-01-09"]):
            for scenario, weights in {
                "low_risk": {"SPY": 0.4, "SHY": 0.6},
                "high_risk": {"SPY": 0.6, "SHY": 0.4},
            }.items():
                for ticker, weight in weights.items():
                    rows.append(
                        {
                            "configuration_id": "selected",
                            "experiment_name": "seed_sweep",
                            "run_id": "run_7",
                            "seed": 7,
                            "selection_checkpoint": "best_checkpoint",
                            "regime_name": "validation_2024",
                            "split": "validation",
                            "in_sample": False,
                            "date": date,
                            "probe": probe,
                            "scenario": scenario,
                            "ticker": ticker,
                            "target_weight": weight,
                            "observed_target_weight": 0.5,
                            "equity_like": ticker == "SPY",
                        }
                    )
    return pd.DataFrame(rows)

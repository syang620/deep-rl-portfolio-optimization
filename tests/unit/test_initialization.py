from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from portfolio_rl.data.dataset import PortfolioDataset
from portfolio_rl.data.feature_store import PortfolioFeatureStore
from portfolio_rl.evaluation.backtest import run_weight_policy_backtest
from portfolio_rl.evaluation.initialization import (
    EqualWeightInitializer,
    InverseVolatilityInitializer,
    SHYInitializer,
    StaticWeightInitializer,
)
from portfolio_rl.evaluation.initialization_sensitivity import (
    evaluate_initialization_sensitivity,
    format_initialization_sensitivity_report,
    write_initialization_sensitivity_artifacts,
)


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


def test_default_and_explicit_equal_weight_initialization_are_identical() -> None:
    store = _store()
    default = run_weight_policy_backtest(
        feature_store=store,
        policy=FixedPolicy([0.7, 0.3]),
        strategy="seed_42",
        max_steps=2,
    )
    explicit = run_weight_policy_backtest(
        feature_store=store,
        policy=FixedPolicy([0.7, 0.3]),
        strategy="seed_42",
        max_steps=2,
        initial_portfolio_provider=EqualWeightInitializer(),
    )

    pd.testing.assert_frame_equal(default.nav, explicit.nav)
    pd.testing.assert_frame_equal(default.trades, explicit.trades)
    assert default.metrics == explicit.metrics


def test_initial_endowment_has_no_establishment_cost() -> None:
    result = run_weight_policy_backtest(
        feature_store=_store(),
        policy=FixedPolicy([0.0, 1.0]),
        strategy="seed_42",
        max_steps=1,
        initial_portfolio_provider=SHYInitializer(),
    )

    assert result.costs["turnover"].iloc[0] == pytest.approx(0.0)
    assert result.costs["transaction_cost_fraction"].iloc[0] == pytest.approx(0.0)


def test_first_rebalance_cost_uses_initializer_as_pre_trade_weights() -> None:
    result = run_weight_policy_backtest(
        feature_store=_store(),
        policy=FixedPolicy([1.0, 0.0]),
        strategy="seed_42",
        max_steps=1,
        transaction_cost_bps=10.0,
        initial_portfolio_provider=SHYInitializer(),
    )

    np.testing.assert_array_equal(
        result.trades["pre_trade_weight"].to_numpy(),
        [0.0, 1.0],
    )
    assert result.costs["turnover"].iloc[0] == pytest.approx(1.0)
    assert result.costs["transaction_cost_fraction"].iloc[0] == pytest.approx(
        0.001
    )


def test_inverse_volatility_uses_exactly_63_strictly_pre_window_rows() -> None:
    store = _store()
    expected_returns = store.get_pre_window_log_returns(63)
    expected_volatility = np.std(expected_returns, axis=0, ddof=0)
    expected = (1.0 / expected_volatility) / (1.0 / expected_volatility).sum()

    actual = InverseVolatilityInitializer(lookback=63).initial_weights(store)

    np.testing.assert_allclose(actual, expected)


def test_static_initializer_flattens_ticker_mapping_in_asset_order() -> None:
    initializer = StaticWeightInitializer(weights={"SHY": 0.25, "SPY": 0.75})

    weights = initializer.initial_weights(_store())

    np.testing.assert_array_equal(weights, [0.75, 0.25])


@pytest.mark.parametrize(
    "weights",
    [
        {"SPY": 1.0},
        {"SPY": 0.5, "SHY": 0.4, "QQQ": 0.1},
        {"SPY": np.nan, "SHY": np.nan},
        {"SPY": 1.1, "SHY": -0.1},
        {"SPY": 0.6, "SHY": 0.3},
    ],
)
def test_static_initializer_rejects_invalid_ticker_weight_contract(
    weights: dict[str, float],
) -> None:
    with pytest.raises(ValueError):
        StaticWeightInitializer(weights=weights).initial_weights(_store())


def test_shy_initializer_requires_shy_in_asset_order() -> None:
    store = PortfolioFeatureStore(_dataset(asset_order=["SPY", "QQQ"]), "validation")

    with pytest.raises(ValueError, match="requires SHY"):
        SHYInitializer().initial_weights(store)


def test_evaluator_uses_fresh_policy_for_every_initializer() -> None:
    instances = []

    def factory() -> FixedPolicy:
        policy = FixedPolicy([0.6, 0.4])
        instances.append(policy)
        return policy

    evaluate_initialization_sensitivity(
        feature_store=_store(),
        candidate_policy_factories={"seed_42": factory},
        initializers={
            "equal_weight": EqualWeightInitializer(),
            "shy": SHYInitializer(),
        },
        configured_test_start_date="2025-01-01",
    )

    assert len(instances) == 2
    assert instances[0] is not instances[1]


def test_convergence_stores_full_and_half_l1_for_targets_and_pre_trade() -> None:
    result = evaluate_initialization_sensitivity(
        feature_store=_store(),
        candidate_policy_factories={
            "seed_42": lambda: FixedPolicy([0.6, 0.4])
        },
        initializers={
            "equal_weight": EqualWeightInitializer(),
            "shy": SHYInitializer(),
        },
        configured_test_start_date="2025-01-01",
        convergence_threshold=0.05,
        convergence_consecutive_decisions=4,
    )

    detail = result.convergence_metrics
    assert {
        "target_full_l1",
        "target_half_l1",
        "pre_trade_full_l1",
        "pre_trade_half_l1",
    }.issubset(detail.columns)
    np.testing.assert_allclose(
        detail["target_half_l1"],
        0.5 * detail["target_full_l1"],
    )
    np.testing.assert_allclose(
        detail["pre_trade_half_l1"],
        0.5 * detail["pre_trade_full_l1"],
    )
    summary = result.convergence_summary.iloc[0]
    assert bool(summary["converged"])
    assert summary["elapsed_decisions"] == 4


def test_early_windows_use_20_and_60_days_and_4_and_12_cost_rows() -> None:
    result = evaluate_initialization_sensitivity(
        feature_store=_store(),
        candidate_policy_factories={
            "seed_42": lambda: FixedPolicy([0.6, 0.4])
        },
        initializers={"equal_weight": EqualWeightInitializer()},
        configured_test_start_date="2025-01-01",
    )
    row = result.results.iloc[0]
    backtest = result.backtests[("seed_42", "equal_weight")]

    assert row["first_4_weeks_transaction_cost_drag"] == pytest.approx(
        backtest.costs["transaction_cost_fraction"].iloc[:4].sum()
    )
    assert row["first_12_weeks_transaction_cost_drag"] == pytest.approx(
        backtest.costs["transaction_cost_fraction"].iloc[:12].sum()
    )
    first_20_return = np.prod(1.0 + backtest.nav["daily_return"].iloc[:20]) - 1.0
    first_60_return = np.prod(1.0 + backtest.nav["daily_return"].iloc[:60]) - 1.0
    assert row["first_4_weeks_total_return"] == pytest.approx(first_20_return)
    assert row["first_12_weeks_total_return"] == pytest.approx(first_60_return)


def test_early_window_sharpe_is_none_for_zero_volatility() -> None:
    result = evaluate_initialization_sensitivity(
        feature_store=_store(),
        candidate_policy_factories={
            "seed_42": lambda: FixedPolicy([0.5, 0.5])
        },
        initializers={"equal_weight": EqualWeightInitializer()},
        configured_test_start_date="2025-01-01",
    )

    assert result.results.iloc[0]["first_4_weeks_sharpe_ratio"] is None


def test_short_early_window_metrics_are_none() -> None:
    store = PortfolioFeatureStore(
        _dataset(),
        split="validation",
        end_date="2024-01-16",
    )
    result = evaluate_initialization_sensitivity(
        feature_store=store,
        candidate_policy_factories={
            "seed_42": lambda: FixedPolicy([0.5, 0.5])
        },
        initializers={"equal_weight": EqualWeightInitializer()},
        configured_test_start_date="2025-01-01",
    )

    assert result.results.iloc[0]["first_4_weeks_sharpe_ratio"] is None
    assert result.results.iloc[0]["first_12_weeks_total_return"] is None


def test_writer_fails_closed_on_output_overwrite(tmp_path: Path) -> None:
    result = evaluate_initialization_sensitivity(
        feature_store=_store(),
        candidate_policy_factories={
            "seed_42": lambda: FixedPolicy([0.6, 0.4])
        },
        initializers={"equal_weight": EqualWeightInitializer()},
        configured_test_start_date="2025-01-01",
    )
    output = tmp_path / "pr12"
    manifest = _report_manifest()

    write_initialization_sensitivity_artifacts(
        result=result,
        output_dir=output,
        manifest=manifest,
    )

    with pytest.raises(FileExistsError):
        write_initialization_sensitivity_artifacts(
            result=result,
            output_dir=output,
            manifest=manifest,
        )


def test_report_scopes_conclusion_to_seed_42_and_defers_ensemble() -> None:
    result = evaluate_initialization_sensitivity(
        feature_store=_store(),
        candidate_policy_factories={
            "seed_42": lambda: FixedPolicy([0.6, 0.4])
        },
        initializers={
            "equal_weight": EqualWeightInitializer(),
            "shy": SHYInitializer(),
        },
        configured_test_start_date="2025-01-01",
    )

    report = format_initialization_sensitivity_report(
        result,
        manifest=_report_manifest(),
    )

    assert "frozen seed-42 policy" in report
    assert "does not establish that PPO generally" in report
    assert "Equal weight remains the official headline" in report
    assert "deferred until PR 13" in report
    assert "Performance diagnostics" in report
    assert "Terminal target half-L1" in report
    assert "2024 consumed development/selection data" in report
    assert f"Frozen seed-42 model SHA-256: `{'a' * 64}`" in report
    assert f"Research freeze-manifest SHA-256: `{'b' * 64}`" in report
    assert "artifacts/initialization_sensitivity/campaign" in report


def test_report_fails_closed_when_provenance_is_missing() -> None:
    result = evaluate_initialization_sensitivity(
        feature_store=_store(),
        candidate_policy_factories={
            "seed_42": lambda: FixedPolicy([0.6, 0.4])
        },
        initializers={"equal_weight": EqualWeightInitializer()},
        configured_test_start_date="2025-01-01",
    )

    with pytest.raises(ValueError, match="artifact_directory"):
        format_initialization_sensitivity_report(result, manifest={})


def test_date_guard_rejects_mislabeled_store_at_test_start() -> None:
    store = PortfolioFeatureStore(
        _dataset(validation_start="2025-01-01"),
        "validation",
    )

    with pytest.raises(ValueError, match="must not access"):
        evaluate_initialization_sensitivity(
            feature_store=store,
            candidate_policy_factories={
                "seed_42": lambda: FixedPolicy([0.5, 0.5])
            },
            initializers={"equal_weight": EqualWeightInitializer()},
            configured_test_start_date="2025-01-01",
        )


def test_split_guard_rejects_test_store() -> None:
    dataset = _dataset()
    dataset = PortfolioDataset(
        dates=dataset.dates,
        splits=np.array(["train"] * 70 + ["test"] * 70),
        market_features=dataset.market_features,
        returns=dataset.returns,
        asset_order=dataset.asset_order,
        feature_version=dataset.feature_version,
        observation_dim=dataset.observation_dim,
    )

    with pytest.raises(ValueError, match="must not access"):
        evaluate_initialization_sensitivity(
            feature_store=PortfolioFeatureStore(dataset, "test"),
            candidate_policy_factories={
                "seed_42": lambda: FixedPolicy([0.5, 0.5])
            },
            initializers={"equal_weight": EqualWeightInitializer()},
            configured_test_start_date="2025-01-01",
        )


def _store() -> PortfolioFeatureStore:
    return PortfolioFeatureStore(_dataset(), split="validation")


def _report_manifest() -> dict[str, Any]:
    return {
        "artifact_directory": "artifacts/initialization_sensitivity/campaign",
        "test_accessed": False,
        "source_hashes": {
            "model": {"sha256": "a" * 64},
            "freeze_manifest": {"sha256": "b" * 64},
        },
    }


def _dataset(
    *,
    asset_order: list[str] | None = None,
    validation_start: str = "2024-01-02",
) -> PortfolioDataset:
    assets = asset_order or ["SPY", "SHY"]
    n_assets = len(assets)
    train_dates = pd.date_range(end="2023-12-29", periods=70, freq="B")
    validation_dates = pd.date_range(validation_start, periods=70, freq="B")
    dates = train_dates.append(validation_dates)
    train_returns = np.column_stack(
        [
            0.001 * np.sin(np.arange(70)),
            0.003 * np.cos(np.arange(70)),
        ]
    )[:, :n_assets]
    validation_returns = np.full((70, n_assets), 0.0001)
    returns = np.vstack([train_returns, validation_returns]).astype(np.float32)
    return PortfolioDataset(
        dates=dates,
        splits=np.array(["train"] * 70 + ["validation"] * 70),
        market_features=np.zeros((140, 2), dtype=np.float32),
        returns=returns,
        asset_order=assets,
        feature_version="v1",
        observation_dim=2 + n_assets,
    )

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from portfolio_rl.data.dataset import PortfolioDataset
from portfolio_rl.data.feature_store import PortfolioFeatureStore
from portfolio_rl.evaluation.ensemble import (
    ENSEMBLE_STRATEGY,
    evaluate_ensemble_campaign,
    write_ensemble_artifacts,
)
from portfolio_rl.evaluation.initialization import (
    EqualWeightInitializer,
    SHYInitializer,
)
from portfolio_rl.evaluation.initialization_sensitivity import (
    evaluate_initialization_sensitivity,
)
from portfolio_rl.policies.baseline_policies import (
    BuyAndHoldEqualWeightPolicy,
    EqualWeightWeeklyPolicy,
    InverseVolatilityPolicy,
    SingleAssetPolicy,
)
from portfolio_rl.policies.ensemble_policy import MeanWeightEnsemblePolicy


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


def test_ensemble_campaign_executes_real_policy_and_writes_artifacts(
    tmp_path: Path,
) -> None:
    stores = _stores()
    member_factories = _member_factories()
    result = evaluate_ensemble_campaign(
        feature_stores=stores,
        member_policy_factories=member_factories,
        baseline_policy_factories=_baseline_factories(),
        representative_seed=42,
        configured_test_start_date="2025-01-01",
    )

    validation_targets = result.ensemble_targets[
        result.ensemble_targets["regime"] == "validation_2024"
    ]
    first_target = validation_targets[
        validation_targets["decision_step"] == 0
    ].sort_values("ticker")
    np.testing.assert_allclose(
        first_target["ensemble_target_weight"],
        [0.52, 0.48],  # SHY, SPY after alphabetical sorting
    )
    replayed = (
        result.member_targets.groupby(
            ["regime", "date", "decision_step", "ticker"],
            as_index=False,
        )["target_weight"]
        .mean()
        .rename(columns={"target_weight": "replayed_target_weight"})
    )
    reconciled = replayed.merge(
        result.ensemble_targets,
        on=["regime", "date", "decision_step", "ticker"],
        validate="one_to_one",
    )
    np.testing.assert_allclose(
        reconciled["replayed_target_weight"],
        reconciled["ensemble_target_weight"],
    )
    assert set(result.member_targets["seed"]) == {7, 42, 101, 202, 999}
    assert result.disagreement_metrics["member_count"].eq(5).all()
    assert "median_pairwise_target_half_l1" in result.disagreement_metrics
    first_live = result.member_targets[
        (result.member_targets["regime"] == "validation_2024")
        & (result.member_targets["decision_step"] == 0)
    ]
    assert first_live.groupby("ticker")["live_current_weight"].nunique().eq(1).all()
    np.testing.assert_allclose(
        first_live.drop_duplicates("ticker")["live_current_weight"],
        [0.5, 0.5],
    )
    assert set(result.seed_tracking["member"]) == {
        "seed_7",
        "seed_42",
        "seed_101",
        "seed_202",
        "seed_999",
    }
    assert (result.metrics["strategy"] == ENSEMBLE_STRATEGY).sum() == 2

    initialization = evaluate_initialization_sensitivity(
        feature_store=stores["validation_2024"],
        candidate_policy_factories={
            ENSEMBLE_STRATEGY: lambda: MeanWeightEnsemblePolicy(
                member_policies={
                    f"seed_{seed}": factory()
                    for seed, factory in member_factories.items()
                }
            )
        },
        initializers={
            "equal_weight": EqualWeightInitializer(),
            "shy_100pct": SHYInitializer(),
        },
        configured_test_start_date="2025-01-01",
    )
    output = tmp_path / "ensemble"
    outputs = write_ensemble_artifacts(
        result=result,
        initialization_result=initialization,
        output_dir=output,
        manifest={
            "member_seed_order": [7, 42, 101, 202, 999],
            "test_accessed": False,
        },
    )

    assert all(path.exists() for path in outputs.values())
    assert (output / "backtest/validation_2024/seed_42/metrics.json").exists()
    assert (
        output
        / f"backtest/validation_2024/{ENSEMBLE_STRATEGY}/metrics.json"
    ).exists()
    assert (
        output
        / "initialization_sensitivity/results_by_model_and_initializer.csv"
    ).exists()
    report = (output / "ensemble_report.md").read_text(encoding="utf-8")
    assert "averages executable PPO target weights" in report
    assert "2024 is consumed development/selection data" in report
    assert "2022 window is an in-sample" in report
    assert "Seed-to-ensemble tracking" in report
    assert "Maximum drawdown comparison" in report
    assert "seed-selection dispersion" in report
    assert "First-rebalance execution" in report
    assert "same ensemble portfolio live current weights" in report
    assert "median decision-level pairwise target half-L1" in report

    with pytest.raises(FileExistsError):
        write_ensemble_artifacts(
            result=result,
            initialization_result=initialization,
            output_dir=output,
            manifest={"member_seed_order": [7, 42, 101, 202, 999]},
        )


def test_ensemble_campaign_rejects_date_based_test_access() -> None:
    stores = _stores(validation_start="2025-01-02")

    with pytest.raises(ValueError, match="must not access"):
        evaluate_ensemble_campaign(
            feature_stores=stores,
            member_policy_factories=_member_factories(),
            baseline_policy_factories=_baseline_factories(),
            representative_seed=42,
            configured_test_start_date="2025-01-01",
        )


def test_ensemble_campaign_rejects_test_split_even_before_test_date() -> None:
    dataset = _dataset("2024-01-02")
    relabeled = PortfolioDataset(
        dates=dataset.dates,
        splits=np.array(["train"] * 70 + ["test"] * 70),
        market_features=dataset.market_features,
        returns=dataset.returns,
        asset_order=dataset.asset_order,
        feature_version=dataset.feature_version,
        observation_dim=dataset.observation_dim,
    )
    stores = {
        "validation_2024": PortfolioFeatureStore(relabeled, "test"),
        "historical_2022": PortfolioFeatureStore(relabeled, "train"),
    }

    with pytest.raises(ValueError, match="must not access"):
        evaluate_ensemble_campaign(
            feature_stores=stores,
            member_policy_factories=_member_factories(),
            baseline_policy_factories=_baseline_factories(),
            representative_seed=42,
            configured_test_start_date="2025-01-01",
        )


def _member_factories() -> dict[int, Any]:
    weights = {
        7: [0.8, 0.2],
        42: [0.6, 0.4],
        101: [0.5, 0.5],
        202: [0.3, 0.7],
        999: [0.2, 0.8],
    }
    return {
        seed: (lambda target=target: FixedPolicy(target))
        for seed, target in weights.items()
    }


def _baseline_factories() -> dict[str, Any]:
    assets = ["SPY", "SHY"]
    return {
        "equal_weight_weekly": lambda: EqualWeightWeeklyPolicy(2),
        "inverse_volatility": lambda: InverseVolatilityPolicy(2),
        "buy_and_hold_equal_weight": lambda: BuyAndHoldEqualWeightPolicy(2),
        "spy_only": lambda: SingleAssetPolicy(assets, "SPY"),
        "shy_only": lambda: SingleAssetPolicy(assets, "SHY"),
    }


def _stores(
    *,
    validation_start: str = "2024-01-02",
) -> dict[str, PortfolioFeatureStore]:
    dataset = _dataset(validation_start)
    return {
        "validation_2024": PortfolioFeatureStore(dataset, "validation"),
        "historical_2022": PortfolioFeatureStore(dataset, "train"),
    }


def _dataset(validation_start: str) -> PortfolioDataset:
    train_dates = pd.date_range("2022-01-03", periods=70, freq="B")
    validation_dates = pd.date_range(validation_start, periods=70, freq="B")
    dates = train_dates.append(validation_dates)
    returns = np.column_stack(
        [
            0.001 * np.sin(np.arange(140)),
            0.001 * np.cos(np.arange(140)),
        ]
    ).astype(np.float32)
    return PortfolioDataset(
        dates=dates,
        splits=np.array(["train"] * 70 + ["validation"] * 70),
        market_features=np.zeros((140, 2), dtype=np.float32),
        returns=returns,
        asset_order=["SPY", "SHY"],
        feature_version="v1",
        observation_dim=4,
    )

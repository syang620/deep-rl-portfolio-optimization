"""Executable five-seed ensemble evaluation and diagnostics."""

from __future__ import annotations

import itertools
import json
import os
import shutil
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from portfolio_rl.data.feature_store import PortfolioFeatureStore
from portfolio_rl.evaluation.backtest import (
    BacktestResult,
    run_weight_policy_backtest,
    write_backtest_artifacts,
)
from portfolio_rl.evaluation.initialization_sensitivity import (
    InitializationSensitivityResult,
)
from portfolio_rl.policies.baseline_policies import WeightPolicy
from portfolio_rl.policies.ensemble_policy import MeanWeightEnsemblePolicy

TRADING_DAYS_PER_YEAR = 252
ENSEMBLE_STRATEGY = "five_seed_mean_weight_ensemble"


@dataclass(frozen=True)
class EnsembleCampaignResult:
    """In-memory outputs from executable ensemble evaluation."""

    metrics: pd.DataFrame
    member_targets: pd.DataFrame
    ensemble_targets: pd.DataFrame
    disagreement_metrics: pd.DataFrame
    seed_tracking: pd.DataFrame
    backtests: dict[tuple[str, str], BacktestResult]


def evaluate_ensemble_campaign(
    *,
    feature_stores: Mapping[str, PortfolioFeatureStore],
    member_policy_factories: Mapping[int, Callable[[], WeightPolicy]],
    baseline_policy_factories: Mapping[str, Callable[[], WeightPolicy]],
    representative_seed: int,
    configured_test_start_date: str | pd.Timestamp,
    primary_regime: str = "validation_2024",
    rebalance_frequency_trading_days: int = 5,
    transaction_cost_bps: float = 10.0,
) -> EnsembleCampaignResult:
    """Execute ensemble, member, and baseline policies on guarded windows."""
    if list(member_policy_factories) != [7, 42, 101, 202, 999]:
        raise ValueError("ensemble member seed order must be [7, 42, 101, 202, 999]")
    if representative_seed not in member_policy_factories:
        raise ValueError("representative_seed must be an ensemble member")
    if primary_regime not in feature_stores:
        raise ValueError("primary_regime must be present in feature_stores")

    backtests: dict[tuple[str, str], BacktestResult] = {}
    metric_rows = []
    member_target_frames = []
    ensemble_target_frames = []
    for regime, feature_store in feature_stores.items():
        _guard_test_access(feature_store, configured_test_start_date)
        for seed, factory in member_policy_factories.items():
            strategy = f"seed_{seed}"
            result = run_weight_policy_backtest(
                feature_store=feature_store,
                policy=factory(),
                strategy=strategy,
                rebalance_frequency_trading_days=(
                    rebalance_frequency_trading_days
                ),
                transaction_cost_bps=transaction_cost_bps,
            )
            backtests[(regime, strategy)] = result
            metric_rows.append(_metrics_row(regime, strategy, result))

        ensemble = MeanWeightEnsemblePolicy(
            member_policies={
                f"seed_{seed}": factory()
                for seed, factory in member_policy_factories.items()
            }
        )
        ensemble_result = run_weight_policy_backtest(
            feature_store=feature_store,
            policy=ensemble,
            strategy=ENSEMBLE_STRATEGY,
            rebalance_frequency_trading_days=rebalance_frequency_trading_days,
            transaction_cost_bps=transaction_cost_bps,
        )
        backtests[(regime, ENSEMBLE_STRATEGY)] = ensemble_result
        metric_rows.append(_metrics_row(regime, ENSEMBLE_STRATEGY, ensemble_result))
        member_target_frames.append(
            _member_target_frame(
                regime=regime,
                records=ensemble.member_target_records,
                asset_order=feature_store.asset_order,
            )
        )
        ensemble_target_frames.append(
            _ensemble_target_frame(regime, ensemble_result)
        )

        if regime == primary_regime:
            for strategy, factory in baseline_policy_factories.items():
                result = run_weight_policy_backtest(
                    feature_store=feature_store,
                    policy=factory(),
                    strategy=strategy,
                    rebalance_frequency_trading_days=(
                        rebalance_frequency_trading_days
                    ),
                    transaction_cost_bps=transaction_cost_bps,
                )
                backtests[(regime, strategy)] = result
                metric_rows.append(_metrics_row(regime, strategy, result))

    member_targets = pd.concat(member_target_frames, ignore_index=True)
    ensemble_targets = pd.concat(ensemble_target_frames, ignore_index=True)
    disagreement = calculate_disagreement_metrics(member_targets)
    tracking = calculate_seed_tracking(
        member_targets=member_targets,
        ensemble_targets=ensemble_targets,
        backtests=backtests,
    )
    return EnsembleCampaignResult(
        metrics=pd.DataFrame(metric_rows),
        member_targets=member_targets,
        ensemble_targets=ensemble_targets,
        disagreement_metrics=disagreement,
        seed_tracking=tracking,
        backtests=backtests,
    )


def calculate_disagreement_metrics(member_targets: pd.DataFrame) -> pd.DataFrame:
    """Summarize cross-seed target disagreement at every decision."""
    rows = []
    group_columns = ["regime", "date", "decision_step"]
    for keys, decision in member_targets.groupby(group_columns, sort=True):
        pivot = decision.pivot(
            index="member",
            columns="ticker",
            values="target_weight",
        ).sort_index()
        values = pivot.to_numpy(dtype=np.float64)
        pair_distances = [
            0.5 * float(np.abs(values[left] - values[right]).sum())
            for left, right in itertools.combinations(range(len(values)), 2)
        ]
        dominant_assets = pivot.idxmax(axis=1)
        dominant_mode = str(dominant_assets.mode().iloc[0])
        rows.append(
            {
                "regime": keys[0],
                "date": keys[1],
                "decision_step": int(keys[2]),
                "member_count": len(pivot),
                "mean_pairwise_target_half_l1": float(np.mean(pair_distances)),
                "median_pairwise_target_half_l1": float(
                    np.median(pair_distances)
                ),
                "max_pairwise_target_half_l1": float(np.max(pair_distances)),
                "mean_asset_target_std": float(
                    np.std(values, axis=0, ddof=0).mean()
                ),
                "max_asset_target_std": float(
                    np.std(values, axis=0, ddof=0).max()
                ),
                "dominant_asset_mode": dominant_mode,
                "dominant_asset_agreement": float(
                    np.mean(dominant_assets == dominant_mode)
                ),
            }
        )
    return pd.DataFrame(rows)


def calculate_seed_tracking(
    *,
    member_targets: pd.DataFrame,
    ensemble_targets: pd.DataFrame,
    backtests: Mapping[tuple[str, str], BacktestResult],
) -> pd.DataFrame:
    """Calculate target distance and realized-return tracking error by seed."""
    rows = []
    for regime in sorted(member_targets["regime"].unique()):
        regime_members = member_targets[member_targets["regime"] == regime]
        regime_ensemble = ensemble_targets[ensemble_targets["regime"] == regime]
        ensemble_returns = backtests[
            (regime, ENSEMBLE_STRATEGY)
        ].nav["daily_return"].to_numpy(dtype=np.float64)
        for member, path in regime_members.groupby("member", sort=True):
            paired = path.merge(
                regime_ensemble,
                on=["regime", "date", "decision_step", "ticker"],
                validate="one_to_one",
            )
            target_distances = (
                paired.assign(
                    absolute_difference=np.abs(
                        paired["target_weight"]
                        - paired["ensemble_target_weight"]
                    )
                )
                .groupby(["date", "decision_step"])["absolute_difference"]
                .sum()
                * 0.5
            )
            seed_returns = backtests[
                (regime, str(member))
            ].nav["daily_return"].to_numpy(dtype=np.float64)
            if len(seed_returns) != len(ensemble_returns):
                raise ValueError("seed and ensemble return paths do not align")
            active_returns = seed_returns - ensemble_returns
            annualized_tracking_error = (
                float(np.std(active_returns, ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))
                if len(active_returns) >= 2
                else None
            )
            rows.append(
                {
                    "regime": regime,
                    "member": member,
                    "mean_target_half_l1_to_ensemble": float(
                        target_distances.mean()
                    ),
                    "terminal_target_half_l1_to_ensemble": float(
                        target_distances.iloc[-1]
                    ),
                    "annualized_return_tracking_error": (
                        annualized_tracking_error
                    ),
                }
            )
    return pd.DataFrame(rows)


def write_ensemble_artifacts(
    *,
    result: EnsembleCampaignResult,
    initialization_result: InitializationSensitivityResult,
    output_dir: str | Path,
    manifest: Mapping[str, Any],
) -> dict[str, Path]:
    """Atomically write an immutable ensemble research bundle."""
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(f"ensemble output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    try:
        outputs = {
            "manifest": temporary / "ensemble_manifest.json",
            "metrics": temporary / "comparison_metrics.csv",
            "member_targets": temporary / "member_targets.parquet",
            "ensemble_targets": temporary / "ensemble_targets.parquet",
            "disagreement": temporary / "disagreement_metrics.parquet",
            "tracking": temporary / "seed_tracking.csv",
            "report": temporary / "ensemble_report.md",
        }
        result.metrics.to_csv(outputs["metrics"], index=False)
        result.member_targets.to_parquet(outputs["member_targets"], index=False)
        result.ensemble_targets.to_parquet(
            outputs["ensemble_targets"],
            index=False,
        )
        result.disagreement_metrics.to_parquet(
            outputs["disagreement"],
            index=False,
        )
        result.seed_tracking.to_csv(outputs["tracking"], index=False)
        for (regime, strategy), backtest in result.backtests.items():
            write_backtest_artifacts(
                backtest,
                temporary / "backtest" / regime / strategy,
            )
        _write_initialization_outputs(
            initialization_result,
            temporary / "initialization_sensitivity",
        )
        full_manifest = {
            "schema_version": 1,
            "created_at": datetime.now(UTC).isoformat(),
            **dict(manifest),
        }
        outputs["manifest"].write_text(
            json.dumps(full_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        outputs["report"].write_text(
            format_ensemble_report(
                result=result,
                initialization_result=initialization_result,
                manifest=full_manifest,
            ),
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {
        key: destination / path.relative_to(temporary)
        for key, path in outputs.items()
    }


def format_ensemble_report(
    *,
    result: EnsembleCampaignResult,
    initialization_result: InitializationSensitivityResult,
    manifest: Mapping[str, Any],
) -> str:
    """Format the executable ensemble research report."""
    lines = [
        "# Five-Seed Mean-Weight Ensemble Report",
        "",
        "## Governance",
        "",
        "- The ensemble averages executable PPO target weights, not raw actions.",
        "- All primary 2024 comparisons start from equal weight.",
        "- 2024 is consumed development/selection data, not independent test evidence.",
        "- The 2022 window is an in-sample historical behavior diagnostic.",
        "- Final-test access: **none**.",
        "",
        f"Frozen member order: `{manifest['member_seed_order']}`",
        "",
        "## 2024 primary comparison",
        "",
        *_metrics_table(
            result.metrics,
            regime="validation_2024",
            strategies=[
                "seed_42",
                ENSEMBLE_STRATEGY,
                "equal_weight_weekly",
                "inverse_volatility",
                "buy_and_hold_equal_weight",
                "spy_only",
                "shy_only",
            ],
        ),
        "",
        _drawdown_summary(result.metrics),
        "",
        "## 2022 ensemble versus individual seeds",
        "",
        *_metrics_table(
            result.metrics,
            regime="historical_2022",
            strategies=[
                ENSEMBLE_STRATEGY,
                "seed_7",
                "seed_42",
                "seed_101",
                "seed_202",
                "seed_999",
            ],
        ),
        "",
        _seed_dispersion_summary(result.metrics),
        "",
        "## First-rebalance execution",
        "",
        _first_rebalance_summary(result.member_targets),
        "",
        "## Cross-seed diagnostics",
        "",
    ]
    for regime, frame in result.disagreement_metrics.groupby("regime", sort=True):
        lines.append(
            f"- {regime}: mean pairwise target half-L1 "
            f"`{frame['mean_pairwise_target_half_l1'].mean():.6f}`; "
            "median decision-level pairwise target half-L1 "
            f"`{frame['median_pairwise_target_half_l1'].median():.6f}`; "
            "mean dominant-asset agreement "
            f"`{frame['dominant_asset_agreement'].mean():.2%}`."
        )
    lines.extend(
        [
            "",
            "### Seed-to-ensemble tracking",
            "",
            *_tracking_table(result.seed_tracking),
            "",
            "## Ensemble initialization sensitivity",
            "",
            *_initialization_table(initialization_result),
            "",
            "### Initialization convergence",
            "",
            *_initialization_convergence_table(initialization_result),
            "",
            (
                "Equal weight remains the headline initialization. The "
                "inverse-volatility and 100% SHY starts are diagnostics only."
            ),
            "",
            "## Known follow-up",
            "",
            (
                "The existing inverse-volatility baseline still receives only "
                "split-bounded trailing returns. Before walk-forward and final "
                "comparisons it must use a full past-only lookback across the "
                "evaluation-window boundary; PR 13 does not change that baseline."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _guard_test_access(
    feature_store: PortfolioFeatureStore,
    configured_test_start_date: str | pd.Timestamp,
) -> None:
    test_start = pd.Timestamp(configured_test_start_date).normalize()
    evaluation_start = feature_store.date_at(0).normalize()
    if feature_store.split == "test" or evaluation_start >= test_start:
        raise ValueError(
            "ensemble evaluation must not access the final test period: "
            f"split={feature_store.split}, evaluation_start={evaluation_start.date()}"
        )


def _member_target_frame(
    *,
    regime: str,
    records: tuple[Any, ...],
    asset_order: list[str],
) -> pd.DataFrame:
    rows = []
    for record in records:
        for ticker, weight in zip(asset_order, record.weights, strict=True):
            rows.append(
                {
                    "regime": regime,
                    "date": record.date,
                    "decision_step": record.decision_step,
                    "member": record.member,
                    "seed": int(record.member.removeprefix("seed_")),
                    "ticker": ticker,
                    "target_weight": weight,
                    "live_current_weight": record.live_current_weights[
                        asset_order.index(ticker)
                    ],
                }
            )
    return pd.DataFrame(rows)


def _ensemble_target_frame(
    regime: str,
    result: BacktestResult,
) -> pd.DataFrame:
    frame = result.weights_target[["date", "ticker", "target_weight"]].copy()
    frame.insert(0, "regime", regime)
    date_steps = {
        date: step for step, date in enumerate(frame["date"].drop_duplicates())
    }
    frame.insert(2, "decision_step", frame["date"].map(date_steps).astype(int))
    return frame.rename(columns={"target_weight": "ensemble_target_weight"})


def _metrics_row(
    regime: str,
    strategy: str,
    result: BacktestResult,
) -> dict[str, Any]:
    return {"regime": regime, "strategy": strategy, **result.metrics}


def _write_initialization_outputs(
    result: InitializationSensitivityResult,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True)
    result.results.to_csv(
        output_dir / "results_by_model_and_initializer.csv",
        index=False,
    )
    result.allocation_paths.to_parquet(
        output_dir / "allocation_paths.parquet",
        index=False,
    )
    result.nav_paths.to_parquet(output_dir / "nav_paths.parquet", index=False)
    result.convergence_metrics.to_csv(
        output_dir / "convergence_metrics.csv",
        index=False,
    )
    result.convergence_summary.to_csv(
        output_dir / "convergence_summary.csv",
        index=False,
    )


def _metrics_table(
    metrics: pd.DataFrame,
    *,
    regime: str,
    strategies: list[str],
) -> list[str]:
    selected = metrics[metrics["regime"] == regime].set_index("strategy")
    lines = [
        "| Strategy | Total return | Sharpe | Max drawdown | Avg turnover | Cost drag |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for strategy in strategies:
        row = selected.loc[strategy]
        lines.append(
            f"| {strategy} | {row['total_return']:.4%} | "
            f"{row['sharpe_ratio']:.3f} | {row['max_drawdown']:.4%} | "
            f"{row['average_weekly_turnover']:.4%} | "
            f"{row['transaction_cost_drag']:.4%} |"
        )
    return lines


def _initialization_table(
    result: InitializationSensitivityResult,
) -> list[str]:
    lines = [
        "| Initializer | Total return | Sharpe | Avg turnover | Cost drag |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in result.results.itertuples(index=False):
        lines.append(
            f"| {row.initializer} | {row.total_return:.4%} | "
            f"{row.sharpe_ratio:.3f} | {row.average_weekly_turnover:.4%} | "
            f"{row.transaction_cost_drag:.4%} |"
        )
    return lines


def _drawdown_summary(metrics: pd.DataFrame) -> str:
    selected = metrics[metrics["regime"] == "validation_2024"].set_index(
        "strategy"
    )
    return (
        "Maximum drawdown comparison: ensemble "
        f"`{selected.loc[ENSEMBLE_STRATEGY, 'max_drawdown']:.4%}`, seed 42 "
        f"`{selected.loc['seed_42', 'max_drawdown']:.4%}`, and weekly equal "
        f"weight `{selected.loc['equal_weight_weekly', 'max_drawdown']:.4%}`."
    )


def _seed_dispersion_summary(metrics: pd.DataFrame) -> str:
    selected = metrics[metrics["regime"] == "historical_2022"].set_index(
        "strategy"
    )
    seed_returns = selected.loc[
        ["seed_7", "seed_42", "seed_101", "seed_202", "seed_999"],
        "total_return",
    ]
    ensemble_return = float(selected.loc[ENSEMBLE_STRATEGY, "total_return"])
    return (
        f"The executed ensemble returned `{ensemble_return:.4%}` versus the "
        f"individual-seed range `{seed_returns.min():.4%}` to "
        f"`{seed_returns.max():.4%}`. It converts the roughly 18-percentage-point "
        "seed-selection dispersion into one near-flat tradable path, but does "
        "not imply that the underlying individual-seed dispersion disappeared."
    )


def _first_rebalance_summary(member_targets: pd.DataFrame) -> str:
    first = member_targets[
        (member_targets["regime"] == "validation_2024")
        & (member_targets["decision_step"] == 0)
    ]
    if first.empty:
        raise ValueError("member targets missing first validation decision")
    unique_by_ticker = first.groupby("ticker")["live_current_weight"].nunique()
    if not unique_by_ticker.eq(1).all():
        raise ValueError("ensemble members received different live current weights")
    weights = (
        first.drop_duplicates("ticker")
        .sort_values("ticker")
        .loc[:, ["ticker", "live_current_weight"]]
    )
    formatted = ", ".join(
        f"{row.ticker} {row.live_current_weight:.4%}"
        for row in weights.itertuples(index=False)
    )
    date = pd.Timestamp(first["date"].iloc[0]).date().isoformat()
    member_count = first["member"].nunique()
    return (
        f"At the first 2024 decision (`{date}`), all {member_count} members "
        "received the same ensemble portfolio live current weights "
        f"({formatted}). Members did not receive separate hypothetical member "
        "portfolios; their targets were averaged against this shared live state."
    )


def _initialization_convergence_table(
    result: InitializationSensitivityResult,
) -> list[str]:
    lines = [
        (
            "| Initializer A | Initializer B | Mean target half-L1 | "
            "Target convergence | Mean pre-trade half-L1 | "
            "Pre-trade convergence |"
        ),
        "| --- | --- | ---: | --- | ---: | --- |",
    ]
    for row in result.convergence_summary.itertuples(index=False):
        lines.append(
            f"| {row.initializer_a} | {row.initializer_b} | "
            f"{row.mean_target_distance:.6f} | "
            f"{row.target_convergence_date or 'n/a'} | "
            f"{row.mean_pre_trade_distance:.6f} | "
            f"{row.pre_trade_convergence_date or 'n/a'} |"
        )
    return lines


def _tracking_table(tracking: pd.DataFrame) -> list[str]:
    lines = [
        (
            "| Regime | Member | Mean target half-L1 | Terminal target "
            "half-L1 | Annualized return tracking error |"
        ),
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in tracking.itertuples(index=False):
        lines.append(
            f"| {row.regime} | {row.member} | "
            f"{row.mean_target_half_l1_to_ensemble:.6f} | "
            f"{row.terminal_target_half_l1_to_ensemble:.6f} | "
            f"{row.annualized_return_tracking_error:.4%} |"
        )
    return lines

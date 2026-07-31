"""Partial-rebalancing turnover/performance frontier evaluation."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Callable, Mapping, Sequence
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
from portfolio_rl.policies.baseline_policies import WeightPolicy
from portfolio_rl.policies.overlays import PartialRebalancePolicy


@dataclass(frozen=True)
class TurnoverOverlayResult:
    """In-memory PR 14 frontier and audit outputs."""

    overlay_results: pd.DataFrame
    target_audit: pd.DataFrame
    turnover_frontier: pd.DataFrame
    cost_frontier: pd.DataFrame
    hurdle_metrics: dict[str, float | None]
    backtests: dict[tuple[str, float], BacktestResult]
    hurdle_backtest: BacktestResult


def evaluate_turnover_overlay_study(
    *,
    feature_store: PortfolioFeatureStore,
    candidate_policy_factories: Mapping[str, Callable[[], WeightPolicy]],
    alphas: Sequence[float],
    hurdle_policy_factory: Callable[[], WeightPolicy],
    configured_test_start_date: str | pd.Timestamp,
    rebalance_frequency_trading_days: int = 5,
    transaction_cost_bps: float = 10.0,
) -> TurnoverOverlayResult:
    """Execute a fixed partial-rebalancing grid without selecting an alpha."""
    _guard_test_access(feature_store, configured_test_start_date)
    if list(alphas) != [0.25, 0.5, 0.75, 1.0]:
        raise ValueError("alphas must be exactly [0.25, 0.5, 0.75, 1.0]")
    if not candidate_policy_factories:
        raise ValueError("candidate_policy_factories must not be empty")

    hurdle = run_weight_policy_backtest(
        feature_store=feature_store,
        policy=hurdle_policy_factory(),
        strategy="equal_weight_weekly",
        rebalance_frequency_trading_days=rebalance_frequency_trading_days,
        transaction_cost_bps=transaction_cost_bps,
    )
    backtests = {}
    metric_rows = []
    target_frames = []
    for candidate, factory in candidate_policy_factories.items():
        for alpha in alphas:
            overlay = PartialRebalancePolicy(
                base_policy=factory(),
                alpha=float(alpha),
            )
            strategy = f"{candidate}_alpha_{alpha:.2f}"
            result = run_weight_policy_backtest(
                feature_store=feature_store,
                policy=overlay,
                strategy=strategy,
                rebalance_frequency_trading_days=(
                    rebalance_frequency_trading_days
                ),
                transaction_cost_bps=transaction_cost_bps,
            )
            backtests[(candidate, float(alpha))] = result
            metric_rows.append(
                {
                    "candidate": candidate,
                    "alpha": float(alpha),
                    **result.metrics,
                }
            )
            target_frames.append(
                _target_audit_frame(
                    candidate=candidate,
                    alpha=float(alpha),
                    records=overlay.records,
                    asset_order=feature_store.asset_order,
                )
            )

    results = _derive_frontier_metrics(
        pd.DataFrame(metric_rows),
        hurdle_metrics=hurdle.metrics,
    )
    target_audit = pd.concat(target_frames, ignore_index=True)
    _reconcile_executed_turnover(
        target_audit=target_audit,
        backtests=backtests,
    )
    turnover_columns = [
        "candidate",
        "alpha",
        "total_return",
        "active_return_vs_equal_weight",
        "return_advantage_retention",
        "average_weekly_turnover",
        "turnover_reduction_vs_alpha_1",
        "turnover_reduction_fraction_vs_alpha_1",
        "sharpe_ratio",
        "max_drawdown",
    ]
    cost_columns = [
        "candidate",
        "alpha",
        "transaction_cost_drag",
        "cost_drag_reduction_vs_alpha_1",
        "average_weekly_turnover",
        "annualized_turnover",
    ]
    return TurnoverOverlayResult(
        overlay_results=results,
        target_audit=target_audit,
        turnover_frontier=results[turnover_columns].copy(),
        cost_frontier=results[cost_columns].copy(),
        hurdle_metrics=hurdle.metrics,
        backtests=backtests,
        hurdle_backtest=hurdle,
    )


def write_turnover_overlay_artifacts(
    *,
    result: TurnoverOverlayResult,
    output_dir: str | Path,
    manifest: Mapping[str, Any],
) -> dict[str, Path]:
    """Atomically write an immutable PR 14 research bundle."""
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(f"turnover overlay output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    try:
        outputs = {
            "manifest": temporary / "overlay_manifest.json",
            "results": temporary / "overlay_results.csv",
            "targets": temporary / "raw_and_executed_targets.parquet",
            "turnover_frontier": temporary / "turnover_frontier.csv",
            "cost_frontier": temporary / "cost_frontier.csv",
            "report": temporary / "turnover_overlay_report.md",
        }
        result.overlay_results.to_csv(outputs["results"], index=False)
        result.target_audit.to_parquet(outputs["targets"], index=False)
        result.turnover_frontier.to_csv(outputs["turnover_frontier"], index=False)
        result.cost_frontier.to_csv(outputs["cost_frontier"], index=False)
        write_backtest_artifacts(
            result.hurdle_backtest,
            temporary / "backtest" / "equal_weight_weekly",
        )
        for (candidate, alpha), backtest in result.backtests.items():
            write_backtest_artifacts(
                backtest,
                temporary
                / "backtest"
                / candidate
                / f"alpha_{str(alpha).replace('.', '_')}",
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
            format_turnover_overlay_report(
                result=result,
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


def format_turnover_overlay_report(
    *,
    result: TurnoverOverlayResult,
    manifest: Mapping[str, Any],
) -> str:
    """Format the non-selective 2024 turnover/performance frontier."""
    ensemble = result.overlay_results[
        result.overlay_results["candidate"] == "five_seed_mean_weight_ensemble"
    ]
    alpha_one = ensemble[ensemble["alpha"] == 1.0].iloc[0]
    advantage_pp = 100.0 * float(alpha_one["active_return_vs_equal_weight"])
    lines = [
        "# Partial-Rebalancing Turnover Frontier",
        "",
        "## Governance",
        "",
        "- The five-seed ensemble is the primary candidate.",
        "- Seed 42 is a secondary diagnostic.",
        "- All strategies begin equal weight.",
        "- 2024 is consumed development/selection data, not independent evidence.",
        "- Final-test access: **none**.",
        "- No alpha is selected in PR 14; all four advance to walk-forward analysis.",
        "",
        (
            "The unmodified ensemble's net return advantage over weekly equal "
            f"weight is `{advantage_pp:.4f}` percentage points."
        ),
        "",
        "## Ensemble frontier",
        "",
        *_frontier_table(ensemble),
        "",
        "## Seed-42 diagnostic",
        "",
        *_frontier_table(
            result.overlay_results[
                result.overlay_results["candidate"] == "seed_42"
            ]
        ),
        "",
        "## Execution contract",
        "",
        "At every decision:",
        "",
        "1. Build the observation using the overlay portfolio's live drifted weights.",
        "2. Ask the candidate policy for a new raw target.",
        "3. Move alpha of the way from live weights toward that target.",
        "4. Charge half-L1 turnover against the live weights.",
        "5. Drift the executed portfolio through the holding period.",
        "",
        (
            "Raw and executed targets are stored separately. Targets are "
            "recomputed from each overlay path rather than replayed from PR 13. "
            "Alpha 1.00 is the exact unmodified policy."
        ),
        "",
        "## Interpretation",
        "",
        (
            "This report establishes a 2024 turnover/performance frontier only. "
            "It does not identify a preferred alpha. Robust selection is deferred "
            "to the predeclared walk-forward and execution-stress gates."
        ),
        "",
        f"Canonical PR 13 manifest SHA-256: `{manifest['pr13_manifest_sha256']}`",
        "",
    ]
    return "\n".join(lines)


def _derive_frontier_metrics(
    results: pd.DataFrame,
    *,
    hurdle_metrics: Mapping[str, float | None],
) -> pd.DataFrame:
    hurdle_return = hurdle_metrics.get("total_return")
    hurdle_sharpe = hurdle_metrics.get("sharpe_ratio")
    hurdle_drawdown = hurdle_metrics.get("max_drawdown")
    if hurdle_return is None or hurdle_sharpe is None or hurdle_drawdown is None:
        raise ValueError("equal-weight hurdle metrics must be complete")
    derived_frames = []
    for candidate, frame in results.groupby("candidate", sort=False):
        frame = frame.sort_values("alpha").copy()
        unmodified = frame[frame["alpha"] == 1.0]
        if len(unmodified) != 1:
            raise ValueError(f"candidate missing unique alpha 1.0: {candidate}")
        alpha_one = unmodified.iloc[0]
        unmodified_advantage = float(alpha_one["total_return"]) - hurdle_return
        if unmodified_advantage <= 0.0:
            raise ValueError(
                f"candidate has no positive alpha-1 advantage: {candidate}"
            )
        frame["active_return_vs_equal_weight"] = (
            frame["total_return"] - hurdle_return
        )
        frame["active_sharpe_vs_equal_weight"] = (
            frame["sharpe_ratio"] - hurdle_sharpe
        )
        frame["drawdown_difference_vs_equal_weight"] = (
            frame["max_drawdown"] - hurdle_drawdown
        )
        frame["return_advantage_retention"] = (
            frame["active_return_vs_equal_weight"] / unmodified_advantage
        )
        frame["turnover_reduction_vs_alpha_1"] = (
            float(alpha_one["average_weekly_turnover"])
            - frame["average_weekly_turnover"]
        )
        frame["turnover_reduction_fraction_vs_alpha_1"] = (
            frame["turnover_reduction_vs_alpha_1"]
            / float(alpha_one["average_weekly_turnover"])
        )
        frame["cost_drag_reduction_vs_alpha_1"] = (
            float(alpha_one["transaction_cost_drag"])
            - frame["transaction_cost_drag"]
        )
        frame["drawdown_difference_vs_alpha_1"] = (
            frame["max_drawdown"] - float(alpha_one["max_drawdown"])
        )
        derived_frames.append(frame)
    return pd.concat(derived_frames, ignore_index=True)


def _target_audit_frame(
    *,
    candidate: str,
    alpha: float,
    records: tuple[Any, ...],
    asset_order: list[str],
) -> pd.DataFrame:
    rows = []
    for record in records:
        for index, ticker in enumerate(asset_order):
            rows.append(
                {
                    "candidate": candidate,
                    "alpha": alpha,
                    "date": record.date,
                    "decision_step": record.decision_step,
                    "ticker": ticker,
                    "current_weight": record.current_weights[index],
                    "raw_policy_target": record.raw_policy_target[index],
                    "executed_target": record.executed_target[index],
                    "raw_half_l1_turnover": record.raw_half_l1_turnover,
                    "executed_half_l1_turnover": (
                        record.executed_half_l1_turnover
                    ),
                    "executed_trade_weight": (
                        record.executed_target[index]
                        - record.current_weights[index]
                    ),
                }
            )
    return pd.DataFrame(rows)


def _reconcile_executed_turnover(
    *,
    target_audit: pd.DataFrame,
    backtests: Mapping[tuple[str, float], BacktestResult],
) -> None:
    for (candidate, alpha), backtest in backtests.items():
        audit = target_audit[
            (target_audit["candidate"] == candidate)
            & np.isclose(target_audit["alpha"], alpha)
        ]
        observed = (
            audit.drop_duplicates(["date", "decision_step"])
            .sort_values("decision_step")["executed_half_l1_turnover"]
            .to_numpy(dtype=np.float64)
        )
        expected = backtest.costs.sort_values("date")["turnover"].to_numpy(
            dtype=np.float64
        )
        if not np.allclose(observed, expected, atol=1e-12, rtol=0.0):
            raise ValueError(
                f"executed turnover does not reconcile: {candidate} alpha={alpha}"
            )


def _guard_test_access(
    feature_store: PortfolioFeatureStore,
    configured_test_start_date: str | pd.Timestamp,
) -> None:
    test_start = pd.Timestamp(configured_test_start_date).normalize()
    evaluation_start = feature_store.date_at(0).normalize()
    if feature_store.split == "test" or evaluation_start >= test_start:
        raise ValueError(
            "turnover overlay study must not access the final test period"
        )


def _frontier_table(frame: pd.DataFrame) -> list[str]:
    lines = [
        (
            "| Alpha | Return | Active return | Advantage retained | Sharpe | "
            "Max drawdown | Avg turnover | Turnover reduction | Cost drag |"
        ),
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in frame.sort_values("alpha").itertuples(index=False):
        lines.append(
            f"| {row.alpha:.2f} | {row.total_return:.4%} | "
            f"{row.active_return_vs_equal_weight:.4%} | "
            f"{row.return_advantage_retention:.2%} | "
            f"{row.sharpe_ratio:.3f} | {row.max_drawdown:.4%} | "
            f"{row.average_weekly_turnover:.4%} | "
            f"{row.turnover_reduction_fraction_vs_alpha_1:.2%} | "
            f"{row.transaction_cost_drag:.4%} |"
        )
    return lines

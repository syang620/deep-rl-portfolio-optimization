"""Initialization-sensitivity diagnostics for frozen portfolio policies."""

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
from portfolio_rl.evaluation.backtest import BacktestResult, run_weight_policy_backtest
from portfolio_rl.evaluation.initialization import InitialPortfolioProvider
from portfolio_rl.evaluation.metrics import calculate_performance_metrics
from portfolio_rl.policies.baseline_policies import WeightPolicy

OUTPUT_FILENAMES = {
    "results": "results_by_model_and_initializer.csv",
    "allocations": "allocation_paths.parquet",
    "nav": "nav_paths.parquet",
    "convergence": "convergence_metrics.csv",
    "convergence_summary": "convergence_summary.csv",
    "manifest": "run_manifest.json",
    "report": "initialization_sensitivity_report.md",
}


@dataclass(frozen=True)
class InitializationSensitivityResult:
    """In-memory outputs from initialization-sensitivity evaluation."""

    results: pd.DataFrame
    allocation_paths: pd.DataFrame
    nav_paths: pd.DataFrame
    convergence_metrics: pd.DataFrame
    convergence_summary: pd.DataFrame
    backtests: dict[tuple[str, str], BacktestResult]


def evaluate_initialization_sensitivity(
    *,
    feature_store: PortfolioFeatureStore,
    candidate_policy_factories: Mapping[str, Callable[[], WeightPolicy]],
    initializers: Mapping[str, InitialPortfolioProvider],
    configured_test_start_date: str | pd.Timestamp,
    rebalance_frequency_trading_days: int = 5,
    transaction_cost_bps: float = 10.0,
    convergence_threshold: float = 0.05,
    convergence_consecutive_decisions: int = 4,
) -> InitializationSensitivityResult:
    """Evaluate fresh policy instances from multiple endowed portfolios."""
    _guard_test_access(feature_store, configured_test_start_date)
    if not candidate_policy_factories:
        raise ValueError("candidate_policy_factories must not be empty")
    if not initializers:
        raise ValueError("initializers must not be empty")
    if not 0.0 <= convergence_threshold <= 1.0:
        raise ValueError("convergence_threshold must be between zero and one")
    if convergence_consecutive_decisions <= 0:
        raise ValueError("convergence_consecutive_decisions must be positive")

    backtests: dict[tuple[str, str], BacktestResult] = {}
    result_rows: list[dict[str, Any]] = []
    allocation_frames = []
    nav_frames = []
    for candidate, policy_factory in candidate_policy_factories.items():
        for initializer_name, initializer in initializers.items():
            policy = policy_factory()
            result = run_weight_policy_backtest(
                feature_store=feature_store,
                policy=policy,
                strategy=candidate,
                rebalance_frequency_trading_days=(
                    rebalance_frequency_trading_days
                ),
                transaction_cost_bps=transaction_cost_bps,
                initial_portfolio_provider=initializer,
            )
            key = (candidate, initializer_name)
            backtests[key] = result
            result_rows.append(
                _result_row(
                    candidate=candidate,
                    initializer=initializer_name,
                    result=result,
                    rebalance_frequency_trading_days=(
                        rebalance_frequency_trading_days
                    ),
                )
            )
            allocation_frames.append(
                _allocation_paths(candidate, initializer_name, result)
            )
            nav_frames.append(
                result.nav.assign(
                    candidate=candidate,
                    initializer=initializer_name,
                )
            )

    allocations = pd.concat(allocation_frames, ignore_index=True)
    nav_paths = pd.concat(nav_frames, ignore_index=True)
    convergence, convergence_summary = _convergence_outputs(
        allocations,
        threshold=convergence_threshold,
        consecutive_decisions=convergence_consecutive_decisions,
    )
    return InitializationSensitivityResult(
        results=pd.DataFrame(result_rows),
        allocation_paths=allocations,
        nav_paths=nav_paths,
        convergence_metrics=convergence,
        convergence_summary=convergence_summary,
        backtests=backtests,
    )


def write_initialization_sensitivity_artifacts(
    *,
    result: InitializationSensitivityResult,
    output_dir: str | Path,
    manifest: Mapping[str, Any],
) -> dict[str, Path]:
    """Atomically write PR 12 outputs and refuse to overwrite prior evidence."""
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(
            f"initialization sensitivity output already exists: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    try:
        outputs = {
            key: temporary / filename for key, filename in OUTPUT_FILENAMES.items()
        }
        result.results.to_csv(outputs["results"], index=False)
        result.allocation_paths.to_parquet(outputs["allocations"], index=False)
        result.nav_paths.to_parquet(outputs["nav"], index=False)
        result.convergence_metrics.to_csv(outputs["convergence"], index=False)
        result.convergence_summary.to_csv(
            outputs["convergence_summary"],
            index=False,
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
            format_initialization_sensitivity_report(
                result,
                manifest=full_manifest,
            ),
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {
        key: destination / filename for key, filename in OUTPUT_FILENAMES.items()
    }


def format_initialization_sensitivity_report(
    result: InitializationSensitivityResult,
    *,
    manifest: Mapping[str, Any],
) -> str:
    """Format the seed-scoped, non-selection interpretation contract."""
    artifact_directory, model_hash, freeze_manifest_hash = _report_provenance(
        manifest
    )
    target_converged = (
        bool(result.convergence_summary["converged"].all())
        if not result.convergence_summary.empty
        else False
    )
    sensitivity = "is not" if target_converged else "is"
    lines = [
        "# Seed-42 Initialization-Sensitivity Report",
        "",
        "## Scope",
        "",
        (
            f"The frozen seed-42 policy {sensitivity} materially sensitive "
            "to the portfolio present at the first 2024 decision under the "
            "configured target half-L1 convergence diagnostic."
        ),
        "",
        (
            "This diagnostic does not establish that PPO generally is "
            "initialization-insensitive."
        ),
        "",
        (
            "The evaluation window is **2024 consumed development/selection "
            "data**, not independent test evidence."
        ),
        "",
        (
            "Each scenario begins at NAV 1.0 already holding its configured "
            "portfolio. Establishment is costless; the first policy rebalance "
            "incurs normal turnover and cost from those endowed weights."
        ),
        "",
        "## Performance diagnostics",
        "",
        (
            "| Initializer | Total return | Sharpe | Max drawdown | Avg weekly "
            "turnover | Cost drag | 4-week return | 4-week Sharpe | "
            "12-week return | 12-week Sharpe |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in result.results.itertuples(index=False):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.initializer),
                    _format_metric(row.total_return),
                    _format_metric(row.sharpe_ratio),
                    _format_metric(row.max_drawdown),
                    _format_metric(row.average_weekly_turnover),
                    _format_metric(row.transaction_cost_drag),
                    _format_metric(row.first_4_weeks_total_return),
                    _format_metric(row.first_4_weeks_sharpe_ratio),
                    _format_metric(row.first_12_weeks_total_return),
                    _format_metric(row.first_12_weeks_sharpe_ratio),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Convergence diagnostics",
            "",
            (
                "| Initializer A | Initializer B | Terminal target half-L1 | "
                "Mean target half-L1 | Target convergence | Terminal pre-trade "
                "half-L1 | Mean pre-trade half-L1 | Pre-trade convergence |"
            ),
            "| --- | --- | ---: | ---: | --- | ---: | ---: | --- |",
        ]
    )
    for row in result.convergence_summary.itertuples(index=False):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.initializer_a),
                    str(row.initializer_b),
                    _format_metric(row.terminal_target_distance),
                    _format_metric(row.mean_target_distance),
                    _format_metric(row.target_convergence_date),
                    _format_metric(row.terminal_pre_trade_distance),
                    _format_metric(row.mean_pre_trade_distance),
                    _format_metric(row.pre_trade_convergence_date),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Provenance",
            "",
            f"- Canonical artifact directory: `{artifact_directory}`",
            f"- Frozen seed-42 model SHA-256: `{model_hash}`",
            f"- Research freeze-manifest SHA-256: `{freeze_manifest_hash}`",
            "",
            "## Governance",
            "",
            "- Equal weight remains the official headline initialization.",
            (
                "- Inverse-volatility and 100% SHY starts are diagnostics; no "
                "initializer may be selected because it performs better."
            ),
            (
                "- Ensemble initialization sensitivity is deferred until PR 13; "
                "broader conclusions require that ensemble and later walk-forward "
                "folds."
            ),
            "",
            "## Convergence convention",
            "",
            (
                "The primary metric is half-L1 portfolio distance: "
                "`0.5 * abs(weights_a - weights_b).sum()`."
            ),
            (
                "A threshold of 0.05 means at most 5% one-way reallocation is "
                "required to match for four consecutive weekly decisions."
            ),
            (
                "Both PPO target convergence and actual pre-trade drifted-path "
                "convergence are reported; target convergence is primary. "
                "Detailed outputs also store explicitly labeled full-L1 values."
            ),
            "",
            "## Early-window interpretation",
            "",
            (
                "Four-week and twelve-week Sharpe values use the first 20 and 60 "
                "trading days (four and twelve completed rebalance periods) and "
                "are descriptive diagnostics, not strong statistical evidence."
            ),
            "",
            "## Follow-up requirement",
            "",
            (
                "Before final walk-forward and final-test comparisons, the "
                "inverse-volatility baseline must use a full past-only lookback "
                "across the evaluation-window boundary. PR 12 does not alter the "
                "frozen baseline results."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _report_provenance(
    manifest: Mapping[str, Any],
) -> tuple[str, str, str]:
    artifact_directory = manifest.get("artifact_directory")
    source_hashes = manifest.get("source_hashes")
    if not isinstance(artifact_directory, str) or not artifact_directory:
        raise ValueError("manifest must include artifact_directory")
    if not isinstance(source_hashes, Mapping):
        raise TypeError("manifest must include source_hashes")
    model_hash = _source_sha256(source_hashes, "model")
    freeze_manifest_hash = _source_sha256(source_hashes, "freeze_manifest")
    return artifact_directory, model_hash, freeze_manifest_hash


def _source_sha256(
    source_hashes: Mapping[str, Any],
    source_name: str,
) -> str:
    source = source_hashes.get(source_name)
    if not isinstance(source, Mapping):
        raise TypeError(f"manifest source_hashes missing {source_name}")
    sha256 = source.get("sha256")
    if not isinstance(sha256, str) or len(sha256) != 64:
        raise ValueError(
            f"manifest source_hashes.{source_name}.sha256 must be a SHA-256 hash"
        )
    return sha256


def _guard_test_access(
    feature_store: PortfolioFeatureStore,
    configured_test_start_date: str | pd.Timestamp,
) -> None:
    test_start = pd.Timestamp(configured_test_start_date).normalize()
    evaluation_start = feature_store.date_at(0).normalize()
    if feature_store.split == "test" or evaluation_start >= test_start:
        raise ValueError(
            "initialization sensitivity must not access the final test period: "
            f"split={feature_store.split}, evaluation_start={evaluation_start.date()}, "
            f"configured_test_start={test_start.date()}"
        )


def _result_row(
    *,
    candidate: str,
    initializer: str,
    result: BacktestResult,
    rebalance_frequency_trading_days: int,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "candidate": candidate,
        "initializer": initializer,
        **result.metrics,
    }
    for weeks in (4, 12):
        days = weeks * rebalance_frequency_trading_days
        metrics = _early_window_metrics(
            result,
            trading_days=days,
            rebalance_periods=weeks,
        )
        for name, value in metrics.items():
            row[f"first_{weeks}_weeks_{name}"] = value
    return row


def _early_window_metrics(
    result: BacktestResult,
    *,
    trading_days: int,
    rebalance_periods: int,
) -> dict[str, float | None]:
    names = (
        "total_return",
        "sharpe_ratio",
        "max_drawdown",
        "average_weekly_turnover",
        "transaction_cost_drag",
    )
    if len(result.nav) < trading_days or len(result.costs) < rebalance_periods:
        return dict.fromkeys(names)
    metrics = calculate_performance_metrics(
        result.nav.iloc[:trading_days].copy(),
        result.costs.iloc[:rebalance_periods].copy(),
    )
    return {name: metrics[name] for name in names}


def _allocation_paths(
    candidate: str,
    initializer: str,
    result: BacktestResult,
) -> pd.DataFrame:
    paths = result.trades[
        ["date", "ticker", "pre_trade_weight", "target_weight"]
    ].copy()
    paths.insert(0, "initializer", initializer)
    paths.insert(0, "candidate", candidate)
    decision_dates = paths["date"].drop_duplicates().tolist()
    decision_step_by_date = {
        date: step for step, date in enumerate(decision_dates)
    }
    paths.insert(
        3,
        "decision_step",
        paths["date"].map(decision_step_by_date).astype(int),
    )
    return paths


def _convergence_outputs(
    allocations: pd.DataFrame,
    *,
    threshold: float,
    consecutive_decisions: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    detail_rows = []
    summary_rows = []
    for candidate, candidate_paths in allocations.groupby("candidate", sort=True):
        initializers = sorted(candidate_paths["initializer"].unique())
        for initializer_a, initializer_b in itertools.combinations(initializers, 2):
            path_a = candidate_paths[
                candidate_paths["initializer"] == initializer_a
            ]
            path_b = candidate_paths[
                candidate_paths["initializer"] == initializer_b
            ]
            merged = path_a.merge(
                path_b,
                on=["candidate", "date", "decision_step", "ticker"],
                suffixes=("_a", "_b"),
                validate="one_to_one",
            )
            grouped_rows = []
            for (date, step), decision in merged.groupby(
                ["date", "decision_step"],
                sort=True,
            ):
                target_full = float(
                    np.abs(
                        decision["target_weight_a"]
                        - decision["target_weight_b"]
                    ).sum()
                )
                pre_trade_full = float(
                    np.abs(
                        decision["pre_trade_weight_a"]
                        - decision["pre_trade_weight_b"]
                    ).sum()
                )
                row = {
                    "candidate": candidate,
                    "initializer_a": initializer_a,
                    "initializer_b": initializer_b,
                    "date": date,
                    "decision_step": int(step),
                    "target_full_l1": target_full,
                    "target_half_l1": 0.5 * target_full,
                    "pre_trade_full_l1": pre_trade_full,
                    "pre_trade_half_l1": 0.5 * pre_trade_full,
                }
                grouped_rows.append(row)
                detail_rows.append(row)
            pair = pd.DataFrame(grouped_rows)
            target_date, elapsed = _first_convergence(
                pair,
                column="target_half_l1",
                threshold=threshold,
                consecutive_decisions=consecutive_decisions,
            )
            pre_trade_date, _pre_trade_elapsed = _first_convergence(
                pair,
                column="pre_trade_half_l1",
                threshold=threshold,
                consecutive_decisions=consecutive_decisions,
            )
            summary_rows.append(
                {
                    "candidate": candidate,
                    "initializer_a": initializer_a,
                    "initializer_b": initializer_b,
                    "terminal_target_distance": _last_or_none(
                        pair["target_half_l1"]
                    ),
                    "mean_target_distance": _mean_or_none(
                        pair["target_half_l1"]
                    ),
                    "target_convergence_date": target_date,
                    "terminal_pre_trade_distance": _last_or_none(
                        pair["pre_trade_half_l1"]
                    ),
                    "mean_pre_trade_distance": _mean_or_none(
                        pair["pre_trade_half_l1"]
                    ),
                    "pre_trade_convergence_date": pre_trade_date,
                    "elapsed_decisions": elapsed,
                    "converged": target_date is not None,
                }
            )
    return pd.DataFrame(detail_rows), pd.DataFrame(summary_rows)


def _first_convergence(
    frame: pd.DataFrame,
    *,
    column: str,
    threshold: float,
    consecutive_decisions: int,
) -> tuple[str | None, int | None]:
    consecutive = 0
    for position, row in enumerate(frame.itertuples(index=False), start=1):
        distance = float(getattr(row, column))
        consecutive = consecutive + 1 if distance <= threshold else 0
        if consecutive >= consecutive_decisions:
            return pd.Timestamp(row.date).date().isoformat(), position
    return None, None


def _last_or_none(values: pd.Series) -> float | None:
    return float(values.iloc[-1]) if not values.empty else None


def _mean_or_none(values: pd.Series) -> float | None:
    return float(values.mean()) if not values.empty else None


def _format_metric(value: Any) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.6f}"
    return str(value)

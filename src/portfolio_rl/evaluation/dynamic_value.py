"""Dynamic-value attribution for frozen target-weight policies."""

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

EXPECTED_ALPHAS = [0.25, 0.5, 0.75, 1.0]
EXPECTED_SHIFTS = [4, 13, 26]
DYNAMIC_DIAGNOSTIC = "dynamic_live"


@dataclass(frozen=True)
class DynamicValueResult:
    """In-memory PR 15 diagnostic outputs."""

    diagnostic_results: pd.DataFrame
    target_sequences: pd.DataFrame
    active_return_decomposition: pd.DataFrame
    backtests: dict[tuple[str, str], BacktestResult]
    hurdle_backtest: BacktestResult


class RecordedTargetPolicy:
    """Replay a date-labeled target sequence and fail on misalignment."""

    def __init__(
        self,
        *,
        decision_dates: Sequence[pd.Timestamp],
        targets: np.ndarray,
    ) -> None:
        self._dates = tuple(pd.Timestamp(date).normalize() for date in decision_dates)
        self._targets = np.asarray(targets, dtype=np.float64)
        if self._targets.ndim != 2:
            raise ValueError("targets must be a two-dimensional array")
        if len(self._dates) != len(self._targets):
            raise ValueError("decision dates and targets must have equal length")
        _validate_target_matrix(self._targets)
        self._step = 0

    def reset(self) -> None:
        self._step = 0

    def target_weights(
        self,
        observation: np.ndarray,
        info: Mapping[str, Any],
    ) -> np.ndarray:
        del observation
        if self._step >= len(self._targets):
            raise IndexError("recorded target sequence is exhausted")
        observed_date = pd.Timestamp(info.get("date")).normalize()
        expected_date = self._dates[self._step]
        if observed_date != expected_date:
            raise ValueError(
                "recorded target date mismatch: "
                f"expected={expected_date.date()}, observed={observed_date.date()}"
            )
        target = self._targets[self._step].copy()
        self._step += 1
        return target


def evaluate_dynamic_value_checks(
    *,
    prior_feature_store: PortfolioFeatureStore,
    evaluation_feature_store: PortfolioFeatureStore,
    ensemble_policy_factory: Callable[[], WeightPolicy],
    hurdle_policy_factory: Callable[[], WeightPolicy],
    alphas: Sequence[float],
    circular_shifts: Sequence[int],
    configured_test_start_date: str | pd.Timestamp,
    rebalance_frequency_trading_days: int = 5,
    transaction_cost_bps: float = 10.0,
) -> DynamicValueResult:
    """Evaluate static and mistimed controls without selecting an alpha."""
    _validate_contract(
        prior_feature_store=prior_feature_store,
        evaluation_feature_store=evaluation_feature_store,
        alphas=alphas,
        circular_shifts=circular_shifts,
        configured_test_start_date=configured_test_start_date,
    )
    candidates: list[tuple[str, float | None]] = [
        ("ensemble_unmodified", None),
        *((f"ensemble_alpha_{alpha:.2f}", float(alpha)) for alpha in alphas),
    ]
    hurdle = run_weight_policy_backtest(
        feature_store=evaluation_feature_store,
        policy=hurdle_policy_factory(),
        strategy="equal_weight_weekly",
        rebalance_frequency_trading_days=rebalance_frequency_trading_days,
        transaction_cost_bps=transaction_cost_bps,
    )
    hurdle_gross = run_weight_policy_backtest(
        feature_store=evaluation_feature_store,
        policy=hurdle_policy_factory(),
        strategy="equal_weight_weekly_gross",
        rebalance_frequency_trading_days=rebalance_frequency_trading_days,
        transaction_cost_bps=0.0,
    )

    result_rows: list[dict[str, Any]] = []
    sequence_frames: list[pd.DataFrame] = []
    backtests: dict[tuple[str, str], BacktestResult] = {}
    gross_metrics: dict[tuple[str, str], Mapping[str, float | None]] = {}
    dynamic_targets: dict[str, np.ndarray] = {}
    dynamic_raw_targets: dict[str, np.ndarray] = {}
    dynamic_dates: dict[str, list[pd.Timestamp]] = {}

    for candidate, alpha in candidates:
        prior_result, _ = _run_live_candidate(
            feature_store=prior_feature_store,
            ensemble_policy_factory=ensemble_policy_factory,
            candidate=candidate,
            alpha=alpha,
            rebalance_frequency_trading_days=rebalance_frequency_trading_days,
            transaction_cost_bps=transaction_cost_bps,
        )
        live_result, live_raw = _run_live_candidate(
            feature_store=evaluation_feature_store,
            ensemble_policy_factory=ensemble_policy_factory,
            candidate=candidate,
            alpha=alpha,
            rebalance_frequency_trading_days=rebalance_frequency_trading_days,
            transaction_cost_bps=transaction_cost_bps,
        )
        dates, live_targets = _target_matrix(
            live_result,
            evaluation_feature_store.asset_order,
        )
        prior_dates, prior_targets = _target_matrix(
            prior_result,
            prior_feature_store.asset_order,
        )
        del prior_dates
        dynamic_targets[candidate] = live_targets
        dynamic_raw_targets[candidate] = live_raw
        dynamic_dates[candidate] = dates
        backtests[(candidate, DYNAMIC_DIAGNOSTIC)] = live_result

        control_sequences = _build_control_sequences(
            live_targets=live_targets,
            prior_targets=prior_targets,
            n_assets=evaluation_feature_store.n_assets,
            circular_shifts=circular_shifts,
        )
        for diagnostic, sequence_spec in control_sequences.items():
            targets = sequence_spec["targets"]
            if diagnostic == DYNAMIC_DIAGNOSTIC:
                net_result = live_result
            else:
                net_result = _run_recorded_backtest(
                    feature_store=evaluation_feature_store,
                    dates=dates,
                    targets=targets,
                    strategy=f"{candidate}_{diagnostic}",
                    rebalance_frequency_trading_days=(
                        rebalance_frequency_trading_days
                    ),
                    transaction_cost_bps=transaction_cost_bps,
                )
                backtests[(candidate, diagnostic)] = net_result
            gross_result = _run_recorded_backtest(
                feature_store=evaluation_feature_store,
                dates=dates,
                targets=targets,
                strategy=f"{candidate}_{diagnostic}_gross",
                rebalance_frequency_trading_days=(
                    rebalance_frequency_trading_days
                ),
                transaction_cost_bps=0.0,
            )
            gross_metrics[(candidate, diagnostic)] = gross_result.metrics
            result_rows.append(
                _metric_row(
                    candidate=candidate,
                    alpha=alpha,
                    diagnostic=diagnostic,
                    deployability=str(sequence_spec["deployability"]),
                    net_metrics=net_result.metrics,
                    gross_metrics=gross_result.metrics,
                    hurdle_metrics=hurdle.metrics,
                    hurdle_gross_metrics=hurdle_gross.metrics,
                )
            )
            sequence_frames.append(
                _sequence_frame(
                    candidate=candidate,
                    alpha=alpha,
                    diagnostic=diagnostic,
                    deployability=str(sequence_spec["deployability"]),
                    dates=dates,
                    targets=targets,
                    asset_order=evaluation_feature_store.asset_order,
                    source_steps=sequence_spec["source_steps"],
                    source_window=str(sequence_spec["source_window"]),
                    live_raw_targets=live_raw,
                    live_executed_targets=live_targets,
                )
            )

    _reconcile_unmodified_and_alpha_one(
        backtests=backtests,
        dynamic_targets=dynamic_targets,
        dynamic_raw_targets=dynamic_raw_targets,
    )
    results = pd.DataFrame(result_rows)
    decomposition = _build_decomposition(results)
    return DynamicValueResult(
        diagnostic_results=results,
        target_sequences=pd.concat(sequence_frames, ignore_index=True),
        active_return_decomposition=decomposition,
        backtests=backtests,
        hurdle_backtest=hurdle,
    )


def write_dynamic_value_artifacts(
    *,
    result: DynamicValueResult,
    output_dir: str | Path,
    manifest: Mapping[str, Any],
) -> dict[str, Path]:
    """Atomically write an immutable PR 15 research bundle."""
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(f"dynamic-value output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    try:
        outputs = {
            "manifest": temporary / "dynamic_value_manifest.json",
            "results": temporary / "diagnostic_results.csv",
            "targets": temporary / "target_sequences.parquet",
            "decomposition": temporary / "active_return_decomposition.csv",
            "report": temporary / "dynamic_value_report.md",
        }
        result.diagnostic_results.to_csv(outputs["results"], index=False)
        result.target_sequences.to_parquet(outputs["targets"], index=False)
        result.active_return_decomposition.to_csv(
            outputs["decomposition"],
            index=False,
        )
        write_backtest_artifacts(
            result.hurdle_backtest,
            temporary / "backtest" / "equal_weight_weekly",
        )
        for (candidate, diagnostic), backtest in result.backtests.items():
            write_backtest_artifacts(
                backtest,
                temporary / "backtest" / candidate / diagnostic,
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
            format_dynamic_value_report(result=result, manifest=full_manifest),
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


def format_dynamic_value_report(
    *,
    result: DynamicValueResult,
    manifest: Mapping[str, Any],
) -> str:
    """Format the descriptive, non-selective PR 15 report."""
    lines = [
        "# Dynamic-Value Diagnostics",
        "",
        "## Governance and interpretation",
        "",
        "- The frozen five-seed ensemble is the only model under study.",
        "- 2024 is consumed development/selection data, not independent evidence.",
        (
            "- Calendar 2023 is a past-only in-sample calibration window. The "
            "ensemble itself was selected using 2024, so this is not a claim "
            "that the portfolio was actually deployable at the start of 2024."
        ),
        (
            "- Oracle static and circular-shift controls are non-deployable. "
            "Lagged and shifted controls replay executed targets for attribution."
        ),
        "- No alpha is selected or eliminated; all four advance to walk-forward.",
        "- Final-test access: **none**.",
        "",
        "## Method",
        "",
        (
            "Each dynamic candidate is rerun with observations built from its "
            "own live drifted portfolio. Only the controls replay its executed "
            "target path. Ex-ante static weights average 2023 targets; oracle "
            "weights average 2024 targets; lag one begins equal weight; positive "
            "circular shifts use target `(t - shift) mod N`."
        ),
        "",
    ]
    for candidate, frame in result.diagnostic_results.groupby(
        "candidate",
        sort=False,
    ):
        lines.extend([f"## {candidate}", "", *_report_table(frame), ""])
    lines.extend(
        [
            "## Conclusion scope",
            "",
            (
                "These diagnostics describe whether static or mistimed executed "
                "allocations explain the frozen ensemble's consumed 2024 result. "
                "They do not establish general PPO timing value and do not choose "
                "an execution alpha."
            ),
            "",
            f"PR 14 manifest SHA-256: `{manifest['pr14_manifest_sha256']}`",
            "",
        ]
    )
    return "\n".join(lines)


def _run_live_candidate(
    *,
    feature_store: PortfolioFeatureStore,
    ensemble_policy_factory: Callable[[], WeightPolicy],
    candidate: str,
    alpha: float | None,
    rebalance_frequency_trading_days: int,
    transaction_cost_bps: float,
) -> tuple[BacktestResult, np.ndarray]:
    base = ensemble_policy_factory()
    policy: WeightPolicy
    overlay: PartialRebalancePolicy | None = None
    if alpha is None:
        policy = base
    else:
        overlay = PartialRebalancePolicy(base_policy=base, alpha=alpha)
        policy = overlay
    result = run_weight_policy_backtest(
        feature_store=feature_store,
        policy=policy,
        strategy=candidate,
        rebalance_frequency_trading_days=rebalance_frequency_trading_days,
        transaction_cost_bps=transaction_cost_bps,
    )
    _, targets = _target_matrix(result, feature_store.asset_order)
    if overlay is None:
        raw_targets = targets.copy()
    else:
        raw_targets = np.asarray(
            [record.raw_policy_target for record in overlay.records],
            dtype=np.float64,
        )
    if raw_targets.shape != targets.shape:
        raise ValueError(f"raw target audit shape mismatch: {candidate}")
    return result, raw_targets


def _run_recorded_backtest(
    *,
    feature_store: PortfolioFeatureStore,
    dates: list[pd.Timestamp],
    targets: np.ndarray,
    strategy: str,
    rebalance_frequency_trading_days: int,
    transaction_cost_bps: float,
) -> BacktestResult:
    return run_weight_policy_backtest(
        feature_store=feature_store,
        policy=RecordedTargetPolicy(decision_dates=dates, targets=targets),
        strategy=strategy,
        rebalance_frequency_trading_days=rebalance_frequency_trading_days,
        transaction_cost_bps=transaction_cost_bps,
    )


def _build_control_sequences(
    *,
    live_targets: np.ndarray,
    prior_targets: np.ndarray,
    n_assets: int,
    circular_shifts: Sequence[int],
) -> dict[str, dict[str, Any]]:
    _validate_target_matrix(live_targets)
    _validate_target_matrix(prior_targets)
    n_decisions = len(live_targets)
    if any(shift >= n_decisions for shift in circular_shifts):
        raise ValueError("circular shifts must be smaller than decision count")
    equal_weight = np.full(n_assets, 1.0 / n_assets, dtype=np.float64)
    lagged = np.vstack([equal_weight, live_targets[:-1]])
    controls: dict[str, dict[str, Any]] = {
        DYNAMIC_DIAGNOSTIC: {
            "targets": live_targets,
            "deployability": "live_feedback_candidate",
            "source_steps": np.arange(n_decisions),
            "source_window": "evaluation_2024",
        },
        "ex_ante_static_2023": {
            "targets": np.tile(prior_targets.mean(axis=0), (n_decisions, 1)),
            "deployability": "past_only_static_control",
            "source_steps": np.full(n_decisions, -1),
            "source_window": "prior_2023_average",
        },
        "oracle_static_2024": {
            "targets": np.tile(live_targets.mean(axis=0), (n_decisions, 1)),
            "deployability": "non_deployable_future_information",
            "source_steps": np.full(n_decisions, -1),
            "source_window": "evaluation_2024_average",
        },
        "lag_1_decision": {
            "targets": lagged,
            "deployability": "sequence_attribution_only",
            "source_steps": np.concatenate(([-1], np.arange(n_decisions - 1))),
            "source_window": "evaluation_2024",
        },
    }
    for shift in circular_shifts:
        controls[f"circular_shift_{shift}"] = {
            "targets": np.roll(live_targets, shift=shift, axis=0),
            "deployability": "non_deployable_sequence_attribution",
            "source_steps": (np.arange(n_decisions) - shift) % n_decisions,
            "source_window": "evaluation_2024",
        }
    for spec in controls.values():
        _validate_target_matrix(np.asarray(spec["targets"]))
    return controls


def _target_matrix(
    result: BacktestResult,
    asset_order: list[str],
) -> tuple[list[pd.Timestamp], np.ndarray]:
    frame = result.weights_target.copy()
    dates = sorted(pd.Timestamp(date) for date in frame["date"].unique())
    pivot = frame.pivot(index="date", columns="ticker", values="target_weight")
    pivot.index = pd.to_datetime(pivot.index)
    pivot = pivot.reindex(index=dates, columns=asset_order)
    if pivot.isna().any().any():
        raise ValueError("target path has missing date/ticker values")
    targets = pivot.to_numpy(dtype=np.float64)
    _validate_target_matrix(targets)
    return dates, targets


def _metric_row(
    *,
    candidate: str,
    alpha: float | None,
    diagnostic: str,
    deployability: str,
    net_metrics: Mapping[str, float | None],
    gross_metrics: Mapping[str, float | None],
    hurdle_metrics: Mapping[str, float | None],
    hurdle_gross_metrics: Mapping[str, float | None],
) -> dict[str, Any]:
    net_return = _metric(net_metrics, "total_return")
    gross_return = _metric(gross_metrics, "total_return")
    hurdle_return = _metric(hurdle_metrics, "total_return")
    hurdle_gross = _metric(hurdle_gross_metrics, "total_return")
    return {
        "candidate": candidate,
        "alpha": alpha,
        "diagnostic": diagnostic,
        "deployability": deployability,
        **dict(net_metrics),
        "gross_total_return": gross_return,
        "cost_return_impact": net_return - gross_return,
        "active_return_vs_equal_weight": net_return - hurdle_return,
        "gross_active_return_vs_equal_weight": gross_return - hurdle_gross,
        "active_sharpe_vs_equal_weight": (
            _metric(net_metrics, "sharpe_ratio")
            - _metric(hurdle_metrics, "sharpe_ratio")
        ),
        "drawdown_difference_vs_equal_weight": (
            _metric(net_metrics, "max_drawdown")
            - _metric(hurdle_metrics, "max_drawdown")
        ),
    }


def _build_decomposition(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for candidate, frame in results.groupby("candidate", sort=False):
        dynamic = frame[frame["diagnostic"] == DYNAMIC_DIAGNOSTIC]
        if len(dynamic) != 1:
            raise ValueError(f"missing unique dynamic row: {candidate}")
        reference = dynamic.iloc[0]
        for control in frame.itertuples(index=False):
            if control.diagnostic == DYNAMIC_DIAGNOSTIC:
                continue
            net_advantage = float(reference.total_return - control.total_return)
            gross_advantage = float(
                reference.gross_total_return - control.gross_total_return
            )
            rows.append(
                {
                    "candidate": candidate,
                    "alpha": control.alpha,
                    "control": control.diagnostic,
                    "dynamic_net_return": reference.total_return,
                    "control_net_return": control.total_return,
                    "net_dynamic_advantage": net_advantage,
                    "dynamic_gross_return": reference.gross_total_return,
                    "control_gross_return": control.gross_total_return,
                    "gross_dynamic_advantage": gross_advantage,
                    "relative_cost_contribution": (
                        net_advantage - gross_advantage
                    ),
                    "dynamic_sharpe": reference.sharpe_ratio,
                    "control_sharpe": control.sharpe_ratio,
                    "dynamic_max_drawdown": reference.max_drawdown,
                    "control_max_drawdown": control.max_drawdown,
                    "dynamic_average_weekly_turnover": (
                        reference.average_weekly_turnover
                    ),
                    "control_average_weekly_turnover": (
                        control.average_weekly_turnover
                    ),
                    "dynamic_transaction_cost_drag": (
                        reference.transaction_cost_drag
                    ),
                    "control_transaction_cost_drag": (
                        control.transaction_cost_drag
                    ),
                }
            )
    return pd.DataFrame(rows)


def _sequence_frame(
    *,
    candidate: str,
    alpha: float | None,
    diagnostic: str,
    deployability: str,
    dates: list[pd.Timestamp],
    targets: np.ndarray,
    asset_order: list[str],
    source_steps: Any,
    source_window: str,
    live_raw_targets: np.ndarray,
    live_executed_targets: np.ndarray,
) -> pd.DataFrame:
    rows = []
    source_indices = np.asarray(source_steps, dtype=np.int64)
    for step, (date, target) in enumerate(zip(dates, targets, strict=True)):
        source_step = int(source_indices[step])
        for asset_index, ticker in enumerate(asset_order):
            rows.append(
                {
                    "candidate": candidate,
                    "alpha": alpha,
                    "diagnostic": diagnostic,
                    "deployability": deployability,
                    "target_basis": "executed_target",
                    "date": date,
                    "decision_step": step,
                    "ticker": ticker,
                    "target_weight": float(target[asset_index]),
                    "source_window": source_window,
                    "source_decision_step": (
                        source_step if source_step >= 0 else None
                    ),
                    "source_date": (
                        dates[source_step] if source_step >= 0 else None
                    ),
                    "dynamic_raw_target": float(
                        live_raw_targets[step, asset_index]
                    ),
                    "dynamic_executed_target": float(
                        live_executed_targets[step, asset_index]
                    ),
                }
            )
    return pd.DataFrame(rows)


def _reconcile_unmodified_and_alpha_one(
    *,
    backtests: Mapping[tuple[str, str], BacktestResult],
    dynamic_targets: Mapping[str, np.ndarray],
    dynamic_raw_targets: Mapping[str, np.ndarray],
) -> None:
    direct = "ensemble_unmodified"
    alpha_one = "ensemble_alpha_1.00"
    if not np.array_equal(dynamic_targets[direct], dynamic_targets[alpha_one]):
        raise ValueError("alpha 1.00 targets do not match unmodified ensemble")
    if not np.array_equal(
        dynamic_raw_targets[direct],
        dynamic_raw_targets[alpha_one],
    ):
        raise ValueError("alpha 1.00 raw targets do not match unmodified ensemble")
    direct_result = backtests[(direct, DYNAMIC_DIAGNOSTIC)]
    alpha_result = backtests[(alpha_one, DYNAMIC_DIAGNOSTIC)]
    for frame_name in ("nav", "trades", "costs"):
        direct_frame = getattr(direct_result, frame_name).drop(columns="strategy")
        alpha_frame = getattr(alpha_result, frame_name).drop(columns="strategy")
        if not direct_frame.equals(alpha_frame):
            raise ValueError(
                f"alpha 1.00 {frame_name} does not match unmodified ensemble"
            )
    if direct_result.metrics != alpha_result.metrics:
        raise ValueError("alpha 1.00 metrics do not match unmodified ensemble")


def _validate_contract(
    *,
    prior_feature_store: PortfolioFeatureStore,
    evaluation_feature_store: PortfolioFeatureStore,
    alphas: Sequence[float],
    circular_shifts: Sequence[int],
    configured_test_start_date: str | pd.Timestamp,
) -> None:
    if list(alphas) != EXPECTED_ALPHAS:
        raise ValueError(f"alphas must equal {EXPECTED_ALPHAS}")
    if list(circular_shifts) != EXPECTED_SHIFTS:
        raise ValueError(f"circular shifts must equal {EXPECTED_SHIFTS}")
    test_start = pd.Timestamp(configured_test_start_date).normalize()
    for store in (prior_feature_store, evaluation_feature_store):
        if store.split == "test" or store.date_at(0).normalize() >= test_start:
            raise ValueError("dynamic-value study must not access final test data")
    prior_end = prior_feature_store.date_at(prior_feature_store.n_rows - 1)
    evaluation_start = evaluation_feature_store.date_at(0)
    if prior_end >= evaluation_start:
        raise ValueError("prior window must end before evaluation starts")
    if prior_feature_store.asset_order != evaluation_feature_store.asset_order:
        raise ValueError("prior and evaluation asset order must match")


def _validate_target_matrix(targets: np.ndarray) -> None:
    if targets.ndim != 2 or targets.shape[0] == 0 or targets.shape[1] == 0:
        raise ValueError("target matrix must be non-empty and two-dimensional")
    if not np.isfinite(targets).all():
        raise ValueError("target weights must be finite")
    if (targets < 0.0).any():
        raise ValueError("target weights must be nonnegative")
    if not np.allclose(targets.sum(axis=1), 1.0):
        raise ValueError("target weights must sum to one")


def _metric(metrics: Mapping[str, float | None], name: str) -> float:
    value = metrics.get(name)
    if value is None or not np.isfinite(value):
        raise ValueError(f"metric must be finite: {name}")
    return float(value)


def _report_table(frame: pd.DataFrame) -> list[str]:
    lines = [
        (
            "| Diagnostic | Return | Active return | Gross return | Sharpe | "
            "Max drawdown | Avg turnover | Cost drag |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in frame.itertuples(index=False):
        lines.append(
            f"| {row.diagnostic} | {row.total_return:.4%} | "
            f"{row.active_return_vs_equal_weight:.4%} | "
            f"{row.gross_total_return:.4%} | {row.sharpe_ratio:.3f} | "
            f"{row.max_drawdown:.4%} | "
            f"{row.average_weekly_turnover:.4%} | "
            f"{row.transaction_cost_drag:.4%} |"
        )
    return lines

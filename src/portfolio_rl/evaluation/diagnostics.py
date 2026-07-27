"""Campaign-level policy behavior diagnostics for selected PPO checkpoints."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from math import isclose
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from portfolio_rl.config.loader import (
    load_phase3_evaluation_config,
    load_universe_config,
)
from portfolio_rl.config.schemas import PolicyBehaviorConfig
from portfolio_rl.data.dataset import load_portfolio_dataset
from portfolio_rl.evaluation import robustness
from portfolio_rl.evaluation.backtest import BacktestResult, run_weight_policy_backtest
from portfolio_rl.policies.sb3_policy import load_sb3_weight_policy

GLOBAL_FEATURES_PATH = Path("data/processed/global_features_normalized_daily.parquet")
RECONCILIATION_METRICS = {
    "total_return": "selection_validation_total_return",
    "sharpe_ratio": "selection_validation_sharpe_ratio",
    "max_drawdown": "selection_validation_max_drawdown",
    "average_weekly_turnover": ("selection_validation_average_weekly_turnover"),
    "transaction_cost_drag": "selection_validation_transaction_cost_drag",
}
OUTPUT_FILENAMES = {
    "summary": "allocation_summary.json",
    "allocations": "allocation_by_regime.parquet",
    "nav": "nav_by_regime.parquet",
    "concentration": "concentration_metrics.parquet",
    "turnover": "turnover_distribution.parquet",
    "monthly_returns": "monthly_returns.parquet",
    "drawdowns": "drawdown_periods.parquet",
    "report": "diagnostics_report.md",
}


@dataclass(frozen=True)
class PolicyBehaviorResult:
    """Written policy-behavior artifacts and their in-memory values."""

    outputs: dict[str, Path]
    allocation_by_regime: pd.DataFrame
    nav_by_regime: pd.DataFrame
    concentration_metrics: pd.DataFrame
    turnover_distribution: pd.DataFrame
    monthly_returns: pd.DataFrame
    drawdown_periods: pd.DataFrame
    summary: dict[str, Any]


def run_policy_behavior_diagnostics(
    *,
    selected_configuration_path: str | Path,
    registry_path: str | Path,
    evaluation_config_path: str | Path,
    universe_config_path: str | Path,
    output_dir: str | Path,
    root: str | Path = ".",
) -> PolicyBehaviorResult:
    """Analyze selected checkpoints over configured non-test regime windows."""
    root_path = Path(root)
    selected_path = _resolve_path(root_path, selected_configuration_path)
    resolved_registry_path = _resolve_path(root_path, registry_path)
    resolved_evaluation_path = _resolve_path(root_path, evaluation_config_path)
    resolved_universe_path = _resolve_path(root_path, universe_config_path)
    destination = _resolve_path(root_path, output_dir)

    selected = _read_json(selected_path)
    registry = _read_registry(resolved_registry_path)
    evaluation_config = load_phase3_evaluation_config(resolved_evaluation_path)
    universe_config = load_universe_config(resolved_universe_path)
    robustness._validate_selected_configuration(selected)
    selected_runs = robustness._resolve_selected_runs(
        selected=selected,
        registry=registry,
        root=root_path,
    )
    selected_rows = _selected_registry_rows(
        selected=selected,
        registry=registry,
    )

    dataset = load_portfolio_dataset(root_path)
    regime_windows, skipped_regimes = robustness._resolve_regime_windows(
        dataset=dataset,
        windows=evaluation_config.robustness.regime_windows,
        rebalance_frequency_trading_days=robustness._rebalance_frequency(selected_runs),
    )
    global_features_path = root_path / GLOBAL_FEATURES_PATH
    global_features = _load_global_features(global_features_path)
    asset_classes = {
        asset.ticker: asset.asset_class for asset in universe_config.assets
    }
    if set(dataset.asset_order) != set(asset_classes):
        raise ValueError("dataset assets do not match configured universe")

    allocations = []
    nav_paths = []
    concentration = []
    turnover = []
    monthly_returns = []
    drawdowns = []
    reconciliation = []
    for run in selected_runs:
        policy = load_sb3_weight_policy(
            run["model_path"],
            action_temperature=run["action_temperature"],
        )
        registry_row = selected_rows[int(run["seed"])]
        for regime in regime_windows:
            backtest = run_weight_policy_backtest(
                feature_store=regime["feature_store"],
                policy=policy,
                strategy=f"ppo_seed_{run['seed']}_{regime['name']}",
                rebalance_frequency_trading_days=run[
                    "rebalance_frequency_trading_days"
                ],
                transaction_cost_bps=run["transaction_cost_bps"],
            )
            metadata = _run_metadata(run, regime)
            nav_paths.append(_daily_nav_frame(backtest.nav, metadata=metadata))
            allocation = _allocation_frame(
                backtest=backtest,
                metadata=metadata,
                global_features=global_features,
                high_volatility_quantile=(
                    evaluation_config.policy_behavior.high_volatility_quantile
                ),
            )
            allocations.append(allocation)
            concentration.append(
                calculate_concentration_metrics(
                    allocation,
                    asset_classes=asset_classes,
                    active_weight_threshold=(
                        evaluation_config.policy_behavior.active_weight_threshold
                    ),
                )
            )
            turnover.append(
                calculate_turnover_distribution(
                    backtest=backtest,
                    allocation=allocation,
                    metadata=metadata,
                    spike_quantile=(
                        evaluation_config.policy_behavior.turnover_spike_quantile
                    ),
                )
            )
            monthly_returns.append(
                calculate_monthly_returns(backtest.nav, metadata=metadata)
            )
            drawdowns.append(
                calculate_drawdown_periods(backtest.nav, metadata=metadata)
            )
            if regime["split"] == "validation" and regime["full_split_window"]:
                reconciliation.append(
                    _reconcile_validation_metrics(
                        backtest.metrics,
                        registry_row=registry_row,
                        run_id=run["run_id"],
                    )
                )

    if len(reconciliation) != len(selected_runs):
        raise ValueError(
            "diagnostics require one full validation regime for every seed"
        )

    allocation_frame = _concat(allocations)
    nav_frame = _concat(nav_paths)
    concentration_frame = _concat(concentration)
    turnover_frame = _concat(turnover)
    monthly_frame = _concat(monthly_returns)
    drawdown_frame = _concat(drawdowns)
    summary = build_policy_behavior_summary(
        selected=selected,
        selected_runs=selected_runs,
        selected_models=[
            {
                "run_id": str(run["run_id"]),
                "seed": int(run["seed"]),
                "selection_checkpoint": str(run["selection_checkpoint"]),
                **_source(Path(run["model_path"]), root_path),
            }
            for run in selected_runs
        ],
        allocations=allocation_frame,
        concentration=concentration_frame,
        turnover=turnover_frame,
        monthly_returns=monthly_frame,
        config=evaluation_config.policy_behavior,
        max_median_turnover=(evaluation_config.selection.max_median_weekly_turnover),
        reconciliation=reconciliation,
        regime_windows=regime_windows,
        skipped_regimes=skipped_regimes,
        sources={
            "selected_configuration": _source(selected_path, root_path),
            "registry": _source(resolved_registry_path, root_path),
            "evaluation_config": _source(
                resolved_evaluation_path,
                root_path,
            ),
            "universe_config": _source(resolved_universe_path, root_path),
            "global_features": _source(global_features_path, root_path),
        },
    )

    destination.mkdir(parents=True, exist_ok=True)
    outputs = {
        key: destination / filename for key, filename in OUTPUT_FILENAMES.items()
    }
    allocation_frame.to_parquet(outputs["allocations"], index=False)
    nav_frame.to_parquet(outputs["nav"], index=False)
    concentration_frame.to_parquet(outputs["concentration"], index=False)
    turnover_frame.to_parquet(outputs["turnover"], index=False)
    monthly_frame.to_parquet(outputs["monthly_returns"], index=False)
    drawdown_frame.to_parquet(outputs["drawdowns"], index=False)
    outputs["summary"].write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    outputs["report"].write_text(
        format_policy_behavior_report(summary),
        encoding="utf-8",
    )
    return PolicyBehaviorResult(
        outputs=outputs,
        allocation_by_regime=allocation_frame,
        nav_by_regime=nav_frame,
        concentration_metrics=concentration_frame,
        turnover_distribution=turnover_frame,
        monthly_returns=monthly_frame,
        drawdown_periods=drawdown_frame,
        summary=summary,
    )


def calculate_concentration_metrics(
    allocation: pd.DataFrame,
    *,
    asset_classes: dict[str, str],
    active_weight_threshold: float,
) -> pd.DataFrame:
    """Calculate per-date allocation concentration and grouped exposures."""
    _require_columns(
        allocation,
        [
            "run_id",
            "seed",
            "regime_name",
            "split",
            "in_sample",
            "date",
            "ticker",
            "target_weight",
            "high_volatility",
        ],
        "allocation",
    )
    unknown = sorted(set(allocation["ticker"]) - set(asset_classes))
    if unknown:
        raise ValueError(f"allocation contains unknown tickers: {unknown}")
    if not allocation["target_weight"].map(np.isfinite).all():
        raise ValueError("target weights must be finite")
    if (allocation["target_weight"] < 0.0).any():
        raise ValueError("target weights must be nonnegative")

    group_columns = [
        "run_id",
        "seed",
        "regime_name",
        "split",
        "in_sample",
        "date",
        "high_volatility",
    ]
    rows = []
    for keys, group in allocation.groupby(group_columns, sort=True):
        weights = group["target_weight"].astype(float)
        if not isclose(float(weights.sum()), 1.0, abs_tol=1e-6):
            raise ValueError("target weights must sum to one by decision date")
        maximum_index = weights.idxmax()
        hhi = float(np.square(weights).sum())
        equity_like = group["ticker"].map(
            lambda ticker: _is_equity_like(asset_classes[str(ticker)])
        )
        shy = group["ticker"] == "SHY"
        row = dict(zip(group_columns, keys, strict=True))
        row.update(
            {
                "max_weight": float(weights.max()),
                "max_ticker": str(group.loc[maximum_index, "ticker"]),
                "hhi": hhi,
                "effective_asset_count": float(1.0 / hhi),
                "active_asset_count": int((weights > active_weight_threshold).sum()),
                "shy_weight": float(weights[shy].sum()),
                "equity_like_weight": float(weights[equity_like].sum()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["regime_name", "seed", "date"],
        kind="mergesort",
        ignore_index=True,
    )


def calculate_turnover_distribution(
    *,
    backtest: BacktestResult,
    allocation: pd.DataFrame,
    metadata: dict[str, Any],
    spike_quantile: float,
) -> pd.DataFrame:
    """Add target-change and spike diagnostics to one-way turnover rows."""
    _require_columns(
        backtest.costs,
        ["date", "turnover", "transaction_cost_fraction"],
        "costs",
    )
    weights = allocation.pivot(
        index="date",
        columns="ticker",
        values="target_weight",
    ).sort_index()
    target_change = 0.5 * weights.diff().abs().sum(axis=1)
    target_change.iloc[0] = 0.0
    result = backtest.costs[["date", "turnover", "transaction_cost_fraction"]].copy()
    result["date"] = pd.to_datetime(result["date"])
    result = result.merge(
        target_change.rename("target_change_turnover").reset_index(),
        on="date",
        how="left",
        validate="one_to_one",
    )
    threshold = float(result["turnover"].quantile(spike_quantile))
    result["turnover_spike_threshold"] = threshold
    result["is_turnover_spike"] = result["turnover"] >= threshold
    high_volatility = allocation[["date", "high_volatility"]].drop_duplicates()
    result = result.merge(
        high_volatility,
        on="date",
        how="left",
        validate="one_to_one",
    )
    return _attach_metadata(result, metadata)


def calculate_monthly_returns(
    nav: pd.DataFrame,
    *,
    metadata: dict[str, Any],
) -> pd.DataFrame:
    """Compound daily net returns into calendar-month returns."""
    _require_columns(nav, ["date", "daily_return"], "nav")
    if nav.empty:
        return pd.DataFrame()
    working = nav[["date", "daily_return"]].copy()
    working["date"] = pd.to_datetime(working["date"])
    working["month"] = working["date"].dt.to_period("M").astype(str)
    monthly = (
        working.groupby("month", sort=True)["daily_return"]
        .apply(lambda values: float((1.0 + values.astype(float)).prod() - 1.0))
        .rename("monthly_return")
        .reset_index()
    )
    best_index = int(monthly["monthly_return"].idxmax())
    monthly["is_best_month"] = False
    monthly.loc[best_index, "is_best_month"] = True
    positive_logs = np.log1p(
        monthly.loc[monthly["monthly_return"] > 0.0, "monthly_return"]
    )
    positive_total = float(positive_logs.sum())
    best_log = float(
        np.log1p(max(float(monthly.loc[best_index, "monthly_return"]), 0.0))
    )
    monthly["best_month_positive_return_share"] = (
        best_log / positive_total if positive_total > 0.0 else 0.0
    )
    without_best = monthly.loc[~monthly["is_best_month"], "monthly_return"]
    monthly["total_return_excluding_best_month"] = float(
        (1.0 + without_best).prod() - 1.0
    )
    return _attach_metadata(monthly, metadata)


def _daily_nav_frame(
    nav: pd.DataFrame,
    *,
    metadata: dict[str, Any],
) -> pd.DataFrame:
    """Attach selected-checkpoint metadata to a validated daily NAV path."""
    columns = ["date", "strategy", "nav", "daily_return", "drawdown"]
    _require_columns(nav, columns, "nav")
    if nav.empty:
        raise ValueError("nav must not be empty")
    result = nav[columns].copy()
    result["date"] = pd.to_datetime(result["date"])
    if result["date"].duplicated().any():
        raise ValueError("nav contains duplicate dates")
    if not result["date"].is_monotonic_increasing:
        raise ValueError("nav dates must be monotonically increasing")
    numeric_columns = ["nav", "daily_return", "drawdown"]
    if not result[numeric_columns].map(np.isfinite).all().all():
        raise ValueError("nav values must be finite")
    if (result["nav"] <= 0.0).any():
        raise ValueError("nav values must be positive")
    return _attach_metadata(result, metadata)


def calculate_drawdown_periods(
    nav: pd.DataFrame,
    *,
    metadata: dict[str, Any],
) -> pd.DataFrame:
    """Extract contiguous underwater periods from a NAV path."""
    _require_columns(nav, ["date", "drawdown"], "nav")
    if nav.empty:
        return pd.DataFrame()
    ordered = nav[["date", "drawdown"]].sort_values("date").reset_index(drop=True)
    rows = []
    start_index: int | None = None
    for index, row in ordered.iterrows():
        underwater = float(row["drawdown"]) < -1e-12
        if underwater and start_index is None:
            start_index = int(index)
        if not underwater and start_index is not None:
            rows.append(
                _drawdown_row(
                    ordered,
                    start_index=start_index,
                    end_index=int(index),
                    recovered=True,
                )
            )
            start_index = None
    if start_index is not None:
        rows.append(
            _drawdown_row(
                ordered,
                start_index=start_index,
                end_index=len(ordered) - 1,
                recovered=False,
            )
        )
    return _attach_metadata(pd.DataFrame(rows), metadata)


def build_policy_behavior_summary(
    *,
    selected: dict[str, Any],
    selected_runs: list[dict[str, Any]],
    selected_models: list[dict[str, Any]],
    allocations: pd.DataFrame,
    concentration: pd.DataFrame,
    turnover: pd.DataFrame,
    monthly_returns: pd.DataFrame,
    config: PolicyBehaviorConfig,
    max_median_turnover: float,
    reconciliation: list[dict[str, Any]],
    regime_windows: list[dict[str, Any]],
    skipped_regimes: list[dict[str, str]],
    sources: dict[str, dict[str, str]],
) -> dict[str, Any]:
    """Aggregate behavior metrics and informational warning flags."""
    group_columns = ["run_id", "seed", "regime_name", "split", "in_sample"]
    groups = []
    for keys, group in concentration.groupby(group_columns, sort=True):
        raw_metadata = dict(zip(group_columns, keys, strict=True))
        metadata = {
            "run_id": str(raw_metadata["run_id"]),
            "seed": int(raw_metadata["seed"]),
            "regime_name": str(raw_metadata["regime_name"]),
            "split": str(raw_metadata["split"]),
            "in_sample": bool(raw_metadata["in_sample"]),
        }
        group_allocations = _matching_group(allocations, metadata)
        group_turnover = _matching_group(turnover, metadata)
        group_months = _matching_group(monthly_returns, metadata)
        spy_dominance = _ticker_dominance_fraction(
            group,
            "SPY",
            threshold=config.dominance_weight_threshold,
        )
        shy_dominance = float(
            (group["shy_weight"] >= config.dominance_weight_threshold).mean()
        )
        median_turnover = float(group_turnover["turnover"].median())
        median_target_change = float(group_turnover["target_change_turnover"].median())
        best_month_share = float(
            group_months["best_month_positive_return_share"].iloc[0]
        )
        return_ex_best = float(
            group_months["total_return_excluding_best_month"].iloc[0]
        )
        total_return = float((1.0 + group_months["monthly_return"]).prod() - 1.0)
        high_vol = group[group["high_volatility"]]
        normal_vol = group[~group["high_volatility"]]
        volatility_shift = _volatility_shift(
            high_vol,
            normal_vol,
            group_turnover,
        )
        warnings = {
            "spy_only_collapse": (spy_dominance >= config.dominance_date_fraction),
            "shy_only_collapse": (shy_dominance >= config.dominance_date_fraction),
            "excessive_concentration": (
                float(group["hhi"].median()) >= config.concentration_hhi_threshold
            ),
            "excessive_turnover": median_turnover > max_median_turnover,
            "static_target_policy": (
                median_target_change <= config.static_target_change_threshold
            ),
            "lucky_month_dependence": (
                total_return > 0.0
                and (
                    return_ex_best <= 0.0
                    or best_month_share >= config.lucky_month_positive_return_share
                )
            ),
            "higher_equity_lower_shy_in_high_volatility": (
                volatility_shift["equity_like_weight"] > 0.0
                and volatility_shift["shy_weight"] < 0.0
            ),
        }
        mean_asset_weights = (
            group_allocations.groupby("ticker", sort=True)["target_weight"]
            .mean()
            .sort_values(ascending=False)
        )
        largest_trades = (
            group_allocations.assign(
                abs_trade_weight=group_allocations["trade_weight"].abs()
            )
            .nlargest(10, "abs_trade_weight")
            .loc[
                :,
                [
                    "date",
                    "ticker",
                    "pre_trade_weight",
                    "target_weight",
                    "trade_weight",
                    "abs_trade_weight",
                ],
            ]
        )
        groups.append(
            {
                **metadata,
                "mean_asset_weights": {
                    str(ticker): float(weight)
                    for ticker, weight in mean_asset_weights.items()
                },
                "largest_trades": [
                    {
                        **{
                            column: (
                                pd.Timestamp(row[column]).date().isoformat()
                                if column == "date"
                                else (
                                    str(row[column])
                                    if column == "ticker"
                                    else float(row[column])
                                )
                            )
                            for column in largest_trades.columns
                        }
                    }
                    for row in largest_trades.to_dict(orient="records")
                ],
                "mean_max_weight": float(group["max_weight"].mean()),
                "median_hhi": float(group["hhi"].median()),
                "mean_effective_asset_count": float(
                    group["effective_asset_count"].mean()
                ),
                "mean_active_asset_count": float(group["active_asset_count"].mean()),
                "mean_shy_weight": float(group["shy_weight"].mean()),
                "mean_equity_like_weight": float(group["equity_like_weight"].mean()),
                "spy_dominance_date_fraction": spy_dominance,
                "shy_dominance_date_fraction": shy_dominance,
                "median_turnover": median_turnover,
                "p95_turnover": float(group_turnover["turnover"].quantile(0.95)),
                "median_target_change_turnover": median_target_change,
                "total_return": total_return,
                "best_month": str(
                    group_months.loc[
                        group_months["is_best_month"],
                        "month",
                    ].iloc[0]
                ),
                "best_month_positive_return_share": best_month_share,
                "total_return_excluding_best_month": return_ex_best,
                "high_volatility_shift": volatility_shift,
                "warnings": warnings,
            }
        )

    warning_names = next(iter(groups))["warnings"] if groups else {}
    campaign_warnings = {
        name: {
            "triggered": any(group["warnings"][name] for group in groups),
            "affected_group_count": sum(group["warnings"][name] for group in groups),
        }
        for name in warning_names
    }
    return {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "configuration_id": selected["configuration_id"],
        "experiment_name": selected["experiment_name"],
        "validation_only": True,
        "test_split_used": False,
        "diagnostic_only": True,
        "seed_count": len(selected_runs),
        "seeds": [int(run["seed"]) for run in selected_runs],
        "selection_checkpoints": {
            str(run["run_id"]): str(run["selection_checkpoint"])
            for run in selected_runs
        },
        "selected_models": selected_models,
        "thresholds": {
            **config.model_dump(mode="json"),
            "max_median_weekly_turnover": max_median_turnover,
        },
        "regime_windows": [
            {key: value for key, value in regime.items() if key != "feature_store"}
            for regime in regime_windows
        ],
        "skipped_regimes": skipped_regimes,
        "validation_reconciliation": reconciliation,
        "all_validation_metrics_reconciled": all(
            row["all_metrics_match"] for row in reconciliation
        ),
        "cross_seed_allocation_stability": (
            _cross_seed_allocation_stability(
                allocations,
                concentration,
            )
        ),
        "campaign_warnings": campaign_warnings,
        "groups": groups,
        "sources": sources,
    }


def format_policy_behavior_report(summary: dict[str, Any]) -> str:
    """Format the campaign diagnostic summary as Markdown."""
    lines = [
        "# Policy Behavior Diagnostics",
        "",
        f"Configuration: `{summary['configuration_id']}`",
        "",
        (
            f"Seeds: {summary['seed_count']}; validation only: "
            f"{str(summary['validation_only']).lower()}; test split used: "
            f"{str(summary['test_split_used']).lower()}."
        ),
        "",
        "These warnings are research diagnostics and do not change model selection.",
        "",
        "## Campaign Warnings",
        "",
    ]
    for name, result in summary["campaign_warnings"].items():
        lines.append(
            f"- `{name}`: {str(result['triggered']).lower()} "
            f"({result['affected_group_count']} seed-regime groups)"
        )
    lines.extend(
        [
            "",
            "## Seed And Regime Summary",
            "",
            (
                "| Seed | Regime | Mean SHY | Mean equity-like | Median HHI | "
                "Median turnover | Return ex-best month |"
            ),
            "|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for group in summary["groups"]:
        lines.append(
            f"| {group['seed']} | {group['regime_name']} | "
            f"{group['mean_shy_weight']:.4f} | "
            f"{group['mean_equity_like_weight']:.4f} | "
            f"{group['median_hhi']:.4f} | "
            f"{group['median_turnover']:.4f} | "
            f"{group['total_return_excluding_best_month']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## High-Volatility Response",
            "",
            (
                "High volatility is the configured upper quantile of normalized "
                "`spy_vol_21d` within each regime."
            ),
            "",
        ]
    )
    for group in summary["groups"]:
        shift = group["high_volatility_shift"]
        lines.append(
            f"- Seed {group['seed']} `{group['regime_name']}`: "
            f"SHY shift={shift['shy_weight']:.4f}, "
            f"equity-like shift={shift['equity_like_weight']:.4f}, "
            f"HHI shift={shift['hhi']:.4f}, "
            f"turnover shift={shift['turnover']:.4f}."
        )
    lines.extend(
        [
            "",
            "## Integrity",
            "",
            (
                "- All selected validation metrics reconciled: "
                f"{str(summary['all_validation_metrics_reconciled']).lower()}."
            ),
            "- The test split was not accessed.",
            "- The 2020 and 2022 windows are in-sample behavior diagnostics.",
            "",
        ]
    )
    return "\n".join(lines)


def _allocation_frame(
    *,
    backtest: BacktestResult,
    metadata: dict[str, Any],
    global_features: pd.DataFrame,
    high_volatility_quantile: float,
) -> pd.DataFrame:
    allocation = backtest.weights_target[["date", "ticker", "target_weight"]].copy()
    allocation["date"] = pd.to_datetime(allocation["date"])
    trades = backtest.trades[
        ["date", "ticker", "pre_trade_weight", "trade_weight"]
    ].copy()
    trades["date"] = pd.to_datetime(trades["date"])
    allocation = allocation.merge(
        trades,
        on=["date", "ticker"],
        how="left",
        validate="one_to_one",
    )
    allocation = allocation.merge(
        global_features[["date", "split", "spy_vol_21d"]],
        on="date",
        how="left",
        validate="many_to_one",
        suffixes=("", "_feature"),
    )
    if allocation["spy_vol_21d"].isna().any():
        raise ValueError("global volatility feature missing for decision dates")
    if not (allocation["split"] == metadata["split"]).all():
        raise ValueError("global feature split does not match diagnostic regime")
    threshold = float(
        allocation[["date", "spy_vol_21d"]]
        .drop_duplicates()["spy_vol_21d"]
        .quantile(high_volatility_quantile)
    )
    allocation["high_volatility_threshold"] = threshold
    allocation["high_volatility"] = allocation["spy_vol_21d"] >= threshold
    allocation = allocation.drop(columns=["split"])
    return _attach_metadata(allocation, metadata)


def _run_metadata(
    run: dict[str, Any],
    regime: dict[str, Any],
) -> dict[str, Any]:
    return {
        "run_id": str(run["run_id"]),
        "seed": int(run["seed"]),
        "selection_checkpoint": str(run["selection_checkpoint"]),
        "regime_name": str(regime["name"]),
        "split": str(regime["split"]),
        "in_sample": bool(regime["in_sample"]),
    }


def _selected_registry_rows(
    *,
    selected: dict[str, Any],
    registry: pd.DataFrame,
) -> dict[int, dict[str, Any]]:
    rows = registry[
        (registry["experiment_name"] == selected["experiment_name"])
        & registry["seed"].isin(selected["eligible_seeds"])
    ]
    missing = [
        column for column in RECONCILIATION_METRICS.values() if column not in rows
    ]
    if missing:
        raise ValueError(f"registry missing reconciliation metrics: {missing}")
    return {int(row["seed"]): row for row in rows.to_dict(orient="records")}


def _reconcile_validation_metrics(
    metrics: dict[str, float | None],
    *,
    registry_row: dict[str, Any],
    run_id: str,
) -> dict[str, Any]:
    comparisons = {}
    for metric, registry_column in RECONCILIATION_METRICS.items():
        actual = metrics.get(metric)
        expected = registry_row.get(registry_column)
        matches = (
            actual is not None
            and expected is not None
            and np.isfinite(float(actual))
            and np.isfinite(float(expected))
            and isclose(
                float(actual),
                float(expected),
                rel_tol=1e-10,
                abs_tol=1e-10,
            )
        )
        comparisons[metric] = {
            "computed": None if actual is None else float(actual),
            "registry": None if expected is None else float(expected),
            "matches": bool(matches),
        }
    if not all(value["matches"] for value in comparisons.values()):
        raise ValueError(f"selected validation metrics do not reconcile: {run_id}")
    return {
        "run_id": run_id,
        "seed": int(registry_row["seed"]),
        "all_metrics_match": True,
        "metrics": comparisons,
    }


def _drawdown_row(
    nav: pd.DataFrame,
    *,
    start_index: int,
    end_index: int,
    recovered: bool,
) -> dict[str, Any]:
    episode = nav.iloc[start_index : end_index + 1]
    trough_index = int(episode["drawdown"].idxmin())
    return {
        "start_date": pd.Timestamp(nav.loc[start_index, "date"]),
        "trough_date": pd.Timestamp(nav.loc[trough_index, "date"]),
        "recovery_date": (
            pd.Timestamp(nav.loc[end_index, "date"]) if recovered else pd.NaT
        ),
        "end_date": pd.Timestamp(nav.loc[end_index, "date"]),
        "trough_drawdown": float(nav.loc[trough_index, "drawdown"]),
        "duration_trading_days": int(end_index - start_index + 1),
        "recovered": recovered,
    }


def _ticker_dominance_fraction(
    concentration: pd.DataFrame,
    ticker: str,
    *,
    threshold: float,
) -> float:
    return float(
        (
            (concentration["max_ticker"] == ticker)
            & (concentration["max_weight"] >= threshold)
        ).mean()
    )


def _volatility_shift(
    high_volatility: pd.DataFrame,
    normal_volatility: pd.DataFrame,
    turnover: pd.DataFrame,
) -> dict[str, float]:
    if high_volatility.empty or normal_volatility.empty:
        raise ValueError("each regime must contain high- and normal-volatility dates")
    high_dates = set(pd.to_datetime(high_volatility["date"]))
    high_turnover = turnover[pd.to_datetime(turnover["date"]).isin(high_dates)][
        "turnover"
    ]
    normal_turnover = turnover[~pd.to_datetime(turnover["date"]).isin(high_dates)][
        "turnover"
    ]
    return {
        "shy_weight": float(
            high_volatility["shy_weight"].mean()
            - normal_volatility["shy_weight"].mean()
        ),
        "equity_like_weight": float(
            high_volatility["equity_like_weight"].mean()
            - normal_volatility["equity_like_weight"].mean()
        ),
        "hhi": float(high_volatility["hhi"].mean() - normal_volatility["hhi"].mean()),
        "turnover": float(high_turnover.mean() - normal_turnover.mean()),
    }


def _cross_seed_allocation_stability(
    allocations: pd.DataFrame,
    concentration: pd.DataFrame,
) -> list[dict[str, Any]]:
    rows = []
    for regime_name, group in concentration.groupby("regime_name", sort=True):
        date_seed = group.pivot(
            index="date",
            columns="seed",
            values="max_weight",
        )
        dominant = group.pivot(
            index="date",
            columns="seed",
            values="max_ticker",
        )
        agreement = dominant.apply(
            lambda row: float(row.value_counts(normalize=True).iloc[0]),
            axis=1,
        )
        regime_allocations = allocations[allocations["regime_name"] == regime_name]
        pairwise_distances = []
        for _, date_group in regime_allocations.groupby("date", sort=True):
            seed_weights = date_group.pivot(
                index="seed",
                columns="ticker",
                values="target_weight",
            ).sort_index()
            values = seed_weights.to_numpy(dtype=float)
            for left in range(len(values)):
                for right in range(left + 1, len(values)):
                    pairwise_distances.append(
                        float(0.5 * np.abs(values[left] - values[right]).sum())
                    )
        rows.append(
            {
                "regime_name": str(regime_name),
                "mean_cross_seed_max_weight_std": float(
                    date_seed.std(axis=1, ddof=1).mean()
                ),
                "mean_dominant_asset_agreement": float(agreement.mean()),
                "mean_pairwise_one_way_allocation_distance": float(
                    np.mean(pairwise_distances) if pairwise_distances else 0.0
                ),
            }
        )
    return rows


def _matching_group(
    frame: pd.DataFrame,
    metadata: dict[str, Any],
) -> pd.DataFrame:
    mask = pd.Series(True, index=frame.index)
    for column, value in metadata.items():
        mask &= frame[column] == value
    result = frame[mask]
    if result.empty:
        raise ValueError(f"missing diagnostic group: {metadata}")
    return result


def _attach_metadata(
    frame: pd.DataFrame,
    metadata: dict[str, Any],
) -> pd.DataFrame:
    result = frame.copy()
    for column, value in reversed(list(metadata.items())):
        result.insert(0, column, value)
    return result


def _is_equity_like(asset_class: str) -> bool:
    return "equity" in asset_class or asset_class == "real_estate"


def _load_global_features(path: Path) -> pd.DataFrame:
    features = pd.read_parquet(path)
    _require_columns(
        features,
        ["date", "split", "spy_vol_21d"],
        "global features",
    )
    features = features[["date", "split", "spy_vol_21d"]].copy()
    features["date"] = pd.to_datetime(features["date"])
    if features["date"].duplicated().any():
        raise ValueError("global features contain duplicate dates")
    if not features["spy_vol_21d"].map(np.isfinite).all():
        raise ValueError("global volatility features must be finite")
    return features


def _read_registry(path: Path) -> pd.DataFrame:
    if path.suffix == ".csv":
        return pd.read_csv(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError("registry must be a CSV or Parquet file")


def _read_json(path: Path) -> dict[str, Any]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise TypeError(f"expected JSON object: {path}")
    return loaded


def _source(path: Path, root: Path) -> dict[str, str]:
    return {
        "path": _display_path(path, root),
        "sha256": _sha256_file(path),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _display_path(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _resolve_path(root: Path, path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return root / candidate


def _require_columns(
    frame: pd.DataFrame,
    columns: list[str],
    label: str,
) -> None:
    missing = [column for column in columns if column not in frame]
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def _concat(frames: list[pd.DataFrame]) -> pd.DataFrame:
    nonempty = [frame for frame in frames if not frame.empty]
    if not nonempty:
        return pd.DataFrame()
    return pd.concat(nonempty, ignore_index=True)

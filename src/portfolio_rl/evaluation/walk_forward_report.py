"""Frozen-checkpoint evaluation and aggregation for PR17 walk-forward folds."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from portfolio_rl.data.feature_store import PortfolioFeatureStore
from portfolio_rl.data.walk_forward import load_outer_evaluation_dataset
from portfolio_rl.evaluation.backtest import (
    BacktestResult,
    run_weight_policy_backtest,
    write_backtest_artifacts,
)
from portfolio_rl.evaluation.ensemble import calculate_disagreement_metrics
from portfolio_rl.policies.baseline_policies import (
    BuyAndHoldEqualWeightPolicy,
    EqualWeightWeeklyPolicy,
    InverseVolatilityPolicy,
    MomentumPolicy,
    SingleAssetPolicy,
    WeightPolicy,
)
from portfolio_rl.policies.ensemble_policy import MeanWeightEnsemblePolicy
from portfolio_rl.policies.overlays import PartialRebalancePolicy
from portfolio_rl.policies.sb3_policy import load_sb3_weight_policy
from portfolio_rl.training.walk_forward_runner import (
    SelectionResult,
    WalkForwardCampaignConfig,
    selection_output_dir,
    verify_selection_freeze,
)

ENSEMBLE = "five_seed_mean_target_ensemble"
PRIMARY_REFERENCE = "equal_weight_weekly"


def evaluate_frozen_selection(
    *,
    config: WalkForwardCampaignConfig,
    fold_id: str,
    selection_results: list[SelectionResult],
    output_dir: str | Path,
    pilot: bool,
) -> Path:
    """Verify freezes first, then load and evaluate the outer fold."""
    expected_seeds = [config.pilot_seed] if pilot else list(config.seeds)
    by_seed = {result.seed: result for result in selection_results}
    if sorted(by_seed) != sorted(expected_seeds):
        raise ValueError("selection freezes do not match required outer seeds")
    verified = {
        seed: verify_selection_freeze(result.output_dir, config=config)
        for seed, result in by_seed.items()
    }
    for seed, result in verified.items():
        if result.fold_id != fold_id:
            raise ValueError(f"selection freeze fold mismatch for seed {seed}")

    # This is intentionally the first outer-data access in this function.
    dataset = load_outer_evaluation_dataset(config.data_root / fold_id)
    store = PortfolioFeatureStore(dataset, "outer_evaluation")
    destination = Path(output_dir)
    if destination.exists():
        _verify_completed_fold(destination, config, fold_id, expected_seeds)
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    try:
        factories = {
            seed: _model_factory(result.selected_model_path, config)
            for seed, result in verified.items()
        }
        backtests: dict[str, BacktestResult] = {}
        categories: dict[str, str] = {}
        receipts = []
        for seed, factory in factories.items():
            started = datetime.now(UTC).isoformat()
            strategy = f"ppo_seed_{seed}"
            backtests[strategy] = _run(store, factory(), strategy, config)
            categories[strategy] = "seed"
            receipts.append(
                {
                    "fold_id": fold_id,
                    "seed": seed,
                    "run_id": verified[seed].run_id,
                    "selection_freeze_path": str(verified[seed].freeze_path),
                    "selection_freeze_sha256": _sha256(verified[seed].freeze_path),
                    "selected_model_sha256": _sha256(
                        verified[seed].selected_model_path
                    ),
                    "outer_evaluation_start_time": started,
                }
            )

        ensemble_member_targets = pd.DataFrame()
        overlay_audit = pd.DataFrame()
        if not pilot:
            ensemble = _ensemble(factories)
            backtests[ENSEMBLE] = _run(store, ensemble, ENSEMBLE, config)
            categories[ENSEMBLE] = "ensemble"
            ensemble_member_targets = _member_targets_frame(
                ensemble,
                fold_id=fold_id,
                asset_order=store.asset_order,
            )
            overlay_frames = []
            for alpha in config.alphas:
                overlay = PartialRebalancePolicy(
                    base_policy=_ensemble(factories),
                    alpha=alpha,
                )
                strategy = f"ensemble_alpha_{alpha:.2f}"
                backtests[strategy] = _run(store, overlay, strategy, config)
                categories[strategy] = "overlay"
                overlay_frames.append(
                    _overlay_records_frame(
                        overlay,
                        fold_id=fold_id,
                        strategy=strategy,
                        asset_order=store.asset_order,
                    )
                )
            overlay_audit = pd.concat(overlay_frames, ignore_index=True)
            _assert_alpha_one_reconciliation(backtests)

        for strategy, factory in _baseline_factories(store, config).items():
            backtests[strategy] = _run(store, factory(), strategy, config)
            categories[strategy] = "baseline"

        metrics = _fold_metrics(
            fold_id=fold_id,
            backtests=backtests,
            categories=categories,
        )
        metrics.to_csv(temporary / "metrics.csv", index=False)
        (temporary / "outer_evaluation_receipts.json").write_text(
            json.dumps(receipts, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        for strategy, result in backtests.items():
            write_backtest_artifacts(result, temporary / "backtests" / strategy)
        if not ensemble_member_targets.empty:
            ensemble_member_targets.to_parquet(
                temporary / "ensemble_member_targets.parquet",
                index=False,
            )
            calculate_disagreement_metrics(
                ensemble_member_targets.rename(columns={"fold_id": "regime"})
            ).rename(columns={"regime": "fold_id"}).to_csv(
                temporary / "cross_seed_disagreement.csv",
                index=False,
            )
        if not overlay_audit.empty:
            overlay_audit.to_parquet(
                temporary / "overlay_target_audit.parquet",
                index=False,
            )
        manifest = {
            "schema_version": 1,
            "campaign_id": config.campaign_id,
            "pilot": pilot,
            "fold_id": fold_id,
            "seeds": expected_seeds,
            "outer_evaluation_start": store.date_at(0).date().isoformat(),
            "outer_evaluation_end": store.date_at(store.n_rows - 1)
            .date()
            .isoformat(),
            "campaign_config_sha256": config.config_sha256,
            "walk_forward_data_manifest_sha256": config.data_manifest_sha256,
            "selection_freeze_sha256": {
                str(seed): _sha256(result.freeze_path)
                for seed, result in verified.items()
            },
            "primary_initialization": "equal_weight",
            "transaction_cost_bps": config.transaction_cost_bps,
            "turnover_convention": "half_l1",
            "candidate_selected": False,
            "dynamic_value_diagnostics_run": False,
            "evaluation_git_commit": _git_commit(config.config_path.parent),
            "completed_at": datetime.now(UTC).isoformat(),
        }
        (temporary / "fold_evaluation_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


def run_evaluation_stage(
    config: WalkForwardCampaignConfig,
    *,
    pilot: bool,
) -> list[Path]:
    """Evaluate pilot or production freezes after a complete freeze barrier."""
    folds = [config.pilot_fold] if pilot else list(config.folds)
    seeds = [config.pilot_seed] if pilot else list(config.seeds)
    all_selections = {
        (fold, seed): verify_selection_freeze(
            selection_output_dir(
                config,
                fold_id=fold,
                seed=seed,
                pilot=pilot,
            ),
            config=config,
        )
        for fold in folds
        for seed in seeds
    }
    outputs = []
    for fold in folds:
        output = evaluate_frozen_selection(
            config=config,
            fold_id=fold,
            selection_results=[all_selections[(fold, seed)] for seed in seeds],
            output_dir=evaluation_output_dir(config, fold_id=fold, pilot=pilot),
            pilot=pilot,
        )
        outputs.append(output)
    if pilot:
        _write_pilot_verification(config, outputs[0], all_selections)
    return outputs


def aggregate_walk_forward_results(config: WalkForwardCampaignConfig) -> Path:
    """Aggregate fold-reset-aware production evidence without selection."""
    _verify_pilot_gate(config)
    fold_dirs = [
        evaluation_output_dir(config, fold_id=fold, pilot=False)
        for fold in config.folds
    ]
    for fold, path in zip(config.folds, fold_dirs, strict=True):
        _verify_completed_fold(path, config, fold, list(config.seeds))
    destination = config.campaign_root / "aggregate"
    if destination.exists():
        raise FileExistsError(f"walk-forward aggregate already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=".aggregate.", dir=destination.parent)
    )
    try:
        fold_metrics = pd.concat(
            [pd.read_csv(path / "metrics.csv") for path in fold_dirs],
            ignore_index=True,
        )
        fold_metrics.to_csv(temporary / "fold_metrics.csv", index=False)
        for category, name in (
            ("seed", "seed_metrics.csv"),
            ("ensemble", "ensemble_metrics.csv"),
            ("overlay", "overlay_metrics.csv"),
            ("baseline", "baseline_metrics.csv"),
        ):
            fold_metrics[fold_metrics["category"] == category].to_csv(
                temporary / name,
                index=False,
            )
        active, monthly = _active_return_paths(fold_dirs, fold_metrics)
        active.to_parquet(temporary / "active_returns.parquet", index=False)
        monthly.to_csv(temporary / "monthly_active_returns.csv", index=False)
        summary = _aggregate_summary(fold_metrics)
        summary.to_csv(temporary / "aggregate_metrics.csv", index=False)
        comparison_by_fold, comparison_summary = _alpha_025_comparisons(
            fold_metrics
        )
        comparison_by_fold.to_csv(
            temporary / "alpha_025_baseline_comparison_by_fold.csv",
            index=False,
        )
        comparison_summary.to_csv(
            temporary / "alpha_025_baseline_comparison_summary.csv",
            index=False,
        )
        dispersion, improvement = _seed_ensemble_summaries(fold_metrics)
        dispersion.to_csv(temporary / "seed_dispersion.csv", index=False)
        improvement.to_csv(
            temporary / "ensemble_improvement_over_median_seed.csv",
            index=False,
        )
        frontier = _overlay_frontier(fold_metrics)
        frontier.to_csv(temporary / "overlay_frontier_by_fold.csv", index=False)
        disagreement = pd.concat(
            [pd.read_csv(path / "cross_seed_disagreement.csv") for path in fold_dirs],
            ignore_index=True,
        )
        disagreement.to_csv(
            temporary / "cross_seed_disagreement.csv",
            index=False,
        )
        manifest = {
            "schema_version": 1,
            "campaign_id": config.campaign_id,
            "folds": list(config.folds),
            "seeds": list(config.seeds),
            "campaign_config_sha256": config.config_sha256,
            "walk_forward_data_manifest_sha256": config.data_manifest_sha256,
            "fold_resets_are_explicit": True,
            "concatenated_live_portfolio_claimed": False,
            "candidate_selected": False,
            "pr18_required_before_selection": True,
            "aggregation_git_commit": _git_commit(config.config_path.parent),
            "completed_at": datetime.now(UTC).isoformat(),
        }
        (temporary / "aggregation_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (temporary / "walk_forward_report.md").write_text(
            _format_report(
                fold_metrics,
                summary,
                comparison_by_fold,
                comparison_summary,
                dispersion,
                improvement,
                frontier,
            ),
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


def evaluation_output_dir(
    config: WalkForwardCampaignConfig,
    *,
    fold_id: str,
    pilot: bool,
) -> Path:
    root = config.campaign_root / ("pilot/evaluation" if pilot else "evaluation")
    return root / fold_id


def _model_factory(
    path: Path,
    config: WalkForwardCampaignConfig,
) -> Callable[[], WeightPolicy]:
    return lambda: load_sb3_weight_policy(path, action_temperature=0.5)


def _ensemble(
    factories: dict[int, Callable[[], WeightPolicy]],
) -> MeanWeightEnsemblePolicy:
    return MeanWeightEnsemblePolicy(
        member_policies={
            f"seed_{seed}": factory() for seed, factory in factories.items()
        }
    )


def _run(
    store: PortfolioFeatureStore,
    policy: WeightPolicy,
    strategy: str,
    config: WalkForwardCampaignConfig,
) -> BacktestResult:
    return run_weight_policy_backtest(
        feature_store=store,
        policy=policy,
        strategy=strategy,
        rebalance_frequency_trading_days=config.rebalance_days,
        transaction_cost_bps=config.transaction_cost_bps,
        inverse_vol_lookback_trading_days=(
            config.momentum_lookback
            if strategy == "momentum_63d_top3_equal_weight"
            else config.inverse_vol_lookback
        ),
    )


def _baseline_factories(store, config):
    n_assets = store.n_assets
    order = store.asset_order
    return {
        PRIMARY_REFERENCE: lambda: EqualWeightWeeklyPolicy(n_assets),
        "buy_and_hold_equal_weight": lambda: BuyAndHoldEqualWeightPolicy(n_assets),
        "inverse_volatility": lambda: InverseVolatilityPolicy(n_assets),
        "momentum_63d_top3_equal_weight": lambda: MomentumPolicy(
            n_assets,
            top_k=config.momentum_top_k,
        ),
        "spy_only": lambda: SingleAssetPolicy(order, "SPY"),
        "shy_only": lambda: SingleAssetPolicy(order, "SHY"),
    }


def _fold_metrics(*, fold_id, backtests, categories):
    gross_returns = {
        strategy: _same_path_gross_total_return(result)
        for strategy, result in backtests.items()
    }
    hurdle = backtests[PRIMARY_REFERENCE].metrics
    hurdle_gross = gross_returns[PRIMARY_REFERENCE]
    rows = []
    for strategy, result in backtests.items():
        metrics = dict(result.metrics)
        rows.append(
            {
                "fold_id": fold_id,
                "strategy": strategy,
                "category": categories[strategy],
                **metrics,
                "gross_total_return": gross_returns[strategy],
                "cost_return_impact": metrics["total_return"]
                - gross_returns[strategy],
                "active_gross_return_vs_equal_weight": gross_returns[strategy]
                - hurdle_gross,
                "active_return_vs_equal_weight": metrics["total_return"]
                - hurdle["total_return"],
                "active_sharpe_vs_equal_weight": _difference(
                    metrics["sharpe_ratio"], hurdle["sharpe_ratio"]
                ),
                "drawdown_difference_vs_equal_weight": metrics["max_drawdown"]
                - hurdle["max_drawdown"],
            }
        )
    return pd.DataFrame(rows)


def _same_path_gross_total_return(result: BacktestResult) -> float:
    net_total_return = float(result.metrics["total_return"])
    cost_fractions = result.costs["transaction_cost_fraction"].to_numpy(
        dtype=np.float64
    )
    cost_multiplier = float(np.prod(1.0 - cost_fractions))
    if not np.isfinite(cost_multiplier) or cost_multiplier <= 0.0:
        raise ValueError("realized transaction-cost multiplier must be positive")
    return (1.0 + net_total_return) / cost_multiplier - 1.0


def _member_targets_frame(ensemble, *, fold_id, asset_order):
    rows = []
    for record in ensemble.member_target_records:
        for ticker, target, current in zip(
            asset_order,
            record.weights,
            record.live_current_weights,
            strict=True,
        ):
            rows.append(
                {
                    "fold_id": fold_id,
                    "date": record.date,
                    "decision_step": record.decision_step,
                    "member": record.member,
                    "ticker": ticker,
                    "target_weight": target,
                    "live_current_weight": current,
                }
            )
    return pd.DataFrame(rows)


def _overlay_records_frame(overlay, *, fold_id, strategy, asset_order):
    rows = []
    for record in overlay.records:
        for ticker, current, raw, executed in zip(
            asset_order,
            record.current_weights,
            record.raw_policy_target,
            record.executed_target,
            strict=True,
        ):
            rows.append(
                {
                    "fold_id": fold_id,
                    "strategy": strategy,
                    "date": record.date,
                    "decision_step": record.decision_step,
                    "ticker": ticker,
                    "live_current_weight": current,
                    "raw_target_weight": raw,
                    "executed_target_weight": executed,
                    "raw_half_l1_turnover": record.raw_half_l1_turnover,
                    "executed_half_l1_turnover": record.executed_half_l1_turnover,
                }
            )
    return pd.DataFrame(rows)


def _assert_alpha_one_reconciliation(backtests):
    ensemble = backtests[ENSEMBLE]
    alpha_one = backtests["ensemble_alpha_1.00"]
    pd.testing.assert_frame_equal(
        ensemble.nav,
        alpha_one.nav.assign(strategy=ENSEMBLE),
        check_exact=True,
    )
    pd.testing.assert_frame_equal(
        ensemble.weights_target,
        alpha_one.weights_target.assign(strategy=ENSEMBLE),
        check_exact=True,
    )
    pd.testing.assert_frame_equal(
        ensemble.costs,
        alpha_one.costs.assign(strategy=ENSEMBLE),
        check_exact=True,
    )


def _active_return_paths(fold_dirs, fold_metrics):
    active_frames = []
    monthly_frames = []
    for fold_dir in fold_dirs:
        fold_id = fold_dir.name
        hurdle = pd.read_parquet(
            fold_dir / "backtests" / PRIMARY_REFERENCE / "nav.parquet"
        )[["date", "daily_return"]].rename(
            columns={"daily_return": "equal_weight_daily_return"}
        )
        strategies = fold_metrics.loc[
            fold_metrics["fold_id"] == fold_id, "strategy"
        ]
        for strategy in strategies:
            nav = pd.read_parquet(
                fold_dir / "backtests" / strategy / "nav.parquet"
            )[["date", "daily_return"]]
            paired = nav.merge(hurdle, on="date", validate="one_to_one")
            paired.insert(0, "strategy", strategy)
            paired.insert(0, "fold_id", fold_id)
            paired["active_daily_return"] = (
                paired["daily_return"] - paired["equal_weight_daily_return"]
            )
            active_frames.append(paired)
            month = paired.assign(
                month=pd.to_datetime(paired["date"]).dt.to_period("M").astype(str)
            ).groupby("month", sort=True)
            monthly_frames.append(
                month.apply(
                    lambda frame: pd.Series(
                        {
                            "candidate_monthly_return": np.prod(
                                1.0 + frame["daily_return"]
                            )
                            - 1.0,
                            "equal_weight_monthly_return": np.prod(
                                1.0 + frame["equal_weight_daily_return"]
                            )
                            - 1.0,
                        }
                    ),
                    include_groups=False,
                )
                .reset_index()
                .assign(fold_id=fold_id, strategy=strategy)
            )
    active = pd.concat(active_frames, ignore_index=True)
    monthly = pd.concat(monthly_frames, ignore_index=True)
    monthly["active_monthly_return"] = (
        monthly["candidate_monthly_return"]
        - monthly["equal_weight_monthly_return"]
    )
    return active, monthly


def _aggregate_summary(metrics):
    rows = []
    for strategy, frame in metrics.groupby("strategy", sort=True):
        rows.append(
            {
                "strategy": strategy,
                "category": frame["category"].iloc[0],
                "fold_count": len(frame),
                "positive_active_return_fold_count": int(
                    (frame["active_return_vs_equal_weight"] > 0.0).sum()
                ),
                "positive_active_sharpe_fold_count": int(
                    (frame["active_sharpe_vs_equal_weight"] > 0.0).sum()
                ),
                "improved_drawdown_fold_count": int(
                    (frame["drawdown_difference_vs_equal_weight"] > 0.0).sum()
                ),
                "median_active_return": frame[
                    "active_return_vs_equal_weight"
                ].median(),
                "median_active_sharpe": frame[
                    "active_sharpe_vs_equal_weight"
                ].median(),
                "worst_fold_active_return": frame[
                    "active_return_vs_equal_weight"
                ].min(),
                "median_drawdown_difference": frame[
                    "drawdown_difference_vs_equal_weight"
                ].median(),
                "worst_fold_drawdown_difference": frame[
                    "drawdown_difference_vs_equal_weight"
                ].min(),
                "median_average_weekly_turnover": frame[
                    "average_weekly_turnover"
                ].median(),
                "median_transaction_cost_drag": frame[
                    "transaction_cost_drag"
                ].median(),
                "worst_fold_transaction_cost_drag": frame[
                    "transaction_cost_drag"
                ].max(),
                "median_active_gross_return": frame[
                    "active_gross_return_vs_equal_weight"
                ].median(),
                "median_active_net_return": frame[
                    "active_return_vs_equal_weight"
                ].median(),
                "worst_fold_active_gross_return": frame[
                    "active_gross_return_vs_equal_weight"
                ].min(),
            }
        )
    return pd.DataFrame(rows)


def _alpha_025_comparisons(
    metrics: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidate_name = "ensemble_alpha_0.25"
    benchmarks = (
        PRIMARY_REFERENCE,
        "inverse_volatility",
        "momentum_63d_top3_equal_weight",
        "buy_and_hold_equal_weight",
    )
    rows = []
    for fold_id in sorted(metrics["fold_id"].unique()):
        fold = metrics[metrics["fold_id"] == fold_id].set_index("strategy")
        candidate = fold.loc[candidate_name]
        for benchmark_name in benchmarks:
            benchmark = fold.loc[benchmark_name]
            rows.append(
                {
                    "fold_id": fold_id,
                    "candidate": candidate_name,
                    "benchmark": benchmark_name,
                    "active_net_return": candidate["total_return"]
                    - benchmark["total_return"],
                    "active_gross_return": candidate["gross_total_return"]
                    - benchmark["gross_total_return"],
                    "active_sharpe": candidate["sharpe_ratio"]
                    - benchmark["sharpe_ratio"],
                    "max_drawdown_difference": candidate["max_drawdown"]
                    - benchmark["max_drawdown"],
                    "candidate_cost_drag": candidate["transaction_cost_drag"],
                    "benchmark_cost_drag": benchmark["transaction_cost_drag"],
                }
            )
    by_fold = pd.DataFrame(rows)
    summary_rows = []
    for benchmark, frame in by_fold.groupby("benchmark", sort=False):
        summary_rows.append(
            {
                "candidate": candidate_name,
                "benchmark": benchmark,
                "positive_active_net_return_fold_count": int(
                    (frame["active_net_return"] > 0.0).sum()
                ),
                "median_active_net_return": frame["active_net_return"].median(),
                "worst_fold_active_net_return": frame["active_net_return"].min(),
                "median_active_gross_return": frame[
                    "active_gross_return"
                ].median(),
                "median_active_sharpe": frame["active_sharpe"].median(),
                "median_max_drawdown_difference": frame[
                    "max_drawdown_difference"
                ].median(),
            }
        )
    return by_fold, pd.DataFrame(summary_rows)


def _seed_ensemble_summaries(metrics):
    seeds = metrics[metrics["category"] == "seed"]
    ensemble = metrics[metrics["strategy"] == ENSEMBLE].set_index("fold_id")
    dispersion_rows = []
    improvement_rows = []
    for fold_id, frame in seeds.groupby("fold_id", sort=True):
        returns = frame["total_return"]
        sharpes = frame["sharpe_ratio"]
        dispersion_rows.append(
            {
                "fold_id": fold_id,
                "return_range": returns.max() - returns.min(),
                "return_std": returns.std(ddof=0),
                "return_iqr": returns.quantile(0.75) - returns.quantile(0.25),
                "sharpe_range": sharpes.max() - sharpes.min(),
                "sharpe_std": sharpes.std(ddof=0),
                "sharpe_iqr": sharpes.quantile(0.75) - sharpes.quantile(0.25),
            }
        )
        member = ensemble.loc[fold_id]
        improvement_rows.append(
            {
                "fold_id": fold_id,
                "ensemble_return_minus_median_seed": member["total_return"]
                - returns.median(),
                "ensemble_sharpe_minus_median_seed": member["sharpe_ratio"]
                - sharpes.median(),
                "ensemble_drawdown_minus_median_seed": member["max_drawdown"]
                - frame["max_drawdown"].median(),
                "ensemble_turnover_minus_median_seed": member[
                    "average_weekly_turnover"
                ]
                - frame["average_weekly_turnover"].median(),
            }
        )
    return pd.DataFrame(dispersion_rows), pd.DataFrame(improvement_rows)


def _overlay_frontier(metrics):
    overlays = metrics[metrics["category"] == "overlay"].copy()
    flags = []
    for _, row in overlays.iterrows():
        peers = overlays[overlays["fold_id"] == row["fold_id"]]
        dominated = (
            (peers["total_return"] >= row["total_return"])
            & (
                peers["average_weekly_turnover"]
                <= row["average_weekly_turnover"]
            )
            & (
                (peers["total_return"] > row["total_return"])
                | (
                    peers["average_weekly_turnover"]
                    < row["average_weekly_turnover"]
                )
            )
        ).any()
        flags.append(not dominated)
    overlays["on_return_turnover_frontier"] = flags
    return overlays


def _format_report(
    fold_metrics,
    summary,
    comparison_by_fold,
    comparison_summary,
    dispersion,
    improvement,
    frontier,
):
    lines = [
        "# Nested Walk-Forward Campaign",
        "",
        (
            "Each outer fold starts from NAV 1.0 and equal weights. Fold returns "
            "are not presented as one uninterrupted live portfolio."
        ),
        "",
        (
            "No candidate is selected in PR 17. PR 18 execution stress and 2022 "
            "attribution remain required."
        ),
        "",
        "## Aggregate Evidence",
        "",
        "| Strategy | Positive return folds | Median active return | Median active Sharpe | Worst active return | Improved drawdown folds | Median drawdown difference | Worst drawdown difference | Median turnover | Median cost drag | Worst cost drag |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.strategy} | {row.positive_active_return_fold_count}/4 | "
            f"{row.median_active_return:.2%} | {row.median_active_sharpe:.3f} | "
            f"{row.worst_fold_active_return:.2%} | "
            f"{row.improved_drawdown_fold_count}/4 | "
            f"{row.median_drawdown_difference:.2%} | "
            f"{row.worst_fold_drawdown_difference:.2%} | "
            f"{row.median_average_weekly_turnover:.2%} | "
            f"{row.median_transaction_cost_drag:.2%} | "
            f"{row.worst_fold_transaction_cost_drag:.2%} |"
        )
    fold_table = fold_metrics[
        fold_metrics["strategy"].isin(
            [
                ENSEMBLE,
                *[f"ensemble_alpha_{alpha:.2f}" for alpha in (0.25, 0.5, 0.75, 1.0)],
                PRIMARY_REFERENCE,
                "inverse_volatility",
                "momentum_63d_top3_equal_weight",
            ]
        )
    ][
        [
            "fold_id",
            "strategy",
            "total_return",
            "active_return_vs_equal_weight",
            "sharpe_ratio",
            "drawdown_difference_vs_equal_weight",
            "average_weekly_turnover",
            "transaction_cost_drag",
            "active_gross_return_vs_equal_weight",
        ]
    ]
    overlay_by_fold = fold_metrics[fold_metrics["category"] == "overlay"][
        [
            "fold_id",
            "strategy",
            "active_return_vs_equal_weight",
            "active_gross_return_vs_equal_weight",
            "active_sharpe_vs_equal_weight",
            "drawdown_difference_vs_equal_weight",
            "transaction_cost_drag",
        ]
    ]
    overlay_cost = summary[summary["category"] == "overlay"][
        [
            "strategy",
            "median_transaction_cost_drag",
            "worst_fold_transaction_cost_drag",
            "median_active_gross_return",
            "median_active_net_return",
        ]
    ]
    lines.extend(
        [
            "",
            "## Fold-Level Primary Results",
            "",
            _markdown_table(fold_table),
            "",
            "## Overlay Results by Fold",
            "",
            _markdown_table(overlay_by_fold),
            "",
            (
                "The failed alpha 0.25 fold can be assessed directly from the "
                "active return and active Sharpe columns above; no fold is hidden "
                "by the median."
            ),
            "",
            "## Overlay Cost Diagnostics",
            "",
            _markdown_table(overlay_cost),
            "",
            (
                "Gross returns reconstruct the same realized target path before "
                "costs by removing each recorded multiplicative transaction-cost "
                "factor. They are not separate counterfactual policy runs."
            ),
            "",
            "## Alpha 0.25 Versus Transparent Rules by Fold",
            "",
            _markdown_table(comparison_by_fold),
            "",
            "## Alpha 0.25 Versus Transparent Rules — Summary",
            "",
            _markdown_table(comparison_summary),
            "",
            "## Seed Dispersion",
            "",
            _markdown_table(dispersion),
            "",
            "## Ensemble Improvement Over Median Seed",
            "",
            _markdown_table(improvement),
            "",
            "## Overlay Frontier",
            "",
            _markdown_table(frontier[
                [
                    "fold_id",
                    "strategy",
                    "total_return",
                    "average_weekly_turnover",
                    "on_return_turnover_frontier",
                ]
            ]),
            "",
            (
            "The outer periods are pseudo-out-of-sample fold evaluations; "
                "2024 remains consumed development/selection data and is not "
                "used here."
            ),
            (
                "No continuous chained NAV is produced. Every fold begins at NAV "
                "1.0 from a new equal-weight endowment and uses newly trained, "
                "fold-specific models. Aggregate evidence is computed from fold-level "
                "metrics and fold-labeled active-return distributions."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "|" + "|".join("---" for _ in columns) + "|"
    rows = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows])


def _write_pilot_verification(config, fold_dir, selections):
    destination = config.campaign_root / "pilot" / "pilot_verification.json"
    if destination.exists():
        _verify_pilot_gate(config)
        return
    result = next(iter(selections.values()))
    freeze = json.loads(result.freeze_path.read_text(encoding="utf-8"))
    metrics = pd.read_csv(fold_dir / "metrics.csv")
    passed = (
        freeze["evaluated_checkpoint_steps"][:2] == [25_000, 50_000]
        and 50_000 <= freeze["actual_total_timesteps"] < 52_080
        and freeze["evaluated_checkpoint_steps"][-1]
        == freeze["actual_total_timesteps"]
        and {"ppo_seed_42", PRIMARY_REFERENCE}.issubset(set(metrics["strategy"]))
        and not metrics.select_dtypes(include="number").isna().all(axis=None)
    )
    payload = {
        "schema_version": 1,
        "campaign_id": config.campaign_id,
        "purpose": "orchestration_and_schema_verification_only",
        "investment_performance_interpretable": False,
        "fold_id": config.pilot_fold,
        "seed": config.pilot_seed,
        "requested_total_timesteps": config.pilot_timesteps,
        "actual_total_timesteps": freeze["actual_total_timesteps"],
        "selection_freeze_sha256": _sha256(result.freeze_path),
        "campaign_config_sha256": config.config_sha256,
        "passed": bool(passed),
        "verified_at": datetime.now(UTC).isoformat(),
    }
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not passed:
        raise RuntimeError("walk-forward pilot verification failed")


def _verify_pilot_gate(config):
    path = config.campaign_root / "pilot" / "pilot_verification.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("campaign_id") != config.campaign_id
        or payload.get("campaign_config_sha256") != config.config_sha256
        or payload.get("passed") is not True
    ):
        raise ValueError("production campaign requires a matching passing pilot")


def _verify_completed_fold(directory, config, fold_id, seeds):
    manifest = json.loads(
        (directory / "fold_evaluation_manifest.json").read_text(encoding="utf-8")
    )
    if (
        manifest.get("campaign_id") != config.campaign_id
        or manifest.get("fold_id") != fold_id
        or manifest.get("seeds") != seeds
        or manifest.get("campaign_config_sha256") != config.config_sha256
    ):
        raise ValueError(f"completed fold output conflicts with campaign: {directory}")
    freeze_hashes = manifest.get("selection_freeze_sha256")
    if not isinstance(freeze_hashes, dict):
        raise TypeError("completed fold manifest has no selection freeze hashes")
    for seed in seeds:
        freeze = selection_output_dir(
            config,
            fold_id=fold_id,
            seed=seed,
            pilot=bool(manifest.get("pilot")),
        ) / "selection_freeze.json"
        if _sha256(freeze) != freeze_hashes.get(str(seed)):
            raise ValueError(
                f"completed fold selection-freeze hash mismatch: {fold_id}/{seed}"
            )
    for required in ("metrics.csv", "outer_evaluation_receipts.json"):
        if not (directory / required).is_file():
            raise FileNotFoundError(directory / required)


def _difference(left, right):
    if left is None or right is None:
        return None
    return float(left) - float(right)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit(path: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None

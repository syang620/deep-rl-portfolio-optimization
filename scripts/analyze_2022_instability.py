"""Build the frozen WF4 2022-2023 attribution bundle for PR 18."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from portfolio_rl.config.loader import load_yaml
from portfolio_rl.data.feature_store import PortfolioFeatureStore
from portfolio_rl.data.walk_forward import (
    load_outer_evaluation_dataset,
    load_training_selection_dataset,
)
from portfolio_rl.evaluation.attribution import (
    BuyAndHoldTargetPolicy,
    ConstantTargetPolicy,
    build_asset_contributions,
    build_exposure_paths,
    calculate_seed_disagreement,
    largest_active_weeks,
    largest_target_change_windows,
)
from portfolio_rl.evaluation.backtest import BacktestResult, run_weight_policy_backtest
from portfolio_rl.evaluation.execution_stress import reconstruct_daily_positions
from portfolio_rl.policies.ensemble_policy import MeanWeightEnsemblePolicy
from portfolio_rl.policies.overlays import PartialRebalancePolicy
from portfolio_rl.policies.sb3_policy import load_sb3_weight_policy
from portfolio_rl.training.walk_forward_runner import (
    load_walk_forward_campaign_config,
    selection_output_dir,
    verify_selection_freeze,
)

REFERENCE = "equal_weight_weekly"
ATTRIBUTION_STRATEGIES = [
    "ensemble_alpha_0.25",
    "ensemble_alpha_0.50",
    "ensemble_alpha_1.00",
]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/research/phase3_execution_stress.yaml"
    )
    parser.add_argument("--root", default=".")
    parser.add_argument("--execution-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args(argv)
    output = run_regime_attribution_study(
        config_path=args.config,
        root=args.root,
        execution_dir=args.execution_dir,
        output_dir=args.output_dir,
    )
    print(f"regime_attribution: {output}")


def run_regime_attribution_study(
    *,
    config_path: str | Path,
    root: str | Path = ".",
    execution_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> Path:
    """Build immutable WF4 attribution from verified PR17/PR18 inputs."""
    root_path = Path(root).resolve()
    config_path = _resolve(root_path, config_path)
    raw = load_yaml(config_path)
    campaign_id = str(raw["campaign_id"])
    wf_record = raw["walk_forward_config"]
    wf_path = _resolve(root_path, wf_record["path"])
    _require_hash(wf_path, wf_record["sha256"])
    wf_config = load_walk_forward_campaign_config(wf_path, root=root_path)
    execution_root = (
        _resolve(root_path, execution_dir)
        if execution_dir is not None
        else _resolve(root_path, raw["output_roots"]["execution_stress"])
        / campaign_id
    )
    execution_manifest_path = execution_root / "execution_stress_manifest.json"
    execution_manifest = _json(execution_manifest_path)
    _verify_execution_bundle(execution_root, execution_manifest)
    destination = (
        _resolve(root_path, output_dir)
        if output_dir is not None
        else _resolve(root_path, raw["output_roots"]["regime_attribution"])
        / campaign_id
    )
    if destination.exists():
        raise FileExistsError(f"regime-attribution output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    try:
        filters = {
            "fold_id": "WF4",
            "scenario_group": "cost_only",
            "cost_scenario": "flat_10_bps",
            "execution_delay_closes": 0,
        }
        nav = _filter(pd.read_parquet(execution_root / "nav_paths.parquet"), filters)
        costs = _filter(pd.read_parquet(execution_root / "costs.parquet"), filters)
        targets = _filter(pd.read_parquet(execution_root / "targets.parquet"), filters)
        positions = _filter(
            pd.read_parquet(execution_root / "daily_positions.parquet"), filters
        )
        members = _filter(
            pd.read_parquet(execution_root / "member_targets.parquet"), filters
        )
        store = PortfolioFeatureStore(
            load_outer_evaluation_dataset(wf_config.data_root / "WF4"),
            "outer_evaluation",
        )
        seed_positions, seed_nav = _load_pr17_seed_paths(
            wf_config=wf_config, store=store
        )
        positions = pd.concat([positions, seed_positions], ignore_index=True)
        nav = pd.concat([nav, seed_nav], ignore_index=True)
        attribution_config = raw["attribution"]
        groups = {
            str(name): [str(ticker) for ticker in tickers]
            for name, tickers in attribution_config["exposure_groups"].items()
        }
        expected_tickers = set(store.asset_order)
        configured_tickers = {ticker for tickers in groups.values() for ticker in tickers}
        if configured_tickers != expected_tickers:
            raise ValueError("attribution exposure groups must cover asset_order exactly")
        contributions = build_asset_contributions(
            positions=positions, nav=nav, reference_strategy=REFERENCE
        )
        exposures = build_exposure_paths(positions=positions, exposure_groups=groups)
        disagreement = calculate_seed_disagreement(members)
        largest = largest_active_weeks(
            nav=nav,
            costs=costs,
            strategies=ATTRIBUTION_STRATEGIES,
            reference_strategy=REFERENCE,
            count_each_tail=int(attribution_config["largest_active_weeks_each_tail"]),
        )
        largest = largest.merge(
            disagreement[
                [
                    "strategy",
                    "decision_step",
                    "median_pairwise_target_half_l1",
                    "dominant_asset_agreement",
                ]
            ],
            on=["strategy", "decision_step"],
            how="left",
            validate="many_to_one",
        )
        change_windows = largest_target_change_windows(
            targets=targets,
            strategies=ATTRIBUTION_STRATEGIES,
            count=int(attribution_config["largest_target_changes"]),
            radius=int(attribution_config["event_window_decisions"]),
        )
        event_turnover = _event_turnover(change_windows, costs, disagreement)
        static_results, static_weights = _static_mix_diagnostics(
            root_path=root_path,
            wf_config=wf_config,
            store=store,
            dynamic_targets=targets,
            execution_metrics=pd.read_csv(
                execution_root / "stress_metrics_by_fold.csv"
            ),
        )
        contribution_summary = _contribution_summary(contributions)
        exposure_summary = _exposure_summary(exposures)
        stabilization = _ensemble_stabilization(wf_config)
        pr15_context = _pr15_context(raw, root_path)
        outputs = {
            "contributions": temporary / "asset_contributions.parquet",
            "contribution_summary": temporary / "asset_contribution_summary.csv",
            "exposures": temporary / "exposure_paths.parquet",
            "exposure_summary": temporary / "exposure_summary.csv",
            "disagreement": temporary / "seed_disagreement.parquet",
            "largest": temporary / "largest_active_periods.csv",
            "events": temporary / "turnover_regime_events.csv",
            "static_results": temporary / "static_mix_results.csv",
            "static_weights": temporary / "static_weights.csv",
            "stabilization": temporary / "ensemble_stabilization.csv",
            "pr15": temporary / "pr15_context.csv",
            "report": temporary / "rate_hike_2022_report.md",
        }
        contributions.to_parquet(outputs["contributions"], index=False)
        contribution_summary.to_csv(outputs["contribution_summary"], index=False)
        exposures.to_parquet(outputs["exposures"], index=False)
        exposure_summary.to_csv(outputs["exposure_summary"], index=False)
        disagreement.to_parquet(outputs["disagreement"], index=False)
        largest.to_csv(outputs["largest"], index=False)
        event_turnover.to_csv(outputs["events"], index=False)
        static_results.to_csv(outputs["static_results"], index=False)
        static_weights.to_csv(outputs["static_weights"], index=False)
        stabilization.to_csv(outputs["stabilization"], index=False)
        pr15_context.to_csv(outputs["pr15"], index=False)
        outputs["report"].write_text(
            _format_report(
                contribution_summary=contribution_summary,
                exposure_summary=exposure_summary,
                largest=largest,
                event_turnover=event_turnover,
                static_results=static_results,
                stabilization=stabilization,
                pr15_context=pr15_context,
                disagreement=disagreement,
            ),
            encoding="utf-8",
        )
        manifest = {
            "schema_version": 1,
            "campaign_id": campaign_id,
            "artifact_directory": _manifest_path(destination, root_path),
            "created_at": datetime.now(UTC).isoformat(),
            "attribution_fold": "WF4",
            "evaluation_period": "2022-01-03 through 2023-12-29",
            "candidate_selected": False,
            "policy_retrained": False,
            "test_accessed": False,
            "development_data_label": "2024 consumed development/selection data",
            "asset_contribution_convention": (
                "daily arithmetic beginning-weight gross contribution; cost effect "
                "stored separately; daily exact, multi-period compounding residual disclosed"
            ),
            "static_control_contract": {
                "ex_ante_source": "WF4 inner validation 2021 executed targets",
                "oracle_source": "WF4 outer 2022-2023 executed targets; non-deployable",
                "execution": ["weekly_rebalance", "buy_and_hold"],
                "initialization": "equal_weight with initial trade cost",
                "transaction_cost_bps": 10.0,
            },
            "source_hashes": {
                "pr18_config": _source(config_path, root_path),
                "execution_stress_manifest": _source(
                    execution_manifest_path, root_path
                ),
                "walk_forward_config": _source(wf_path, root_path),
            },
            "output_sha256": {
                path.name: _sha256(path) for path in outputs.values()
            },
        }
        (temporary / "regime_attribution_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


def _load_pr17_seed_paths(*, wf_config, store):
    fold_root = wf_config.campaign_root / "evaluation" / "WF4" / "backtests"
    positions = []
    navs = []
    for strategy in [
        "ppo_seed_7",
        "ppo_seed_42",
        "ppo_seed_101",
        "ppo_seed_202",
        "ppo_seed_999",
        "five_seed_mean_target_ensemble",
    ]:
        root = fold_root / strategy
        result = BacktestResult(
            nav=pd.read_parquet(root / "nav.parquet"),
            weights_target=pd.read_parquet(root / "weights_target.parquet"),
            weights_drifted=pd.read_parquet(root / "weights_drifted.parquet"),
            trades=pd.read_parquet(root / "trades.parquet"),
            costs=pd.read_parquet(root / "costs.parquet"),
            metrics=_json(root / "metrics.json"),
        )
        frame = reconstruct_daily_positions(
            feature_store=store,
            backtest=result,
            strategy=strategy,
            rebalance_days=wf_config.rebalance_days,
        )
        frame["fold_id"] = "WF4"
        positions.append(frame)
        nav = result.nav.copy()
        nav["fold_id"] = "WF4"
        navs.append(nav)
    return pd.concat(positions, ignore_index=True), pd.concat(navs, ignore_index=True)


def _static_mix_diagnostics(
    *, root_path, wf_config, store, dynamic_targets, execution_metrics
):
    verified = {
        seed: verify_selection_freeze(
            selection_output_dir(
                wf_config, fold_id="WF4", seed=seed, pilot=False
            ),
            config=wf_config,
        )
        for seed in wf_config.seeds
    }
    factories = {
        seed: (
            lambda path=result.selected_model_path: load_sb3_weight_policy(
                path, action_temperature=0.5
            )
        )
        for seed, result in verified.items()
    }
    validation_store = PortfolioFeatureStore(
        load_training_selection_dataset(wf_config.data_root / "WF4"),
        "inner_validation",
    )
    hurdle = execution_metrics[
        (execution_metrics["fold_id"] == "WF4")
        & (execution_metrics["scenario_group"] == "cost_only")
        & (execution_metrics["cost_scenario"] == "flat_10_bps")
        & (execution_metrics["strategy"] == REFERENCE)
    ].iloc[0]
    result_rows = []
    weight_rows = []
    for strategy in ATTRIBUTION_STRATEGIES:
        alpha = float(strategy.rsplit("_", 1)[1])
        validation = run_weight_policy_backtest(
            feature_store=validation_store,
            policy=_overlay(factories, alpha),
            strategy=strategy,
            rebalance_frequency_trading_days=wf_config.rebalance_days,
            transaction_cost_bps=10.0,
            inverse_vol_lookback_trading_days=wf_config.inverse_vol_lookback,
        )
        ex_ante = _mean_target(validation.weights_target, validation_store.asset_order)
        live = dynamic_targets[dynamic_targets["strategy"] == strategy]
        oracle = _mean_target(live, store.asset_order)
        for source, target, deployable in (
            ("ex_ante_2021", ex_ante, True),
            ("oracle_2022_2023", oracle, False),
        ):
            for ticker, weight in zip(store.asset_order, target, strict=True):
                weight_rows.append(
                    {
                        "strategy": strategy,
                        "source": source,
                        "ticker": ticker,
                        "weight": float(weight),
                        "deployable": deployable,
                    }
                )
            for convention, policy in (
                ("weekly_rebalance", ConstantTargetPolicy(target)),
                ("buy_and_hold", BuyAndHoldTargetPolicy(target)),
            ):
                result = run_weight_policy_backtest(
                    feature_store=store,
                    policy=policy,
                    strategy=f"{strategy}_{source}_{convention}",
                    rebalance_frequency_trading_days=wf_config.rebalance_days,
                    transaction_cost_bps=10.0,
                    inverse_vol_lookback_trading_days=wf_config.inverse_vol_lookback,
                )
                result_rows.append(
                    {
                        "strategy": strategy,
                        "diagnostic": source,
                        "execution_convention": convention,
                        "deployable": deployable,
                        **result.metrics,
                        "active_return_vs_equal_weight": result.metrics["total_return"]
                        - hurdle["total_return"],
                        "active_sharpe_vs_equal_weight": result.metrics["sharpe_ratio"]
                        - hurdle["sharpe_ratio"],
                    }
                )
        live_metric = execution_metrics[
            (execution_metrics["fold_id"] == "WF4")
            & (execution_metrics["scenario_group"] == "cost_only")
            & (execution_metrics["cost_scenario"] == "flat_10_bps")
            & (execution_metrics["strategy"] == strategy)
        ].iloc[0]
        metric_keys = [
            "total_return",
            "cagr",
            "annualized_volatility",
            "sharpe_ratio",
            "sortino_ratio",
            "max_drawdown",
            "calmar_ratio",
            "average_weekly_turnover",
            "annualized_turnover",
            "transaction_cost_drag",
            "hit_rate",
            "best_month",
            "worst_month",
            "active_return_vs_equal_weight",
            "active_sharpe_vs_equal_weight",
        ]
        result_rows.append(
            {
                "strategy": strategy,
                "diagnostic": "dynamic_live",
                "execution_convention": "closed_loop",
                "deployable": True,
                **{key: live_metric[key] for key in metric_keys},
            }
        )
    del root_path
    return pd.DataFrame(result_rows), pd.DataFrame(weight_rows)


def _overlay(factories, alpha):
    ensemble = MeanWeightEnsemblePolicy(
        member_policies={
            f"seed_{seed}": factory() for seed, factory in factories.items()
        }
    )
    return PartialRebalancePolicy(base_policy=ensemble, alpha=alpha)


def _mean_target(frame, asset_order):
    pivot = frame.pivot(index="date", columns="ticker", values="target_weight")
    values = pivot.reindex(columns=asset_order).to_numpy(dtype=np.float64).mean(axis=0)
    return values / values.sum()


def _contribution_summary(contributions):
    frame = contributions.copy()
    frame["year"] = pd.to_datetime(frame["date"]).dt.year.astype(str)
    combined = frame.assign(year="2022-2023")
    return pd.concat([frame, combined]).groupby(
        ["strategy", "year", "ticker"], as_index=False
    )[["gross_return_contribution", "active_gross_contribution"]].sum()


def _exposure_summary(exposures):
    frame = exposures.copy()
    frame["year"] = pd.to_datetime(frame["date"]).dt.year.astype(str)
    combined = frame.assign(year="2022-2023")
    return pd.concat([frame, combined]).groupby(
        ["strategy", "year", "exposure_group"], as_index=False
    )["exposure"].mean()


def _event_turnover(windows, costs, disagreement):
    cost = costs.copy()
    cost["decision_step"] = cost.groupby(["fold_id", "strategy"]).cumcount()
    result = windows.merge(
        cost[["strategy", "decision_step", "date", "turnover", "transaction_cost_fraction"]],
        on=["strategy", "date"],
        how="left",
    )
    return result.merge(
        disagreement[
            [
                "strategy",
                "date",
                "median_pairwise_target_half_l1",
                "dominant_asset_agreement",
            ]
        ],
        on=["strategy", "date"],
        how="left",
    )


def _ensemble_stabilization(wf_config):
    metrics = pd.read_csv(wf_config.campaign_root / "aggregate" / "fold_metrics.csv")
    selected = metrics[
        (metrics["fold_id"] == "WF4")
        & (
            metrics["strategy"].str.startswith("ppo_seed_")
            | (metrics["strategy"] == "five_seed_mean_target_ensemble")
        )
    ].copy()
    return selected[
        [
            "strategy",
            "total_return",
            "sharpe_ratio",
            "max_drawdown",
            "average_weekly_turnover",
            "transaction_cost_drag",
        ]
    ]


def _pr15_context(raw, root):
    record = raw["pr15_dynamic_value_manifest"]
    manifest_path = _resolve(root, record["path"])
    _require_hash(manifest_path, record["sha256"])
    manifest = _json(manifest_path)
    results = pd.read_csv(
        _resolve(root, manifest["artifact_directory"]) / "diagnostic_results.csv"
    )
    return results[
        results["candidate"].isin(ATTRIBUTION_STRATEGIES)
        & results["diagnostic"].isin(
            ["dynamic_live", "ex_ante_static_2023", "oracle_static_2024"]
        )
    ].copy()


def _format_report(
    *, contribution_summary, exposure_summary, largest, event_turnover,
    static_results, stabilization, pr15_context, disagreement
):
    alpha = static_results[static_results["strategy"] == "ensemble_alpha_0.25"]
    dynamic = alpha[alpha["diagnostic"] == "dynamic_live"].iloc[0]
    ex_ante = alpha[
        (alpha["diagnostic"] == "ex_ante_2021")
        & (alpha["execution_convention"] == "weekly_rebalance")
    ].iloc[0]
    oracle = alpha[
        (alpha["diagnostic"] == "oracle_2022_2023")
        & (alpha["execution_convention"] == "weekly_rebalance")
    ].iloc[0]
    alpha_one = static_results[
        (static_results["strategy"] == "ensemble_alpha_1.00")
        & (static_results["diagnostic"] == "dynamic_live")
    ].iloc[0]
    if (
        ex_ante["total_return"] >= dynamic["total_return"]
        and oracle["total_return"] >= dynamic["total_return"]
    ):
        classification = "static mix / insufficient adaptation"
    elif dynamic["total_return"] > alpha_one["total_return"]:
        classification = "smoothing noisy overreaction"
    else:
        classification = "mixed evidence"
    drivers = contribution_summary[
        (contribution_summary["strategy"] == "ensemble_alpha_0.25")
        & (contribution_summary["year"] == "2022-2023")
    ].sort_values("active_gross_contribution")
    losses = ", ".join(
        f"{row.ticker} ({row.active_gross_contribution:.2%})"
        for row in drivers.head(3).itertuples()
    )
    gains = ", ".join(
        f"{row.ticker} ({row.active_gross_contribution:.2%})"
        for row in drivers.tail(3).sort_values(
            "active_gross_contribution", ascending=False
        ).itertuples()
    )
    seeds = stabilization[stabilization["strategy"].str.startswith("ppo_seed_")]
    ensemble = stabilization[
        stabilization["strategy"] == "five_seed_mean_target_ensemble"
    ].iloc[0]
    median_seed_return = float(seeds["total_return"].median())
    median_seed_turnover = float(seeds["average_weekly_turnover"].median())
    median_disagreement = float(
        disagreement[
            disagreement["strategy"] == "ensemble_alpha_0.25"
        ]["median_pairwise_target_half_l1"].median()
    )
    lines = [
        "# WF4 2022-2023 Root-Cause Attribution",
        "",
        "This report analyzes frozen fold-specific models. No policy is retrained and no candidate is selected.",
        "WF4 begins at NAV 1.0 and equal weight. Final-test access: **none**.",
        "2024 remains consumed development/selection data; PR 15 results appear only as development context.",
        "",
        "## Alpha 0.25 static-versus-dynamic diagnosis",
        "",
        f"- Dynamic total return: {dynamic['total_return']:.4%}.",
        f"- Past-only 2021 static weekly return: {ex_ante['total_return']:.4%}.",
        f"- Non-deployable 2022-2023 oracle static weekly return: {oracle['total_return']:.4%}.",
        f"- Evidence classification: **{classification}**.",
        (
            f"- The past-only static portfolio beat dynamic alpha 0.25 by "
            f"{ex_ante['total_return'] - dynamic['total_return']:.2%}; the oracle "
            f"static control beat it by {oracle['total_return'] - dynamic['total_return']:.2%}."
        ),
        (
            f"- Alpha 0.25 cut average weekly turnover from "
            f"{alpha_one['average_weekly_turnover']:.2%} to "
            f"{dynamic['average_weekly_turnover']:.2%} and improved maximum drawdown "
            f"by {dynamic['max_drawdown'] - alpha_one['max_drawdown']:.2%}, but "
            f"returned {dynamic['total_return'] - alpha_one['total_return']:.2%} less."
        ),
        (
            "- The evidence therefore favors a favorable strategic mix plus slower "
            "adaptation, not a claim that alpha 0.25 created return by avoiding every "
            "noisy overreaction. Its clearer benefit in WF4 was turnover and drawdown."
        ),
        f"- Largest active asset detractors: {losses}.",
        f"- Largest active asset contributors: {gains}.",
        (
            f"- Median alpha-0.25 cross-seed target disagreement was "
            f"{median_disagreement:.2%} half-L1. The executable ensemble returned "
            f"{ensemble['total_return']:.2%} versus a {median_seed_return:.2%} median "
            f"seed and reduced turnover from {median_seed_turnover:.2%} to "
            f"{ensemble['average_weekly_turnover']:.2%}."
        ),
        "",
        "Daily asset contributions use beginning-of-day weights and reconcile exactly to gross daily return. Transaction-cost effects are stored separately; annual sums are descriptive arithmetic attribution and include a compounding residual relative to total return.",
        "",
        "## Static-mix controls",
        "",
        _markdown(static_results),
        "",
        "## Average asset-class exposures",
        "",
        _markdown(exposure_summary[exposure_summary["strategy"].isin(ATTRIBUTION_STRATEGIES)]),
        "",
        "## Asset contributions",
        "",
        _markdown(contribution_summary[contribution_summary["strategy"].isin(ATTRIBUTION_STRATEGIES)]),
        "",
        "## Largest active gain and loss weeks",
        "",
        _markdown(largest),
        "",
        "## Policy-defined regime-change windows",
        "",
        _markdown(event_turnover),
        "",
        "## Ensemble stabilization versus individual seeds",
        "",
        _markdown(stabilization),
        "",
        "## PR 15 consumed-2024 context",
        "",
        _markdown(pr15_context),
        "",
        "Oracle controls are non-deployable. No alpha is selected or eliminated in PR 18.",
        "",
    ]
    return "\n".join(lines)


def _markdown(frame):
    if frame.empty:
        return "_No rows._"
    display = frame.copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(lambda value: f"{value:.6f}")
    headers = [str(value) for value in display.columns]
    rows = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    rows.extend(
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in display.itertuples(index=False, name=None)
    )
    return "\n".join(rows)


def _filter(frame, values):
    result = frame.copy()
    for column, value in values.items():
        result = result[result[column] == value]
    return result.reset_index(drop=True)


def _verify_execution_bundle(root, manifest):
    if manifest.get("test_accessed") is not False:
        raise ValueError("execution bundle test-access declaration is invalid")
    for filename, expected in manifest["output_sha256"].items():
        _require_hash(root / filename, expected)


def _json(path):
    loaded = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise TypeError(f"expected JSON object: {path}")
    return loaded


def _source(path, root):
    return {"path": _manifest_path(path, root), "sha256": _sha256(path)}


def _manifest_path(path, root):
    try:
        return str(Path(path).resolve().relative_to(root))
    except ValueError:
        return str(Path(path).resolve())


def _resolve(root, path):
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_hash(path, expected):
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(f"hash mismatch for {path}: expected={expected}, actual={actual}")


if __name__ == "__main__":
    main()

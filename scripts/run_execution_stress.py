"""Run the frozen PR 18 execution-stress matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from portfolio_rl.config.loader import load_yaml
from portfolio_rl.data.feature_store import PortfolioFeatureStore
from portfolio_rl.data.walk_forward import load_outer_evaluation_dataset
from portfolio_rl.evaluation.execution_stress import (
    ExecutionCostScenario,
    run_execution_stress_backtest,
)
from portfolio_rl.policies.baseline_policies import (
    EqualWeightWeeklyPolicy,
    InverseVolatilityPolicy,
    MomentumPolicy,
)
from portfolio_rl.policies.ensemble_policy import MeanWeightEnsemblePolicy
from portfolio_rl.policies.overlays import PartialRebalancePolicy
from portfolio_rl.policies.sb3_policy import load_sb3_weight_policy
from portfolio_rl.training.walk_forward_runner import (
    load_walk_forward_campaign_config,
    selection_output_dir,
    verify_selection_freeze,
)

PRIMARY_REFERENCE = "equal_weight_weekly"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/research/phase3_execution_stress.yaml"
    )
    parser.add_argument("--root", default=".")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args(argv)
    output = run_execution_stress_study(
        config_path=args.config,
        root=args.root,
        output_dir=args.output_dir,
    )
    print(f"execution_stress: {output}")


def run_execution_stress_study(
    *,
    config_path: str | Path,
    root: str | Path = ".",
    output_dir: str | Path | None = None,
) -> Path:
    """Verify PR 17 freezes, execute stresses, and write an immutable bundle."""
    root_path = Path(root).resolve()
    config_path = _resolve(root_path, config_path)
    raw = load_yaml(config_path)
    if int(_required(raw, "schema_version")) != 1:
        raise ValueError("PR18 schema_version must be 1")
    campaign_id = _text(raw, "campaign_id")
    wf_record = _mapping(raw, "walk_forward_config")
    wf_path = _resolve(root_path, _text(wf_record, "path"))
    _require_hash(wf_path, _text(wf_record, "sha256"))
    pr17_record = _mapping(raw, "pr17_aggregation_manifest")
    pr17_path = _resolve(root_path, _text(pr17_record, "path"))
    _require_hash(pr17_path, _text(pr17_record, "sha256"))
    wf_config = load_walk_forward_campaign_config(wf_path, root=root_path)
    strategies = [str(value) for value in _list(raw, "primary_strategies")]
    expected = [
        "ensemble_alpha_0.25",
        "ensemble_alpha_0.50",
        "ensemble_alpha_1.00",
        PRIMARY_REFERENCE,
        "inverse_volatility",
        "momentum_63d_top3_equal_weight",
    ]
    if strategies != expected:
        raise ValueError(f"primary_strategies must equal {expected}")
    scenarios = _cost_scenarios(raw)
    _validate_scenario_contract(raw)
    scenario_specs = [
        ("delay_only", "flat_10_bps", 1, strategies),
        *[
            ("cost_only", name, 0, strategies)
            for name in (
                "flat_10_bps",
                "flat_25_bps",
                "flat_50_bps",
                "asset_tier",
            )
        ],
        (
            "joint",
            "asset_tier",
            1,
            ["ensemble_alpha_0.25", PRIMARY_REFERENCE],
        ),
    ]
    destination = (
        _resolve(root_path, output_dir)
        if output_dir is not None
        else _resolve(
            root_path,
            _text(_mapping(raw, "output_roots"), "execution_stress"),
        )
        / campaign_id
    )
    if destination.exists():
        raise FileExistsError(f"execution-stress output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    try:
        metric_rows = []
        nav_frames = []
        trade_frames = []
        cost_frames = []
        target_frames = []
        position_frames = []
        audit_frames = []
        member_frames = []
        freeze_records: dict[str, dict[str, Any]] = {}
        for fold_id in wf_config.folds:
            verified = {
                seed: verify_selection_freeze(
                    selection_output_dir(
                        wf_config,
                        fold_id=fold_id,
                        seed=seed,
                        pilot=False,
                    ),
                    config=wf_config,
                )
                for seed in wf_config.seeds
            }
            freeze_records[fold_id] = {
                str(seed): {
                    "freeze_sha256": _sha256(result.freeze_path),
                    "model_sha256": _sha256(result.selected_model_path),
                    "selected_model_path": _manifest_path(
                        result.selected_model_path, root_path
                    ),
                }
                for seed, result in verified.items()
            }
            factories = {
                seed: (
                    lambda path=result.selected_model_path: load_sb3_weight_policy(
                        path, action_temperature=0.5
                    )
                )
                for seed, result in verified.items()
            }
            store = PortfolioFeatureStore(
                load_outer_evaluation_dataset(wf_config.data_root / fold_id),
                "outer_evaluation",
            )
            if store.date_at(store.n_rows - 1) >= pd.Timestamp("2024-01-01"):
                raise ValueError("PR18 execution folds must end before 2024")
            for group, scenario_name, delay, group_strategies in scenario_specs:
                for strategy in group_strategies:
                    policy, ensemble = _fresh_policy(
                        strategy, factories, store, wf_config
                    )
                    path = run_execution_stress_backtest(
                        feature_store=store,
                        policy=policy,
                        strategy=strategy,
                        cost_scenario=scenarios[scenario_name],
                        execution_delay_closes=delay,
                        rebalance_frequency_trading_days=wf_config.rebalance_days,
                        inverse_vol_lookback_trading_days=(
                            wf_config.momentum_lookback
                            if strategy == "momentum_63d_top3_equal_weight"
                            else wf_config.inverse_vol_lookback
                        ),
                    )
                    if group == "cost_only" and scenario_name == "flat_10_bps":
                        _assert_pr17_reconciliation(
                            path.backtest.metrics,
                            wf_config.campaign_root,
                            fold_id,
                            strategy,
                        )
                    labels = {
                        "fold_id": fold_id,
                        "scenario_group": group,
                        "cost_scenario": scenario_name,
                        "execution_delay_closes": delay,
                        "strategy": strategy,
                    }
                    gross_return = _gross_total_return(path.backtest)
                    metric_rows.append(
                        {
                            **labels,
                            **path.backtest.metrics,
                            "gross_total_return": gross_return,
                            "net_cost_return_impact": float(
                                path.backtest.metrics["total_return"]
                                - gross_return
                            ),
                        }
                    )
                    for frame, collection in (
                        (path.backtest.nav, nav_frames),
                        (path.backtest.trades, trade_frames),
                        (path.backtest.costs, cost_frames),
                        (path.backtest.weights_target, target_frames),
                        (path.daily_positions, position_frames),
                        (path.execution_audit, audit_frames),
                    ):
                        collection.append(_label(frame, labels))
                    if ensemble is not None:
                        member_frames.append(
                            _member_targets(
                                ensemble,
                                labels=labels,
                                asset_order=store.asset_order,
                            )
                        )
        metrics = _add_active_metrics(pd.DataFrame(metric_rows))
        delay_results = _delay_results(metrics)
        cost_results = metrics[metrics["scenario_group"] == "cost_only"].copy()
        asset_cost_results = cost_results[
            cost_results["cost_scenario"] == "asset_tier"
        ].copy()
        summary = _summarize(metrics)
        outputs = {
            "metrics": temporary / "stress_metrics_by_fold.csv",
            "summary": temporary / "stress_summary.csv",
            "delay": temporary / "delay_results.csv",
            "cost": temporary / "cost_results.csv",
            "asset_cost": temporary / "asset_cost_results.csv",
            "nav": temporary / "nav_paths.parquet",
            "trades": temporary / "trades.parquet",
            "costs": temporary / "costs.parquet",
            "targets": temporary / "targets.parquet",
            "positions": temporary / "daily_positions.parquet",
            "audit": temporary / "execution_audit.parquet",
            "members": temporary / "member_targets.parquet",
            "report": temporary / "execution_stress_report.md",
        }
        metrics.to_csv(outputs["metrics"], index=False)
        summary.to_csv(outputs["summary"], index=False)
        delay_results.to_csv(outputs["delay"], index=False)
        cost_results.to_csv(outputs["cost"], index=False)
        asset_cost_results.to_csv(outputs["asset_cost"], index=False)
        for key, frames in (
            ("nav", nav_frames),
            ("trades", trade_frames),
            ("costs", cost_frames),
            ("targets", target_frames),
            ("positions", position_frames),
            ("audit", audit_frames),
            ("members", member_frames),
        ):
            pd.concat(frames, ignore_index=True).to_parquet(outputs[key], index=False)
        outputs["report"].write_text(
            _format_report(metrics, summary, delay_results), encoding="utf-8"
        )
        output_hashes = {
            path.name: _sha256(path)
            for path in outputs.values()
        }
        manifest = {
            "schema_version": 1,
            "campaign_id": campaign_id,
            "artifact_directory": _manifest_path(destination, root_path),
            "created_at": datetime.now(UTC).isoformat(),
            "candidate_selected": False,
            "policy_retrained": False,
            "test_accessed": False,
            "latest_evaluation_date": "2023-12-31",
            "primary_initialization": "equal_weight",
            "turnover_convention": "half_l1",
            "scenario_contract": {
                "delay_only": "one close, flat 10 bps, all primary strategies",
                "cost_only": "no delay, flat 10/25/50 bps and asset tiers",
                "joint": "one close plus asset tiers, alpha 0.25 and equal weight",
            },
            "source_hashes": {
                "pr18_config": _source(config_path, root_path),
                "walk_forward_config": _source(wf_path, root_path),
                "pr17_aggregation_manifest": _source(pr17_path, root_path),
            },
            "fold_selection_freezes": freeze_records,
            "output_sha256": output_hashes,
        }
        (temporary / "execution_stress_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


def _fresh_policy(strategy, factories, store, config):
    if strategy.startswith("ensemble_alpha_"):
        alpha = float(strategy.rsplit("_", 1)[1])
        ensemble = MeanWeightEnsemblePolicy(
            member_policies={
                f"seed_{seed}": factory() for seed, factory in factories.items()
            }
        )
        return PartialRebalancePolicy(base_policy=ensemble, alpha=alpha), ensemble
    if strategy == PRIMARY_REFERENCE:
        return EqualWeightWeeklyPolicy(store.n_assets), None
    if strategy == "inverse_volatility":
        return InverseVolatilityPolicy(store.n_assets), None
    if strategy == "momentum_63d_top3_equal_weight":
        return MomentumPolicy(store.n_assets, top_k=config.momentum_top_k), None
    raise ValueError(f"unsupported PR18 strategy: {strategy}")


def _cost_scenarios(raw):
    flat = _mapping(raw, "flat_cost_bps")
    tiers = _mapping(raw, "asset_cost_bps")
    return {
        name: ExecutionCostScenario(name=name, flat_bps=float(value))
        for name, value in flat.items()
    } | {
        "asset_tier": ExecutionCostScenario(
            name="asset_tier",
            asset_bps={str(key): float(value) for key, value in tiers.items()},
        )
    }


def _validate_scenario_contract(raw):
    scenarios = _mapping(raw, "execution_scenarios")
    expected = {
        "delay_only": {
            "delay_closes": 1,
            "cost_scenario": "flat_10_bps",
        },
        "cost_only": {
            "delay_closes": 0,
            "cost_scenarios": [
                "flat_10_bps",
                "flat_25_bps",
                "flat_50_bps",
                "asset_tier",
            ],
        },
        "joint": {
            "delay_closes": 1,
            "cost_scenario": "asset_tier",
            "strategies": ["ensemble_alpha_0.25", PRIMARY_REFERENCE],
        },
    }
    if scenarios != expected:
        raise ValueError("execution_scenarios do not match the frozen PR18 matrix")


def _assert_pr17_reconciliation(
    metrics, campaign_root, fold_id, strategy
):
    path = (
        campaign_root
        / "evaluation"
        / fold_id
        / "backtests"
        / strategy
        / "metrics.json"
    )
    expected = json.loads(path.read_text(encoding="utf-8"))
    if metrics != expected:
        raise AssertionError(
            f"PR18 no-delay flat-10 metrics do not reconcile for {fold_id}/{strategy}"
        )


def _member_targets(ensemble, *, labels, asset_order):
    rows = []
    for record in ensemble.member_target_records:
        for ticker, target, current in zip(
            asset_order, record.weights, record.live_current_weights, strict=True
        ):
            rows.append(
                {
                    **labels,
                    "date": record.date,
                    "decision_step": record.decision_step,
                    "member": record.member,
                    "ticker": ticker,
                    "target_weight": target,
                    "live_current_weight": current,
                }
            )
    return pd.DataFrame(rows)


def _label(frame, labels):
    result = frame.copy()
    for key, value in labels.items():
        if key != "strategy":
            result[key] = value
    return result


def _gross_total_return(backtest):
    cost_multiplier = float(
        np.prod(1.0 - backtest.costs["transaction_cost_fraction"].to_numpy())
    )
    return float((1.0 + backtest.metrics["total_return"]) / cost_multiplier - 1.0)


def _add_active_metrics(metrics):
    reference = metrics[metrics["strategy"] == PRIMARY_REFERENCE].set_index(
        ["fold_id", "scenario_group", "cost_scenario", "execution_delay_closes"]
    )
    rows = []
    for _, row in metrics.iterrows():
        key = (
            row["fold_id"],
            row["scenario_group"],
            row["cost_scenario"],
            row["execution_delay_closes"],
        )
        hurdle = reference.loc[key]
        rows.append(
            {
                **row.to_dict(),
                "active_return_vs_equal_weight": row["total_return"]
                - hurdle["total_return"],
                "active_sharpe_vs_equal_weight": _difference(
                    row["sharpe_ratio"], hurdle["sharpe_ratio"]
                ),
                "drawdown_difference_vs_equal_weight": row["max_drawdown"]
                - hurdle["max_drawdown"],
            }
        )
    return pd.DataFrame(rows)


def _delay_results(metrics):
    delayed = metrics[metrics["scenario_group"] == "delay_only"].copy()
    reference = metrics[
        (metrics["scenario_group"] == "cost_only")
        & (metrics["cost_scenario"] == "flat_10_bps")
    ][
        [
            "fold_id",
            "strategy",
            "active_return_vs_equal_weight",
            "active_sharpe_vs_equal_weight",
            "sharpe_ratio",
            "average_weekly_turnover",
        ]
    ].rename(
        columns={
            "active_return_vs_equal_weight": "no_delay_active_return",
            "active_sharpe_vs_equal_weight": "no_delay_active_sharpe",
            "sharpe_ratio": "no_delay_sharpe",
            "average_weekly_turnover": "no_delay_turnover",
        }
    )
    result = delayed.merge(reference, on=["fold_id", "strategy"], validate="one_to_one")
    result["active_return_degradation"] = (
        result["no_delay_active_return"] - result["active_return_vs_equal_weight"]
    )
    result["active_sharpe_degradation"] = (
        result["no_delay_active_sharpe"] - result["active_sharpe_vs_equal_weight"]
    )
    result["sharpe_degradation"] = result["no_delay_sharpe"] - result["sharpe_ratio"]
    result["turnover_change"] = (
        result["average_weekly_turnover"] - result["no_delay_turnover"]
    )
    return result


def _summarize(metrics):
    rows = []
    for keys, group in metrics.groupby(
        ["scenario_group", "cost_scenario", "execution_delay_closes", "strategy"],
        sort=True,
    ):
        rows.append(
            {
                "scenario_group": keys[0],
                "cost_scenario": keys[1],
                "execution_delay_closes": keys[2],
                "strategy": keys[3],
                "positive_active_return_fold_count": int(
                    (group["active_return_vs_equal_weight"] > 0.0).sum()
                ),
                "median_active_return": group[
                    "active_return_vs_equal_weight"
                ].median(),
                "median_active_sharpe": group[
                    "active_sharpe_vs_equal_weight"
                ].median(),
                "median_net_return": group["total_return"].median(),
                "median_gross_return": group["gross_total_return"].median(),
                "median_sharpe": group["sharpe_ratio"].median(),
                "median_max_drawdown": group["max_drawdown"].median(),
                "median_turnover": group["average_weekly_turnover"].median(),
                "median_cost_drag": group["transaction_cost_drag"].median(),
                "median_max_drawdown_difference": group[
                    "drawdown_difference_vs_equal_weight"
                ].median(),
            }
        )
    return pd.DataFrame(rows)


def _format_report(metrics, summary, delay):
    alpha = summary[
        (summary["scenario_group"] == "delay_only")
        & (summary["strategy"] == "ensemble_alpha_0.25")
    ].iloc[0]
    gate = float(alpha["median_active_return"]) > 0.0
    lines = [
        "# PR 18 Execution Stress",
        "",
        "Frozen PR 17 fold-specific policies were evaluated without retraining or selection.",
        "Every fold and scenario starts from NAV 1.0 and equal weight. Final-test access: **none**.",
        "",
        "## Alpha 0.25 delay gate",
        "",
        f"- Positive active-return folds after delay: {int(alpha['positive_active_return_fold_count'])}/4.",
        f"- Median delayed active return: {float(alpha['median_active_return']):.4%}.",
        f"- Required positive-median gate: **{'PASS' if gate else 'FAIL'}**.",
        "",
        "## Scenario summary",
        "",
        _markdown(summary),
        "",
        "## Delay results by fold",
        "",
        _markdown(delay[
            [
                "fold_id",
                "strategy",
                "active_return_vs_equal_weight",
                "active_return_degradation",
                "sharpe_degradation",
                "turnover_change",
            ]
        ]),
        "",
        "Cost tiers are versioned evaluation scenarios, not validated market estimates.",
        "Alpha 0.75 is retained only as PR 17 context. No candidate is selected or eliminated.",
        "2024 remains consumed development/selection data and is not part of these folds.",
        "",
    ]
    del metrics
    return "\n".join(lines)


def _difference(left, right):
    return None if left is None or right is None else float(left - right)


def _markdown(frame):
    if frame.empty:
        return "_No rows._"
    display = frame.copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(lambda value: f"{value:.6f}")
    headers = [str(value) for value in display.columns]
    rows = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    rows.extend(
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in display.itertuples(index=False, name=None)
    )
    return "\n".join(rows)


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


def _required(mapping, key):
    if key not in mapping:
        raise ValueError(f"missing PR18 config key: {key}")
    return mapping[key]


def _mapping(mapping, key):
    value = _required(mapping, key)
    if not isinstance(value, dict):
        raise TypeError(f"PR18 config key must be a mapping: {key}")
    return value


def _list(mapping, key):
    value = _required(mapping, key)
    if not isinstance(value, list):
        raise TypeError(f"PR18 config key must be a list: {key}")
    return value


def _text(mapping, key):
    value = str(_required(mapping, key)).strip()
    if not value:
        raise ValueError(f"PR18 config key must not be empty: {key}")
    return value


if __name__ == "__main__":
    main()

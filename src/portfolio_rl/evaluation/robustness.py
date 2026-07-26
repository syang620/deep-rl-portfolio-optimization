"""Validation-only cost and regime robustness for selected configurations."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from math import isclose, isfinite
from pathlib import Path
from typing import Any

import pandas as pd

from portfolio_rl.config.loader import load_env_config, load_phase3_evaluation_config
from portfolio_rl.config.schemas import RegimeWindowConfig
from portfolio_rl.data.dataset import PortfolioDataset, load_portfolio_dataset
from portfolio_rl.data.feature_store import PortfolioFeatureStore
from portfolio_rl.evaluation.backtest import run_weight_policy_backtest
from portfolio_rl.policies.sb3_policy import load_sb3_weight_policy

REQUIRED_REGISTRY_COLUMNS = [
    "run_id",
    "experiment_name",
    "git_commit",
    "seed",
    "total_timesteps",
    "action_temperature",
    "ent_coef",
    "selection_eligible",
    "eligibility_issues",
    "selection_checkpoint",
    "selection_model_path",
    "manifest_path",
]
SELECTION_CHECKPOINTS = {"best_checkpoint", "final_endpoint"}
OVERRIDE_REGISTRY_COLUMNS = {
    "env.action_temperature": "action_temperature",
    "ppo.ent_coef": "ent_coef",
}
ROBUSTNESS_METRICS = [
    "total_return",
    "cagr",
    "sharpe_ratio",
    "max_drawdown",
    "average_weekly_turnover",
    "annualized_turnover",
    "transaction_cost_drag",
]
RESULT_COLUMNS = [
    "configuration_id",
    "experiment_name",
    "run_id",
    "seed",
    "selection_checkpoint",
    "selection_model_path",
    "transaction_cost_bps",
    *ROBUSTNESS_METRICS,
]
SUMMARY_COLUMNS = [
    "configuration_id",
    "experiment_name",
    "transaction_cost_bps",
    "seed_count",
    *[
        f"{metric}_{statistic}"
        for metric in ROBUSTNESS_METRICS
        for statistic in ("median", "std", "min", "max")
    ],
    "total_return_median_delta_vs_baseline",
    "sharpe_ratio_median_delta_vs_baseline",
    "max_drawdown_median_delta_vs_baseline",
    "transaction_cost_drag_median_delta_vs_baseline",
]
REGIME_RESULT_COLUMNS = [
    "configuration_id",
    "experiment_name",
    "regime_name",
    "configured_start_date",
    "configured_end_date",
    "actual_start_date",
    "actual_end_date",
    "split",
    "in_sample",
    "full_split_window",
    "run_id",
    "seed",
    "selection_checkpoint",
    "selection_model_path",
    "transaction_cost_bps",
    *ROBUSTNESS_METRICS,
]
REGIME_SUMMARY_COLUMNS = [
    "configuration_id",
    "experiment_name",
    "regime_name",
    "configured_start_date",
    "configured_end_date",
    "actual_start_date",
    "actual_end_date",
    "split",
    "in_sample",
    "full_split_window",
    "transaction_cost_bps",
    "seed_count",
    *[
        f"{metric}_{statistic}"
        for metric in ROBUSTNESS_METRICS
        for statistic in ("median", "std", "min", "max")
    ],
]


@dataclass(frozen=True)
class RobustnessResult:
    """Written transaction-cost and regime robustness artifacts."""

    outputs: dict[str, Path]
    results: pd.DataFrame
    summary: pd.DataFrame
    regime_results: pd.DataFrame
    regime_summary: pd.DataFrame
    manifest: dict[str, Any]


def run_transaction_cost_robustness(
    *,
    selected_configuration_path: str | Path,
    registry_path: str | Path,
    evaluation_config_path: str | Path,
    output_dir: str | Path,
    root: str | Path = ".",
) -> RobustnessResult:
    """Run selected checkpoints across configured costs and regime windows."""
    root_path = Path(root)
    selected_path = _resolve_path(root_path, selected_configuration_path)
    resolved_registry_path = _resolve_path(root_path, registry_path)
    config_path = _resolve_path(root_path, evaluation_config_path)
    destination = _resolve_path(root_path, output_dir)

    selected = _read_json(selected_path)
    registry = _read_registry(resolved_registry_path)
    evaluation_config = load_phase3_evaluation_config(config_path)
    _validate_selected_configuration(selected)
    selected_runs = _resolve_selected_runs(
        selected=selected,
        registry=registry,
        root=root_path,
    )
    cost_grid = sorted(
        {float(value) for value in evaluation_config.robustness.transaction_cost_bps}
    )
    baseline_cost_bps = _baseline_transaction_cost(selected_runs)
    if not any(isclose(value, baseline_cost_bps) for value in cost_grid):
        raise ValueError(
            "robustness cost grid must include the selected environment cost: "
            f"{baseline_cost_bps}"
        )

    dataset = load_portfolio_dataset(root_path)
    validation_store = PortfolioFeatureStore(
        dataset,
        split=evaluation_config.validation.split,
    )
    regime_windows, skipped_regimes = _resolve_regime_windows(
        dataset=dataset,
        windows=evaluation_config.robustness.regime_windows,
        rebalance_frequency_trading_days=_rebalance_frequency(selected_runs),
    )
    rows: list[dict[str, Any]] = []
    regime_rows: list[dict[str, Any]] = []
    for run in selected_runs:
        policy = load_sb3_weight_policy(
            run["model_path"],
            action_temperature=run["action_temperature"],
        )
        for transaction_cost_bps in cost_grid:
            backtest = run_weight_policy_backtest(
                feature_store=validation_store,
                policy=policy,
                strategy=(f"ppo_seed_{run['seed']}_cost_{transaction_cost_bps:g}bps"),
                rebalance_frequency_trading_days=run[
                    "rebalance_frequency_trading_days"
                ],
                transaction_cost_bps=transaction_cost_bps,
            )
            metrics = _finite_metrics(backtest.metrics, run["run_id"])
            rows.append(
                {
                    "configuration_id": selected["configuration_id"],
                    "experiment_name": selected["experiment_name"],
                    "run_id": run["run_id"],
                    "seed": run["seed"],
                    "selection_checkpoint": run["selection_checkpoint"],
                    "selection_model_path": _display_path(
                        run["model_path"],
                        root_path,
                    ),
                    "transaction_cost_bps": transaction_cost_bps,
                    **metrics,
                }
            )
        for regime in regime_windows:
            backtest = run_weight_policy_backtest(
                feature_store=regime["feature_store"],
                policy=policy,
                strategy=f"ppo_seed_{run['seed']}_{regime['name']}",
                rebalance_frequency_trading_days=run[
                    "rebalance_frequency_trading_days"
                ],
                transaction_cost_bps=baseline_cost_bps,
            )
            metrics = _finite_metrics(backtest.metrics, run["run_id"])
            regime_rows.append(
                {
                    "configuration_id": selected["configuration_id"],
                    "experiment_name": selected["experiment_name"],
                    "regime_name": regime["name"],
                    "configured_start_date": regime["configured_start_date"],
                    "configured_end_date": regime["configured_end_date"],
                    "actual_start_date": regime["actual_start_date"],
                    "actual_end_date": regime["actual_end_date"],
                    "split": regime["split"],
                    "in_sample": regime["in_sample"],
                    "full_split_window": regime["full_split_window"],
                    "run_id": run["run_id"],
                    "seed": run["seed"],
                    "selection_checkpoint": run["selection_checkpoint"],
                    "selection_model_path": _display_path(
                        run["model_path"],
                        root_path,
                    ),
                    "transaction_cost_bps": baseline_cost_bps,
                    **metrics,
                }
            )

    results = (
        pd.DataFrame(rows, columns=RESULT_COLUMNS)
        .sort_values(
            ["seed", "transaction_cost_bps"],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )
    summary = aggregate_transaction_cost_results(
        results,
        baseline_cost_bps=baseline_cost_bps,
    )
    regime_results = (
        pd.DataFrame(regime_rows, columns=REGIME_RESULT_COLUMNS)
        .sort_values(
            ["regime_name", "seed"],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )
    regime_summary = aggregate_regime_results(regime_results)
    diagnostics = _diagnostics(
        results,
        regime_results=regime_results,
        expected_seed_count=len(selected_runs),
        expected_cost_count=len(cost_grid),
        expected_regime_count=len(regime_windows),
        baseline_cost_bps=baseline_cost_bps,
    )
    manifest = _robustness_manifest(
        selected=selected,
        selected_runs=selected_runs,
        selected_path=selected_path,
        registry_path=resolved_registry_path,
        config_path=config_path,
        root=root_path,
        cost_grid=cost_grid,
        baseline_cost_bps=baseline_cost_bps,
        diagnostics=diagnostics,
        transaction_cost_evaluation_count=len(results),
        regime_evaluation_count=len(regime_results),
        regime_windows=regime_windows,
        skipped_regimes=skipped_regimes,
    )
    destination.mkdir(parents=True, exist_ok=True)
    outputs = {
        "results_csv": destination / "transaction_cost_results.csv",
        "summary_csv": destination / "transaction_cost_summary.csv",
        "regime_results_csv": destination / "regime_results.csv",
        "regime_summary_csv": destination / "regime_summary.csv",
        "manifest": destination / "robustness_manifest.json",
        "report": destination / "robustness_report.md",
    }
    results.to_csv(outputs["results_csv"], index=False)
    summary.to_csv(outputs["summary_csv"], index=False)
    regime_results.to_csv(outputs["regime_results_csv"], index=False)
    regime_summary.to_csv(outputs["regime_summary_csv"], index=False)
    outputs["manifest"].write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    outputs["report"].write_text(
        _robustness_report(
            selected=selected,
            results=results,
            summary=summary,
            regime_results=regime_results,
            regime_summary=regime_summary,
            baseline_cost_bps=baseline_cost_bps,
            diagnostics=diagnostics,
            skipped_regimes=skipped_regimes,
        ),
        encoding="utf-8",
    )
    return RobustnessResult(
        outputs=outputs,
        results=results,
        summary=summary,
        regime_results=regime_results,
        regime_summary=regime_summary,
        manifest=manifest,
    )


def aggregate_transaction_cost_results(
    results: pd.DataFrame,
    *,
    baseline_cost_bps: float,
) -> pd.DataFrame:
    """Aggregate per-seed robustness metrics by transaction-cost level."""
    missing = [column for column in RESULT_COLUMNS if column not in results]
    if missing:
        raise ValueError(f"robustness results missing required columns: {missing}")
    if results.empty:
        raise ValueError("robustness results must not be empty")
    baseline_rows = results[
        results["transaction_cost_bps"].map(
            lambda value: isclose(float(value), baseline_cost_bps)
        )
    ]
    if baseline_rows.empty:
        raise ValueError(
            f"robustness results missing baseline cost: {baseline_cost_bps}"
        )

    rows = []
    for transaction_cost_bps, group in results.groupby(
        "transaction_cost_bps",
        sort=True,
    ):
        row: dict[str, Any] = {
            "configuration_id": str(group["configuration_id"].iloc[0]),
            "experiment_name": str(group["experiment_name"].iloc[0]),
            "transaction_cost_bps": float(transaction_cost_bps),
            "seed_count": int(group["seed"].nunique()),
        }
        for metric in ROBUSTNESS_METRICS:
            values = group[metric].astype(float)
            row[f"{metric}_median"] = float(values.median())
            row[f"{metric}_std"] = float(values.std(ddof=1))
            row[f"{metric}_min"] = float(values.min())
            row[f"{metric}_max"] = float(values.max())
        rows.append(row)

    summary = pd.DataFrame(rows)
    baseline_summary = summary[
        summary["transaction_cost_bps"].map(
            lambda value: isclose(float(value), baseline_cost_bps)
        )
    ].iloc[0]
    for metric in [
        "total_return",
        "sharpe_ratio",
        "max_drawdown",
        "transaction_cost_drag",
    ]:
        column = f"{metric}_median"
        summary[f"{column}_delta_vs_baseline"] = summary[column] - float(
            baseline_summary[column]
        )
    return summary.loc[:, SUMMARY_COLUMNS]


def aggregate_regime_results(results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-seed metrics by configured regime window."""
    missing = [column for column in REGIME_RESULT_COLUMNS if column not in results]
    if missing:
        raise ValueError(f"regime results missing required columns: {missing}")
    if results.empty:
        return pd.DataFrame(columns=REGIME_SUMMARY_COLUMNS)

    metadata_columns = [
        "configuration_id",
        "experiment_name",
        "regime_name",
        "configured_start_date",
        "configured_end_date",
        "actual_start_date",
        "actual_end_date",
        "split",
        "in_sample",
        "full_split_window",
        "transaction_cost_bps",
    ]
    rows = []
    for regime_name, group in results.groupby("regime_name", sort=True):
        del regime_name
        row: dict[str, Any] = {
            column: group[column].iloc[0] for column in metadata_columns
        }
        row["seed_count"] = int(group["seed"].nunique())
        for metric in ROBUSTNESS_METRICS:
            values = group[metric].astype(float)
            row[f"{metric}_median"] = float(values.median())
            row[f"{metric}_std"] = float(values.std(ddof=1))
            row[f"{metric}_min"] = float(values.min())
            row[f"{metric}_max"] = float(values.max())
        rows.append(row)
    return pd.DataFrame(rows).loc[:, REGIME_SUMMARY_COLUMNS]


def _resolve_regime_windows(
    *,
    dataset: PortfolioDataset,
    windows: list[RegimeWindowConfig],
    rebalance_frequency_trading_days: int,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    names = [window.name for window in windows]
    if len(set(names)) != len(names):
        raise ValueError("robustness regime window names must be unique")
    resolved = []
    skipped = []
    for window in windows:
        start = pd.Timestamp(window.start_date)
        end = pd.Timestamp(window.end_date)
        date_mask = (dataset.dates >= start) & (dataset.dates <= end)
        configured = {
            "name": window.name,
            "configured_start_date": window.start_date.isoformat(),
            "configured_end_date": window.end_date.isoformat(),
        }
        if not date_mask.any():
            skipped.append(configured | {"reason": "no_available_rows"})
            continue
        splits = set(dataset.splits[date_mask].astype(str))
        if "test" in splits:
            raise ValueError(
                f"robustness regime window touches test split: {window.name}"
            )
        if len(splits) != 1:
            raise ValueError(
                f"robustness regime window must remain within one split: {window.name}"
            )
        split = splits.pop()
        feature_store = PortfolioFeatureStore(
            dataset,
            split=split,
            start_date=window.start_date,
            end_date=window.end_date,
        )
        if feature_store.n_rows <= rebalance_frequency_trading_days:
            skipped.append(configured | {"reason": "insufficient_rows"})
            continue
        split_mask = dataset.splits == split
        split_dates = dataset.dates[split_mask]
        actual_start = feature_store.date_at(0)
        actual_end = feature_store.date_at(feature_store.n_rows - 1)
        resolved.append(
            configured
            | {
                "actual_start_date": actual_start.date().isoformat(),
                "actual_end_date": actual_end.date().isoformat(),
                "split": split,
                "in_sample": split == "train",
                "full_split_window": (
                    actual_start == pd.Timestamp(split_dates[0])
                    and actual_end == pd.Timestamp(split_dates[-1])
                ),
                "feature_store": feature_store,
            }
        )
    return resolved, skipped


def _validate_selected_configuration(selected: dict[str, Any]) -> None:
    if selected.get("validation_only") is not True:
        raise ValueError("selected configuration must be validation-only")
    if selected.get("test_split_used") is not False:
        raise ValueError("selected configuration must not use the test split")
    gate_results = selected.get("gate_results")
    if (
        not isinstance(gate_results, dict)
        or not gate_results
        or any(value is not True for value in gate_results.values())
    ):
        raise ValueError("selected configuration must pass all validation gates")
    required = [
        "configuration_id",
        "experiment_name",
        "total_timesteps",
        "overrides",
        "planned_seeds",
        "eligible_seeds",
    ]
    missing = [key for key in required if key not in selected]
    if missing:
        raise ValueError(f"selected configuration missing required fields: {missing}")
    planned_seeds = selected["planned_seeds"]
    eligible_seeds = selected["eligible_seeds"]
    if (
        not isinstance(planned_seeds, list)
        or not isinstance(eligible_seeds, list)
        or not planned_seeds
        or any(
            isinstance(seed, bool) or not isinstance(seed, int)
            for seed in planned_seeds + eligible_seeds
        )
        or len(set(planned_seeds)) != len(planned_seeds)
        or set(planned_seeds) != set(eligible_seeds)
    ):
        raise ValueError(
            "selected configuration must have unique, fully eligible seeds"
        )
    overrides = selected["overrides"]
    if not isinstance(overrides, dict):
        raise TypeError("selected configuration overrides must be a mapping")
    unsupported = sorted(set(overrides) - set(OVERRIDE_REGISTRY_COLUMNS))
    if unsupported:
        raise ValueError(
            "selected configuration has overrides unavailable in registry: "
            + ", ".join(unsupported)
        )


def _resolve_selected_runs(
    *,
    selected: dict[str, Any],
    registry: pd.DataFrame,
    root: Path,
) -> list[dict[str, Any]]:
    missing = [column for column in REQUIRED_REGISTRY_COLUMNS if column not in registry]
    if missing:
        raise ValueError(f"registry missing required columns: {missing}")
    eligible_seeds = [int(seed) for seed in selected["eligible_seeds"]]
    candidates = registry[
        (registry["experiment_name"] == selected["experiment_name"])
        & registry["seed"].isin(eligible_seeds)
    ].copy()
    duplicated = candidates[candidates["seed"].duplicated(keep=False)]
    if not duplicated.empty:
        raise ValueError("registry contains duplicate selected seeds")
    actual_seeds = set(candidates["seed"].astype(int))
    if actual_seeds != set(eligible_seeds):
        missing_seeds = sorted(set(eligible_seeds) - actual_seeds)
        raise ValueError(f"registry missing selected seeds: {missing_seeds}")

    expected_counts = selected.get("selection_checkpoint_counts")
    checkpoint_counts: dict[str, int] = {}
    runs = []
    for row in candidates.sort_values("seed", kind="mergesort").to_dict(
        orient="records"
    ):
        run_id = str(row["run_id"])
        if not _as_bool(row["selection_eligible"]):
            raise ValueError(f"selected registry run is ineligible: {run_id}")
        if int(row["total_timesteps"]) != int(selected["total_timesteps"]):
            raise ValueError(
                f"selected registry run has mismatched timesteps: {run_id}"
            )
        for override, expected_value in selected["overrides"].items():
            column = OVERRIDE_REGISTRY_COLUMNS[override]
            if not isclose(float(row[column]), float(expected_value)):
                raise ValueError(
                    f"selected registry run has mismatched override "
                    f"{override}: {run_id}"
                )
        checkpoint = str(row["selection_checkpoint"])
        if checkpoint not in SELECTION_CHECKPOINTS:
            raise ValueError(
                f"selected registry run has unsupported checkpoint: {run_id}"
            )
        checkpoint_counts[checkpoint] = checkpoint_counts.get(checkpoint, 0) + 1
        model_path = _resolve_path(root, str(row["selection_model_path"]))
        manifest_path = _resolve_path(root, str(row["manifest_path"]))
        env_path = manifest_path.parent / "env.yaml"
        if not model_path.is_file():
            raise FileNotFoundError(f"selected model not found: {model_path}")
        if not manifest_path.is_file():
            raise FileNotFoundError(f"run manifest not found: {manifest_path}")
        if not env_path.is_file():
            raise FileNotFoundError(f"run environment config not found: {env_path}")
        expected_model_name = (
            "model.zip" if checkpoint == "final_endpoint" else "best_model.zip"
        )
        if model_path.name != expected_model_name:
            raise ValueError(f"selected model does not match checkpoint: {run_id}")
        env_config = load_env_config(env_path)
        if not isclose(
            env_config.action_temperature,
            float(row["action_temperature"]),
        ):
            raise ValueError(f"run environment action temperature mismatch: {run_id}")
        runs.append(
            {
                "run_id": run_id,
                "seed": int(row["seed"]),
                "git_commit": str(row["git_commit"]),
                "selection_checkpoint": checkpoint,
                "model_path": model_path,
                "manifest_path": manifest_path,
                "env_path": env_path,
                "action_temperature": env_config.action_temperature,
                "rebalance_frequency_trading_days": (
                    env_config.rebalance_frequency_trading_days
                ),
                "transaction_cost_bps": env_config.transaction_cost_bps,
            }
        )
    if isinstance(expected_counts, dict) and checkpoint_counts != expected_counts:
        raise ValueError(
            "registry checkpoint counts do not match selected configuration"
        )
    if len({run["git_commit"] for run in runs}) != 1:
        raise ValueError("selected registry runs must share one git commit")
    return runs


def _baseline_transaction_cost(selected_runs: list[dict[str, Any]]) -> float:
    costs = {float(run["transaction_cost_bps"]) for run in selected_runs}
    if len(costs) != 1:
        raise ValueError(
            "selected environments must share one baseline transaction cost"
        )
    return costs.pop()


def _rebalance_frequency(selected_runs: list[dict[str, Any]]) -> int:
    frequencies = {
        int(run["rebalance_frequency_trading_days"]) for run in selected_runs
    }
    if len(frequencies) != 1:
        raise ValueError("selected environments must share one rebalance frequency")
    return frequencies.pop()


def _finite_metrics(
    metrics: dict[str, float | None],
    run_id: str,
) -> dict[str, float]:
    values: dict[str, float] = {}
    for metric in ROBUSTNESS_METRICS:
        value = metrics.get(metric)
        if (
            value is None
            or isinstance(value, bool)
            or not isinstance(value, int | float)
            or not isfinite(float(value))
        ):
            raise ValueError(f"robustness backtest has invalid {metric}: {run_id}")
        values[metric] = float(value)
    return values


def _diagnostics(
    results: pd.DataFrame,
    *,
    regime_results: pd.DataFrame,
    expected_seed_count: int,
    expected_cost_count: int,
    expected_regime_count: int,
    baseline_cost_bps: float,
) -> dict[str, bool]:
    zero_cost = results[
        results["transaction_cost_bps"].map(lambda value: isclose(float(value), 0.0))
    ]
    nonincreasing = all(
        (
            group.sort_values("transaction_cost_bps")["total_return"].diff().dropna()
            <= 1e-12
        ).all()
        for _, group in results.groupby("seed")
    )
    return {
        "all_expected_cost_evaluations_complete": bool(
            len(results) == expected_seed_count * expected_cost_count
            and results["seed"].nunique() == expected_seed_count
            and results["transaction_cost_bps"].nunique() == expected_cost_count
        ),
        "zero_cost_drag_is_zero": bool(
            not zero_cost.empty
            and (zero_cost["transaction_cost_drag"].abs() <= 1e-12).all()
        ),
        "net_return_nonincreasing_with_cost": bool(nonincreasing),
        "all_expected_regime_evaluations_complete": bool(
            len(regime_results) == expected_seed_count * expected_regime_count
            and (
                expected_regime_count == 0
                or (
                    regime_results["seed"].nunique() == expected_seed_count
                    and regime_results["regime_name"].nunique() == expected_regime_count
                )
            )
        ),
        "full_validation_regime_matches_baseline_cost": (
            _validation_regime_matches_baseline(
                results,
                regime_results,
                baseline_cost_bps=baseline_cost_bps,
            )
        ),
    }


def _validation_regime_matches_baseline(
    cost_results: pd.DataFrame,
    regime_results: pd.DataFrame,
    *,
    baseline_cost_bps: float,
) -> bool:
    baseline = cost_results[
        cost_results["transaction_cost_bps"].map(
            lambda value: isclose(float(value), baseline_cost_bps)
        )
    ]
    validation = regime_results[
        (regime_results["split"] == "validation")
        & regime_results["full_split_window"].map(_as_bool)
    ]
    if baseline.empty or validation.empty:
        return False
    if set(baseline["run_id"]) != set(validation["run_id"]):
        return False
    paired = baseline.set_index("run_id").join(
        validation.set_index("run_id"),
        lsuffix="_cost",
        rsuffix="_regime",
        how="inner",
    )
    return bool(
        all(
            (paired[f"{metric}_cost"] - paired[f"{metric}_regime"]).abs().max() <= 1e-12
            for metric in ROBUSTNESS_METRICS
        )
    )


def _robustness_manifest(
    *,
    selected: dict[str, Any],
    selected_runs: list[dict[str, Any]],
    selected_path: Path,
    registry_path: Path,
    config_path: Path,
    root: Path,
    cost_grid: list[float],
    baseline_cost_bps: float,
    diagnostics: dict[str, bool],
    transaction_cost_evaluation_count: int,
    regime_evaluation_count: int,
    regime_windows: list[dict[str, Any]],
    skipped_regimes: list[dict[str, str]],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "analyses": [
            "transaction_cost_sensitivity",
            "regime_windows",
        ],
        "created_at": datetime.now(UTC).isoformat(),
        "configuration_id": selected["configuration_id"],
        "experiment_name": selected["experiment_name"],
        "validation_split": "validation",
        "test_split_used": False,
        "seeds": [run["seed"] for run in selected_runs],
        "transaction_cost_bps": cost_grid,
        "baseline_transaction_cost_bps": baseline_cost_bps,
        "evaluation_count": (
            transaction_cost_evaluation_count + regime_evaluation_count
        ),
        "transaction_cost_evaluation_count": (transaction_cost_evaluation_count),
        "regime_evaluation_count": regime_evaluation_count,
        "regime_windows": [
            {key: value for key, value in regime.items() if key != "feature_store"}
            for regime in regime_windows
        ],
        "skipped_regime_windows": skipped_regimes,
        "diagnostics": diagnostics,
        "sources": {
            "selected_configuration": _source(selected_path, root),
            "registry": _source(registry_path, root),
            "evaluation_config": _source(config_path, root),
        },
        "models": [
            {
                "run_id": run["run_id"],
                "seed": run["seed"],
                "git_commit": run["git_commit"],
                "selection_checkpoint": run["selection_checkpoint"],
                "model": _source(run["model_path"], root),
                "environment_config": _source(run["env_path"], root),
                "run_manifest": _source(run["manifest_path"], root),
            }
            for run in selected_runs
        ],
    }


def _robustness_report(
    *,
    selected: dict[str, Any],
    results: pd.DataFrame,
    summary: pd.DataFrame,
    regime_results: pd.DataFrame,
    regime_summary: pd.DataFrame,
    baseline_cost_bps: float,
    diagnostics: dict[str, bool],
    skipped_regimes: list[dict[str, str]],
) -> str:
    lines = [
        "# Robustness Report",
        "",
        (
            f"Selected configuration: `{selected['configuration_id']}` "
            f"from `{selected['experiment_name']}`."
        ),
        (
            f"Evaluated {results['seed'].nunique()} selected checkpoints across "
            f"{results['transaction_cost_bps'].nunique()} transaction-cost "
            f"levels on the validation split."
        ),
        "The test split was not accessed.",
        "",
        "## Transaction-Cost Sensitivity",
        "",
        (f"Baseline transaction cost: `{baseline_cost_bps:g}` basis points."),
        "",
        (
            "| Cost (bps) | Seeds | Median return | Median Sharpe | "
            "Median drawdown | Median turnover | Median cost drag |"
        ),
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary.to_dict(orient="records"):
        lines.append(
            f"| {row['transaction_cost_bps']:.4g} "
            f"| {int(row['seed_count'])} "
            f"| {row['total_return_median']:.6f} "
            f"| {row['sharpe_ratio_median']:.6f} "
            f"| {row['max_drawdown_median']:.6f} "
            f"| {row['average_weekly_turnover_median']:.6f} "
            f"| {row['transaction_cost_drag_median']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Regime Windows",
            "",
            (
                "Training-split windows are in-sample stress replays, not "
                "independent out-of-sample evidence."
            ),
            "",
            (
                "| Regime | Split | Interpretation | Actual dates | Seeds | "
                "Median return | Median CAGR | Median Sharpe | "
                "Median drawdown |"
            ),
            "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in regime_summary.to_dict(orient="records"):
        interpretation = (
            "in-sample stress replay"
            if _as_bool(row["in_sample"])
            else "out-of-sample validation"
        )
        lines.append(
            f"| {row['regime_name']} "
            f"| {row['split']} "
            f"| {interpretation} "
            f"| {row['actual_start_date']} to {row['actual_end_date']} "
            f"| {int(row['seed_count'])} "
            f"| {row['total_return_median']:.6f} "
            f"| {row['cagr_median']:.6f} "
            f"| {row['sharpe_ratio_median']:.6f} "
            f"| {row['max_drawdown_median']:.6f} |"
        )
    if skipped_regimes:
        lines.extend(["", "### Skipped Regimes", ""])
        lines.extend(
            f"- `{regime['name']}`: {regime['reason']}" for regime in skipped_regimes
        )
    lines.extend(
        [
            "",
            "## Diagnostics",
            "",
            *[
                f"- {name.replace('_', ' ').capitalize()}: "
                f"{'pass' if passed else 'fail'}"
                for name, passed in diagnostics.items()
            ],
            "",
            "## Per-Seed Return Sensitivity",
            "",
            "| Seed | Checkpoint | Cost (bps) | Total return | Sharpe |",
            "| ---: | --- | ---: | ---: | ---: |",
        ]
    )
    for row in results.to_dict(orient="records"):
        lines.append(
            f"| {int(row['seed'])} "
            f"| {row['selection_checkpoint']} "
            f"| {row['transaction_cost_bps']:.4g} "
            f"| {row['total_return']:.6f} "
            f"| {row['sharpe_ratio']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Per-Seed Regime Results",
            "",
            (
                "| Regime | Seed | Checkpoint | Split | Total return | "
                "CAGR | Sharpe | Drawdown |"
            ),
            "| --- | ---: | --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in regime_results.to_dict(orient="records"):
        lines.append(
            f"| {row['regime_name']} "
            f"| {int(row['seed'])} "
            f"| {row['selection_checkpoint']} "
            f"| {row['split']} "
            f"| {row['total_return']:.6f} "
            f"| {row['cagr']:.6f} "
            f"| {row['sharpe_ratio']:.6f} "
            f"| {row['max_drawdown']:.6f} |"
        )
    lines.extend(
        [
            "",
            (
                "These diagnostics describe sensitivity of the frozen "
                "configuration; they do not retune or re-rank it."
            ),
            "",
        ]
    )
    return "\n".join(lines)


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


def _display_path(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_path(root: Path, path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return root / candidate


def _as_bool(value: Any) -> bool:
    if pd.api.types.is_bool(value):
        return bool(value)
    if isinstance(value, str) and value.lower() in {"true", "false"}:
        return value.lower() == "true"
    raise ValueError(f"expected boolean value, received: {value!r}")

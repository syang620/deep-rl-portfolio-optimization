"""Counterfactual feature-sensitivity probes for selected PPO policies."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from portfolio_rl.config.loader import (
    load_phase3_evaluation_config,
    load_universe_config,
)
from portfolio_rl.data.dataset import PortfolioDataset, load_portfolio_dataset
from portfolio_rl.evaluation import robustness
from portfolio_rl.features.feature_spec import FeatureSpec, load_feature_spec
from portfolio_rl.policies.sb3_policy import load_sb3_weight_policy

REQUIRED_RISK_FEATURES = [
    "vix_z_21d",
    "vix_z_63d",
    "credit_spread_z_63d",
    "spy_vol_21d",
    "spy_drawdown_63d",
]
PROBE_SCENARIOS = {
    "spy_volatility": {
        "low_risk": {"spy_vol_21d": "low"},
        "high_risk": {"spy_vol_21d": "high"},
    },
    "global_risk": {
        "low_risk": {
            "vix_z_21d": "low",
            "vix_z_63d": "low",
            "credit_spread_z_63d": "low",
            "spy_vol_21d": "low",
            "spy_drawdown_63d": "high",
        },
        "high_risk": {
            "vix_z_21d": "high",
            "vix_z_63d": "high",
            "credit_spread_z_63d": "high",
            "spy_vol_21d": "high",
            "spy_drawdown_63d": "low",
        },
    },
}
OUTPUT_FILENAMES = {
    "allocations": "sensitivity_allocations.parquet",
    "summary": "sensitivity_summary.csv",
    "manifest": "sensitivity_manifest.json",
    "report": "sensitivity_report.md",
}


@dataclass(frozen=True)
class PolicySensitivityResult:
    """Written sensitivity artifacts and in-memory results."""

    outputs: dict[str, Path]
    allocations: pd.DataFrame
    summary: pd.DataFrame
    manifest: dict[str, Any]


def run_policy_sensitivity(
    *,
    selected_configuration_path: str | Path,
    registry_path: str | Path,
    diagnostics_dir: str | Path,
    evaluation_config_path: str | Path,
    universe_config_path: str | Path,
    feature_spec_path: str | Path,
    output_dir: str | Path,
    root: str | Path = ".",
) -> PolicySensitivityResult:
    """Replay selected actions and probe counterfactual risk-feature values."""
    root_path = Path(root)
    selected_path = _resolve_path(root_path, selected_configuration_path)
    resolved_registry_path = _resolve_path(root_path, registry_path)
    resolved_diagnostics_dir = _resolve_path(root_path, diagnostics_dir)
    resolved_evaluation_path = _resolve_path(root_path, evaluation_config_path)
    resolved_universe_path = _resolve_path(root_path, universe_config_path)
    resolved_feature_spec_path = _resolve_path(root_path, feature_spec_path)
    destination = _resolve_path(root_path, output_dir)

    selected = _read_json(selected_path)
    registry = _read_registry(resolved_registry_path)
    diagnostics_summary_path = resolved_diagnostics_dir / "allocation_summary.json"
    diagnostic_allocations_path = (
        resolved_diagnostics_dir / "allocation_by_regime.parquet"
    )
    diagnostic_summary = _read_json(diagnostics_summary_path)
    diagnostic_allocations = pd.read_parquet(diagnostic_allocations_path)
    evaluation_config = load_phase3_evaluation_config(resolved_evaluation_path)
    universe = load_universe_config(resolved_universe_path)
    feature_spec = load_feature_spec(resolved_feature_spec_path)

    robustness._validate_selected_configuration(selected)
    selected_runs = robustness._resolve_selected_runs(
        selected=selected,
        registry=registry,
        root=root_path,
    )
    _validate_diagnostic_inputs(
        selected=selected,
        diagnostic_summary=diagnostic_summary,
        allocations=diagnostic_allocations,
        selected_runs=selected_runs,
    )

    dataset = load_portfolio_dataset(root_path)
    _validate_feature_contract(
        dataset=dataset,
        feature_spec=feature_spec,
        universe_tickers=universe.tickers,
    )
    feature_indices = global_feature_indices(
        feature_spec,
        REQUIRED_RISK_FEATURES,
    )
    quantiles = train_feature_quantiles(
        dataset,
        feature_indices=feature_indices,
        low_quantile=(evaluation_config.policy_behavior.sensitivity_low_quantile),
        high_quantile=(evaluation_config.policy_behavior.sensitivity_high_quantile),
    )
    equity_like_tickers = {
        asset.ticker
        for asset in universe.assets
        if "equity" in asset.asset_class or asset.asset_class == "real_estate"
    }

    rows = []
    replay_errors = []
    date_indices = {
        pd.Timestamp(date): index for index, date in enumerate(dataset.dates)
    }
    for run in selected_runs:
        policy = load_sb3_weight_policy(
            run["model_path"],
            action_temperature=run["action_temperature"],
        )
        run_allocations = diagnostic_allocations[
            diagnostic_allocations["run_id"] == run["run_id"]
        ]
        if run_allocations.empty:
            raise ValueError(f"diagnostics missing selected run: {run['run_id']}")
        decision_columns = [
            "regime_name",
            "split",
            "in_sample",
            "date",
        ]
        for keys, decision in run_allocations.groupby(
            decision_columns,
            sort=True,
        ):
            metadata = dict(zip(decision_columns, keys, strict=True))
            date = pd.Timestamp(metadata["date"])
            if date not in date_indices:
                raise ValueError(f"decision date missing from dataset: {date}")
            dataset_index = date_indices[date]
            if str(dataset.splits[dataset_index]) != str(metadata["split"]):
                raise ValueError(f"decision split does not match dataset: {date}")
            ordered = decision.set_index("ticker").reindex(feature_spec.asset_order)
            if ordered.isna().any().any():
                raise ValueError(f"decision is missing configured assets: {date}")
            current_weights = ordered["pre_trade_weight"].to_numpy(dtype=np.float64)
            observed_weights = ordered["target_weight"].to_numpy(dtype=np.float64)
            market_features = dataset.market_features[dataset_index].astype(
                np.float64,
                copy=True,
            )
            observation = np.concatenate([market_features, current_weights]).astype(
                np.float32
            )
            policy_info = {"asset_order": feature_spec.asset_order}
            replayed = np.asarray(
                policy.target_weights(observation, policy_info),
                dtype=np.float64,
            )
            replay_error = float(np.max(np.abs(replayed - observed_weights)))
            replay_errors.append(
                {
                    "run_id": str(run["run_id"]),
                    "seed": int(run["seed"]),
                    "regime_name": str(metadata["regime_name"]),
                    "date": date.date().isoformat(),
                    "max_absolute_weight_error": replay_error,
                }
            )
            if replay_error > 1e-6:
                raise ValueError(
                    "observed action replay does not reconcile: "
                    f"{run['run_id']} {date.date()} error={replay_error}"
                )
            base_metadata = {
                "configuration_id": str(selected["configuration_id"]),
                "experiment_name": str(selected["experiment_name"]),
                "run_id": str(run["run_id"]),
                "seed": int(run["seed"]),
                "selection_checkpoint": str(run["selection_checkpoint"]),
                "regime_name": str(metadata["regime_name"]),
                "split": str(metadata["split"]),
                "in_sample": bool(metadata["in_sample"]),
                "date": date,
            }
            rows.extend(
                _allocation_rows(
                    metadata=base_metadata,
                    probe="observed",
                    scenario="observed",
                    tickers=feature_spec.asset_order,
                    weights=replayed,
                    observed_weights=observed_weights,
                    equity_like_tickers=equity_like_tickers,
                    overrides={},
                )
            )
            scenarios = build_counterfactual_observations(
                observation,
                feature_indices=feature_indices,
                quantiles=quantiles,
            )
            for (probe, scenario), scenario_values in scenarios.items():
                counterfactual = np.asarray(
                    policy.target_weights(scenario_values["observation"], policy_info),
                    dtype=np.float64,
                )
                _validate_weights(counterfactual, feature_spec.asset_order)
                rows.extend(
                    _allocation_rows(
                        metadata=base_metadata,
                        probe=probe,
                        scenario=scenario,
                        tickers=feature_spec.asset_order,
                        weights=counterfactual,
                        observed_weights=observed_weights,
                        equity_like_tickers=equity_like_tickers,
                        overrides=scenario_values["overrides"],
                    )
                )

    allocations = pd.DataFrame(rows).sort_values(
        ["regime_name", "seed", "date", "probe", "scenario", "ticker"],
        kind="mergesort",
        ignore_index=True,
    )
    summary = aggregate_sensitivity_results(
        allocations,
        material_weight_shift=(
            evaluation_config.policy_behavior.sensitivity_material_weight_shift
        ),
    )
    campaign_interpretation = interpret_sensitivity_results(summary)
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "configuration_id": str(selected["configuration_id"]),
        "experiment_name": str(selected["experiment_name"]),
        "validation_only": True,
        "test_split_used": False,
        "diagnostic_only": True,
        "seed_count": len(selected_runs),
        "seeds": [int(run["seed"]) for run in selected_runs],
        "feature_indices": feature_indices,
        "train_quantiles": quantiles,
        "scenario_definitions": PROBE_SCENARIOS,
        "replay_tolerance": 1e-6,
        "maximum_replay_error": max(
            row["max_absolute_weight_error"] for row in replay_errors
        ),
        "all_observed_actions_reconciled": True,
        "material_weight_shift": (
            evaluation_config.policy_behavior.sensitivity_material_weight_shift
        ),
        "campaign_interpretation": campaign_interpretation,
        "selected_models": [
            {
                "run_id": str(run["run_id"]),
                "seed": int(run["seed"]),
                "selection_checkpoint": str(run["selection_checkpoint"]),
                **_source(Path(run["model_path"]), root_path),
            }
            for run in selected_runs
        ],
        "sources": {
            "selected_configuration": _source(selected_path, root_path),
            "registry": _source(resolved_registry_path, root_path),
            "diagnostic_summary": _source(
                diagnostics_summary_path,
                root_path,
            ),
            "diagnostic_allocations": _source(
                diagnostic_allocations_path,
                root_path,
            ),
            "evaluation_config": _source(
                resolved_evaluation_path,
                root_path,
            ),
            "universe_config": _source(
                resolved_universe_path,
                root_path,
            ),
            "feature_spec": _source(
                resolved_feature_spec_path,
                root_path,
            ),
        },
    }

    destination.mkdir(parents=True, exist_ok=True)
    outputs = {
        key: destination / filename for key, filename in OUTPUT_FILENAMES.items()
    }
    allocations.to_parquet(outputs["allocations"], index=False)
    summary.to_csv(outputs["summary"], index=False)
    outputs["manifest"].write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    outputs["report"].write_text(
        format_sensitivity_report(
            summary=summary,
            manifest=manifest,
        ),
        encoding="utf-8",
    )
    return PolicySensitivityResult(
        outputs=outputs,
        allocations=allocations,
        summary=summary,
        manifest=manifest,
    )


def global_feature_indices(
    feature_spec: FeatureSpec,
    feature_names: list[str],
) -> dict[str, int]:
    """Map named global features to their flattened observation indices."""
    missing = [
        feature
        for feature in feature_names
        if feature not in feature_spec.global_features
    ]
    if missing:
        raise ValueError(f"feature spec missing global features: {missing}")
    offset = len(feature_spec.asset_order) * len(feature_spec.per_asset_features)
    return {
        feature: offset + feature_spec.global_features.index(feature)
        for feature in feature_names
    }


def train_feature_quantiles(
    dataset: PortfolioDataset,
    *,
    feature_indices: dict[str, int],
    low_quantile: float,
    high_quantile: float,
) -> dict[str, dict[str, float]]:
    """Calculate low/high normalized feature values from train rows only."""
    if not 0.0 < low_quantile < 0.5 < high_quantile < 1.0:
        raise ValueError("sensitivity quantiles must straddle the median")
    train = dataset.market_features[dataset.splits == "train"]
    if train.size == 0:
        raise ValueError("dataset has no train rows for sensitivity quantiles")
    quantiles = {}
    for feature, index in feature_indices.items():
        if index < 0 or index >= dataset.market_feature_dim:
            raise ValueError(f"feature index outside market observation: {feature}")
        values = train[:, index].astype(float)
        quantiles[feature] = {
            "low": float(np.quantile(values, low_quantile)),
            "high": float(np.quantile(values, high_quantile)),
        }
    return quantiles


def build_counterfactual_observations(
    observation: np.ndarray,
    *,
    feature_indices: dict[str, int],
    quantiles: dict[str, dict[str, float]],
) -> dict[tuple[str, str], dict[str, Any]]:
    """Create low/high risk observations while preserving all other values."""
    base = np.asarray(observation, dtype=np.float32)
    if base.ndim != 1 or not np.isfinite(base).all():
        raise ValueError("observation must be a finite one-dimensional array")
    results = {}
    for probe, scenarios in PROBE_SCENARIOS.items():
        for scenario, assignments in scenarios.items():
            counterfactual = base.copy()
            overrides = {}
            for feature, quantile_name in assignments.items():
                if feature not in feature_indices or feature not in quantiles:
                    raise ValueError(
                        f"missing sensitivity feature definition: {feature}"
                    )
                value = float(quantiles[feature][quantile_name])
                counterfactual[feature_indices[feature]] = value
                overrides[feature] = value
            results[(probe, scenario)] = {
                "observation": counterfactual,
                "overrides": overrides,
            }
    return results


def aggregate_sensitivity_results(
    allocations: pd.DataFrame,
    *,
    material_weight_shift: float,
) -> pd.DataFrame:
    """Aggregate high-minus-low risk effects for each seed and regime."""
    required = [
        "configuration_id",
        "experiment_name",
        "run_id",
        "seed",
        "selection_checkpoint",
        "regime_name",
        "split",
        "in_sample",
        "date",
        "probe",
        "scenario",
        "ticker",
        "target_weight",
        "observed_target_weight",
        "equity_like",
    ]
    _require_columns(allocations, required, "sensitivity allocations")
    scenario_allocations = allocations[allocations["probe"] != "observed"]
    exposure_columns = [
        "configuration_id",
        "experiment_name",
        "run_id",
        "seed",
        "selection_checkpoint",
        "regime_name",
        "split",
        "in_sample",
        "date",
        "probe",
        "scenario",
    ]
    exposure_rows = []
    for keys, group in scenario_allocations.groupby(exposure_columns, sort=True):
        weights = group["target_weight"].astype(float)
        observed = group["observed_target_weight"].astype(float)
        exposure_rows.append(
            {
                **dict(zip(exposure_columns, keys, strict=True)),
                "equity_like_weight": float(weights[group["equity_like"]].sum()),
                "shy_weight": float(weights[group["ticker"] == "SHY"].sum()),
                "hhi": float(np.square(weights).sum()),
                "effective_asset_count": float(1.0 / np.square(weights).sum()),
                "distance_from_observed": float(
                    0.5 * np.abs(weights.to_numpy() - observed.to_numpy()).sum()
                ),
            }
        )
    exposures = pd.DataFrame(exposure_rows)
    group_columns = [
        "configuration_id",
        "experiment_name",
        "run_id",
        "seed",
        "selection_checkpoint",
        "regime_name",
        "split",
        "in_sample",
        "probe",
    ]
    rows = []
    for keys, group in exposures.groupby(group_columns, sort=True):
        low = group[group["scenario"] == "low_risk"].set_index("date")
        high = group[group["scenario"] == "high_risk"].set_index("date")
        if set(low.index) != set(high.index):
            raise ValueError("low/high sensitivity dates do not match")
        high = high.reindex(low.index)
        equity_delta = high["equity_like_weight"] - low["equity_like_weight"]
        shy_delta = high["shy_weight"] - low["shy_weight"]
        hhi_delta = high["hhi"] - low["hhi"]
        distance = _scenario_allocation_distance(
            scenario_allocations,
            metadata=dict(zip(group_columns, keys, strict=True)),
        )
        median_equity_delta = float(equity_delta.median())
        median_shy_delta = float(shy_delta.median())
        material_equity_increase = median_equity_delta >= material_weight_shift
        material_shy_decrease = median_shy_delta <= -material_weight_shift
        rows.append(
            {
                **dict(zip(group_columns, keys, strict=True)),
                "decision_count": len(low),
                "median_equity_like_weight_delta": median_equity_delta,
                "median_shy_weight_delta": median_shy_delta,
                "median_hhi_delta": float(hhi_delta.median()),
                "median_one_way_allocation_distance": float(distance.median()),
                "median_low_risk_distance_from_observed": float(
                    low["distance_from_observed"].median()
                ),
                "median_high_risk_distance_from_observed": float(
                    high["distance_from_observed"].median()
                ),
                "equity_increase_date_fraction": float((equity_delta > 0.0).mean()),
                "shy_decrease_date_fraction": float((shy_delta < 0.0).mean()),
                "material_equity_increase": bool(material_equity_increase),
                "material_shy_decrease": bool(material_shy_decrease),
                "material_pro_risk_response": bool(
                    median_equity_delta > 0.0
                    and median_shy_delta < 0.0
                    and (material_equity_increase or material_shy_decrease)
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["regime_name", "probe", "seed"],
        kind="mergesort",
        ignore_index=True,
    )


def interpret_sensitivity_results(
    summary: pd.DataFrame,
) -> dict[str, Any]:
    """Interpret cross-seed consistency without changing model selection."""
    rows = []
    for keys, group in summary.groupby(["regime_name", "probe"], sort=True):
        regime_name, probe = keys
        material_fraction = float(group["material_pro_risk_response"].mean())
        rows.append(
            {
                "regime_name": str(regime_name),
                "probe": str(probe),
                "seed_count": int(group["seed"].nunique()),
                "median_seed_equity_like_weight_delta": float(
                    group["median_equity_like_weight_delta"].median()
                ),
                "median_seed_shy_weight_delta": float(
                    group["median_shy_weight_delta"].median()
                ),
                "material_pro_risk_seed_fraction": material_fraction,
                "majority_material_pro_risk": material_fraction >= 0.60,
            }
        )
    aggregate = pd.DataFrame(rows)
    stop_regimes = []
    for regime_name, group in aggregate.groupby("regime_name", sort=True):
        if set(group["probe"]) == set(PROBE_SCENARIOS) and bool(
            group["majority_material_pro_risk"].all()
        ):
            stop_regimes.append(str(regime_name))
    return {
        "by_regime_and_probe": rows,
        "stop_before_packaging": bool(stop_regimes),
        "stop_regimes": stop_regimes,
        "recommendation": (
            "investigate_policy_or_feature_design_before_packaging"
            if stop_regimes
            else "counterfactual_probe_does_not_block_packaging"
        ),
    }


def format_sensitivity_report(
    *,
    summary: pd.DataFrame,
    manifest: dict[str, Any],
) -> str:
    """Format counterfactual sensitivity results as Markdown."""
    interpretation = manifest["campaign_interpretation"]
    lines = [
        "# Counterfactual Policy Sensitivity",
        "",
        f"Configuration: `{manifest['configuration_id']}`",
        "",
        (
            f"Seeds: {manifest['seed_count']}; observed actions reconciled: "
            f"{str(manifest['all_observed_actions_reconciled']).lower()}; "
            f"test split used: {str(manifest['test_split_used']).lower()}."
        ),
        "",
        "## Interpretation",
        "",
        f"- Recommendation: `{interpretation['recommendation']}`",
        (
            "- Stop before packaging: "
            f"{str(interpretation['stop_before_packaging']).lower()}"
        ),
        (
            "- Regimes triggering the stop condition: "
            f"{', '.join(interpretation['stop_regimes']) or 'none'}"
        ),
        "",
        "## Seed-Level Effects",
        "",
        (
            "| Seed | Regime | Probe | Equity Δ | SHY Δ | HHI Δ | "
            "One-way distance | Material pro-risk |"
        ),
        "|---:|---|---|---:|---:|---:|---:|---|",
    ]
    for row in summary.to_dict(orient="records"):
        lines.append(
            f"| {int(row['seed'])} | {row['regime_name']} | "
            f"{row['probe']} | "
            f"{row['median_equity_like_weight_delta']:.4f} | "
            f"{row['median_shy_weight_delta']:.4f} | "
            f"{row['median_hhi_delta']:.4f} | "
            f"{row['median_one_way_allocation_distance']:.4f} | "
            f"{str(bool(row['material_pro_risk_response'])).lower()} |"
        )
    lines.extend(
        [
            "",
            "## Cross-Seed Effects",
            "",
            (
                "| Regime | Probe | Median equity Δ | Median SHY Δ | "
                "Material pro-risk seed fraction | Majority |"
            ),
            "|---|---|---:|---:|---:|---|",
        ]
    )
    for row in interpretation["by_regime_and_probe"]:
        lines.append(
            f"| {row['regime_name']} | {row['probe']} | "
            f"{row['median_seed_equity_like_weight_delta']:.4f} | "
            f"{row['median_seed_shy_weight_delta']:.4f} | "
            f"{row['material_pro_risk_seed_fraction']:.2f} | "
            f"{str(row['majority_material_pro_risk']).lower()} |"
        )
    lines.extend(
        [
            "",
            (
                "The probes alter normalized train-quantile risk inputs while "
                "holding all other market features and actual pre-trade weights "
                "fixed."
            ),
            "",
            "The results are diagnostic and do not retune or re-rank the policy.",
            "",
        ]
    )
    return "\n".join(lines)


def _scenario_allocation_distance(
    allocations: pd.DataFrame,
    *,
    metadata: dict[str, Any],
) -> pd.Series:
    mask = pd.Series(True, index=allocations.index)
    for column, value in metadata.items():
        mask &= allocations[column] == value
    selected = allocations[mask]
    pivot = selected.pivot(
        index=["date", "ticker"],
        columns="scenario",
        values="target_weight",
    )
    return (pivot["high_risk"] - pivot["low_risk"]).abs().groupby("date").sum() * 0.5


def _allocation_rows(
    *,
    metadata: dict[str, Any],
    probe: str,
    scenario: str,
    tickers: list[str],
    weights: np.ndarray,
    observed_weights: np.ndarray,
    equity_like_tickers: set[str],
    overrides: dict[str, float],
) -> list[dict[str, Any]]:
    _validate_weights(weights, tickers)
    return [
        {
            **metadata,
            "probe": probe,
            "scenario": scenario,
            "ticker": ticker,
            "target_weight": float(weight),
            "observed_target_weight": float(observed),
            "weight_delta_from_observed": float(weight - observed),
            "equity_like": ticker in equity_like_tickers,
            "feature_overrides": json.dumps(
                overrides,
                sort_keys=True,
                separators=(",", ":"),
            ),
        }
        for ticker, weight, observed in zip(
            tickers,
            weights,
            observed_weights,
            strict=True,
        )
    ]


def _validate_weights(weights: np.ndarray, tickers: list[str]) -> None:
    if weights.shape != (len(tickers),):
        raise ValueError("sensitivity policy returned invalid weight shape")
    if not np.isfinite(weights).all() or (weights < 0.0).any():
        raise ValueError("sensitivity policy returned invalid weights")
    if not np.isclose(weights.sum(), 1.0, atol=1e-6):
        raise ValueError("sensitivity policy weights must sum to one")


def _validate_diagnostic_inputs(
    *,
    selected: dict[str, Any],
    diagnostic_summary: dict[str, Any],
    allocations: pd.DataFrame,
    selected_runs: list[dict[str, Any]],
) -> None:
    if (
        diagnostic_summary.get("test_split_used") is not False
        or diagnostic_summary.get("validation_only") is not True
        or diagnostic_summary.get("all_validation_metrics_reconciled") is not True
    ):
        raise ValueError("policy diagnostics must be validation-only and reconciled")
    if diagnostic_summary.get("configuration_id") != selected.get("configuration_id"):
        raise ValueError("policy diagnostics configuration does not match selection")
    required = [
        "run_id",
        "seed",
        "selection_checkpoint",
        "regime_name",
        "split",
        "in_sample",
        "date",
        "ticker",
        "pre_trade_weight",
        "target_weight",
    ]
    _require_columns(allocations, required, "diagnostic allocations")
    if allocations.empty:
        raise ValueError("diagnostic allocations must not be empty")
    if (allocations["split"] == "test").any():
        raise ValueError("sensitivity diagnostics must not touch the test split")
    expected_runs = {str(run["run_id"]) for run in selected_runs}
    if set(allocations["run_id"].astype(str)) != expected_runs:
        raise ValueError("diagnostic allocation runs do not match selection")


def _validate_feature_contract(
    *,
    dataset: PortfolioDataset,
    feature_spec: FeatureSpec,
    universe_tickers: list[str],
) -> None:
    if dataset.asset_order != feature_spec.asset_order:
        raise ValueError("dataset and feature-spec asset order do not match")
    if feature_spec.asset_order != universe_tickers:
        raise ValueError("feature-spec and universe asset order do not match")
    if dataset.observation_dim != feature_spec.observation_dim:
        raise ValueError("dataset and feature-spec observation dimensions differ")


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
    return candidate if candidate.is_absolute() else root / candidate


def _require_columns(
    frame: pd.DataFrame,
    columns: list[str],
    label: str,
) -> None:
    missing = [column for column in columns if column not in frame]
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")

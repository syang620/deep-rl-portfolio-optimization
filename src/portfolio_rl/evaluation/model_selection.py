"""Validation-only seed-stability aggregation for Phase 3 experiments."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from math import isfinite
from pathlib import Path
from typing import Any

import pandas as pd

from portfolio_rl.config.loader import load_phase3_evaluation_config
from portfolio_rl.config.schemas import SelectionConfig
from portfolio_rl.evaluation.reports import collect_baseline_metrics


REQUIRED_REGISTRY_COLUMNS = [
    "run_id",
    "experiment_name",
    "selection_eligible",
    "eligibility_issues",
    "selection_checkpoint",
    "selection_model_path",
    "selection_validation_sharpe_ratio",
    "selection_validation_total_return",
    "selection_validation_max_drawdown",
    "selection_validation_average_weekly_turnover",
    "selection_validation_transaction_cost_drag",
]
SELECTION_CHECKPOINTS = {"best_checkpoint", "final_endpoint"}
SELECTION_METRIC_SOURCE = "best_available_checkpoint"
SELECTION_METRIC_COLUMNS = {
    "validation_sharpe_ratio": "selection_validation_sharpe_ratio",
    "validation_total_return": "selection_validation_total_return",
    "validation_max_drawdown": "selection_validation_max_drawdown",
    "validation_average_weekly_turnover": (
        "selection_validation_average_weekly_turnover"
    ),
    "validation_transaction_cost_drag": (
        "selection_validation_transaction_cost_drag"
    ),
}
AGGREGATE_METRICS = [
    metric
    for metric in SELECTION_METRIC_COLUMNS
    if metric != "validation_sharpe_ratio"
]
SEED_STABILITY_COLUMNS = [
    "configuration_id",
    "experiment_name",
    "metric_source",
    "selection_checkpoint_counts",
    "total_timesteps",
    "overrides",
    "planned_seed_count",
    "eligible_seed_count",
    "ineligible_seed_count",
    "coverage_ratio",
    "planned_seeds",
    "eligible_seeds",
    "ineligible_run_ids",
    "ineligibility_issues",
    "ranking_ready",
    "validation_sharpe_ratio_mean",
    "validation_sharpe_ratio_median",
    "validation_sharpe_ratio_std",
    "validation_sharpe_ratio_min",
    "validation_sharpe_ratio_max",
    *[
        f"{metric}_{statistic}"
        for metric in AGGREGATE_METRICS
        for statistic in ("median", "std")
    ],
]
REQUIRED_BASELINES = [
    "equal_weight_weekly",
    "buy_and_hold_equal_weight",
    "spy_only",
    "shy_only",
    "inverse_volatility",
]
BASELINE_METRICS = [
    "total_return",
    "sharpe_ratio",
    "max_drawdown",
    "average_weekly_turnover",
    "transaction_cost_drag",
]
REQUIRED_STABILITY_COLUMNS = [
    "configuration_id",
    "experiment_name",
    "metric_source",
    "selection_checkpoint_counts",
    "total_timesteps",
    "overrides",
    "planned_seed_count",
    "eligible_seed_count",
    "planned_seeds",
    "eligible_seeds",
    "ranking_ready",
    "validation_sharpe_ratio_median",
    "validation_sharpe_ratio_std",
    "validation_total_return_median",
    "validation_max_drawdown_median",
    "validation_average_weekly_turnover_median",
    "validation_transaction_cost_drag_median",
]


@dataclass(frozen=True)
class CandidateSelectionResult:
    """Written ranking artifacts and an optional selected configuration."""

    outputs: dict[str, Path]
    selected_configuration: dict[str, Any] | None


class NoPassingCandidateError(RuntimeError):
    """Raised after audit outputs are written when no candidate passes."""

    def __init__(self, outputs: dict[str, Path]) -> None:
        super().__init__("no configuration passed all validation gates")
        self.outputs = outputs


def rank_candidate_configurations(
    stability: pd.DataFrame,
    baseline_metrics: dict[str, dict[str, float | None]],
    selection_config: SelectionConfig,
) -> pd.DataFrame:
    """Apply validation gates and deterministically rank configurations."""
    _validate_stability(stability)
    _validate_baselines(baseline_metrics)
    _validate_selection_config(selection_config)
    equal_weight = baseline_metrics["equal_weight_weekly"]
    shy = baseline_metrics["shy_only"]
    rows = []
    for candidate in stability.to_dict(orient="records"):
        metrics = _candidate_metrics(candidate)
        finite_metrics = all(value is not None for value in metrics.values())
        gate_results = {
            "gate_finite_metrics": finite_metrics,
            "gate_seed_coverage": (
                _coerce_bool(candidate["ranking_ready"])
                and int(candidate["eligible_seed_count"])
                >= selection_config.min_eligible_seeds
            ),
            "gate_shy_total_return": (
                finite_metrics
                and (
                    not selection_config.require_beats_shy_total_return
                    or float(metrics["total_return"])
                    >= float(shy["total_return"])
                )
            ),
            "gate_equal_weight_materiality": (
                finite_metrics
                and not (
                    float(metrics["sharpe_ratio"])
                    < float(equal_weight["sharpe_ratio"])
                    - selection_config.equal_weight_sharpe_tolerance
                    and float(metrics["max_drawdown"])
                    < float(equal_weight["max_drawdown"])
                    - selection_config.equal_weight_drawdown_tolerance
                )
            ),
            "gate_weekly_turnover": (
                finite_metrics
                and float(metrics["average_weekly_turnover"])
                <= selection_config.max_median_weekly_turnover
            ),
            "gate_transaction_cost_drag": (
                finite_metrics
                and float(metrics["transaction_cost_drag"])
                <= selection_config.max_median_transaction_cost_drag
            ),
        }
        failed_gates = [
            name for name, passed in gate_results.items() if not passed
        ]
        rows.append(
            {
                **candidate,
                "validation_total_return_vs_shy": (
                    float(metrics["total_return"]) - float(shy["total_return"])
                    if metrics["total_return"] is not None
                    else None
                ),
                "validation_sharpe_vs_equal_weight": (
                    float(metrics["sharpe_ratio"])
                    - float(equal_weight["sharpe_ratio"])
                    if metrics["sharpe_ratio"] is not None
                    else None
                ),
                "validation_drawdown_vs_equal_weight": (
                    float(metrics["max_drawdown"])
                    - float(equal_weight["max_drawdown"])
                    if metrics["max_drawdown"] is not None
                    else None
                ),
                **gate_results,
                "passes_all_gates": not failed_gates,
                "failed_gates": _canonical_json(failed_gates),
            }
        )
    ranked = pd.DataFrame(rows)
    ranked = ranked.sort_values(
        by=[
            "passes_all_gates",
            "validation_sharpe_ratio_median",
            "validation_max_drawdown_median",
            "validation_average_weekly_turnover_median",
            "validation_transaction_cost_drag_median",
            "validation_sharpe_ratio_std",
            "configuration_id",
        ],
        ascending=[False, False, False, True, True, True, True],
        na_position="last",
        kind="mergesort",
    ).reset_index(drop=True)
    ranked.insert(0, "rank", range(1, len(ranked) + 1))
    return ranked


def write_candidate_ranking(
    *,
    seed_stability_path: str | Path,
    baseline_root: str | Path,
    evaluation_config_path: str | Path,
    output_dir: str | Path,
) -> CandidateSelectionResult:
    """Write validation ranking artifacts and freeze the best configuration."""
    stability_path = Path(seed_stability_path)
    baseline_path = Path(baseline_root)
    config_path = Path(evaluation_config_path)
    destination = Path(output_dir)
    stability = pd.read_csv(stability_path)
    evaluation_config = load_phase3_evaluation_config(config_path)
    baselines = collect_baseline_metrics(baseline_path)
    ranked = rank_candidate_configurations(
        stability,
        baselines,
        evaluation_config.selection,
    )

    destination.mkdir(parents=True, exist_ok=True)
    csv_path = destination / "candidate_ranking.csv"
    markdown_path = destination / "candidate_ranking.md"
    report_path = destination / "validation_selection_report.md"
    selected_path = destination / "selected_configuration.json"
    ranked.to_csv(csv_path, index=False)
    markdown_path.write_text(
        _candidate_ranking_markdown(ranked),
        encoding="utf-8",
    )
    passing = ranked[ranked["passes_all_gates"]]
    selected = None
    if not passing.empty:
        selected = _selected_configuration(
            passing.iloc[0],
            evaluation_config.selection,
            baselines,
            stability_path,
            baseline_path,
            config_path,
        )
        selected_path.write_text(
            json.dumps(selected, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    elif selected_path.exists():
        selected_path.unlink()
    report_path.write_text(
        _validation_selection_report(
            ranked,
            baselines,
            evaluation_config.selection,
            selected,
        ),
        encoding="utf-8",
    )
    outputs = {
        "ranking_csv": csv_path,
        "ranking_markdown": markdown_path,
        "report": report_path,
    }
    if selected is not None:
        outputs["selected_configuration"] = selected_path
    result = CandidateSelectionResult(
        outputs=outputs,
        selected_configuration=selected,
    )
    if selected is None:
        raise NoPassingCandidateError(outputs)
    return result


def aggregate_seed_stability(
    registry: pd.DataFrame,
    matrix_manifest_path: str | Path,
) -> pd.DataFrame:
    """Aggregate eligible validation results by matrix configuration."""
    _validate_registry(registry)
    manifest_path = Path(matrix_manifest_path)
    manifest = _read_json(manifest_path)
    experiment_name, planned_runs = _planned_runs(manifest, manifest_path)
    matrix_run_ids = {run["run_id"] for run in planned_runs}

    registry_by_run = registry.set_index("run_id", drop=False)
    matched_registry = registry[registry["run_id"].isin(matrix_run_ids)]
    mismatched = matched_registry[
        matched_registry["experiment_name"] != experiment_name
    ]
    if not mismatched.empty:
        raise ValueError(
            "registry experiment_name does not match matrix manifest for runs: "
            + ", ".join(sorted(mismatched["run_id"].astype(str)))
        )
    unexpected = registry[
        (registry["experiment_name"] == experiment_name)
        & ~registry["run_id"].isin(matrix_run_ids)
    ]
    if not unexpected.empty:
        raise ValueError(
            "registry contains runs outside the scoped matrix: "
            + ", ".join(sorted(unexpected["run_id"].astype(str)))
        )

    grouped: dict[str, list[dict[str, Any]]] = {}
    seen_seeds: set[tuple[str, int]] = set()
    for planned in planned_runs:
        config_id = _configuration_id(
            planned["total_timesteps"],
            planned["overrides"],
        )
        seed_key = (config_id, planned["seed"])
        if seed_key in seen_seeds:
            raise ValueError(
                "matrix contains duplicate seed for configuration "
                f"{config_id}: {planned['seed']}"
            )
        seen_seeds.add(seed_key)
        run = dict(planned)
        run["configuration_id"] = config_id
        run["registry_row"] = (
            registry_by_run.loc[planned["run_id"]]
            if planned["run_id"] in registry_by_run.index
            else None
        )
        grouped.setdefault(config_id, []).append(run)

    rows = [
        _aggregate_configuration(experiment_name, config_id, runs)
        for config_id, runs in sorted(
            grouped.items(),
            key=lambda item: (
                item[1][0]["total_timesteps"],
                _canonical_json(item[1][0]["overrides"]),
                item[0],
            ),
        )
    ]
    return pd.DataFrame(rows, columns=SEED_STABILITY_COLUMNS)


def write_seed_stability_report(
    *,
    registry_path: str | Path,
    matrix_manifest_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Write CSV and Markdown seed-stability artifacts."""
    registry = _read_registry(Path(registry_path))
    stability = aggregate_seed_stability(registry, matrix_manifest_path)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    csv_path = destination / "seed_stability.csv"
    markdown_path = destination / "seed_stability.md"
    stability.to_csv(csv_path, index=False)
    markdown_path.write_text(
        _seed_stability_markdown(stability),
        encoding="utf-8",
    )
    return {"csv": csv_path, "markdown": markdown_path}


def _planned_runs(
    manifest: dict[str, Any],
    manifest_path: Path,
) -> tuple[str, list[dict[str, Any]]]:
    experiment_name = manifest.get("experiment_name")
    runs = manifest.get("runs")
    if not isinstance(experiment_name, str) or not experiment_name:
        raise ValueError(
            f"matrix manifest has no experiment_name: {manifest_path}"
        )
    if not isinstance(runs, list) or manifest.get("run_count") != len(runs):
        raise ValueError(
            f"matrix manifest has an invalid run inventory: {manifest_path}"
        )

    validated = []
    seen_run_ids: set[str] = set()
    for run in runs:
        if not isinstance(run, dict):
            raise ValueError(f"matrix run must be an object: {manifest_path}")
        run_id = run.get("run_id")
        seed = run.get("seed")
        total_timesteps = run.get("total_timesteps")
        overrides = run.get("overrides")
        if not isinstance(run_id, str) or not run_id:
            raise ValueError(f"matrix run has an invalid run_id: {manifest_path}")
        if run_id in seen_run_ids:
            raise ValueError(f"matrix contains duplicate run_id: {run_id}")
        if (
            not isinstance(seed, int)
            or isinstance(seed, bool)
            or seed < 0
        ):
            raise ValueError(f"matrix run has an invalid seed: {run_id}")
        if (
            not isinstance(total_timesteps, int)
            or isinstance(total_timesteps, bool)
            or total_timesteps <= 0
        ):
            raise ValueError(
                f"matrix run has invalid total_timesteps: {run_id}"
            )
        if not isinstance(overrides, dict):
            raise ValueError(f"matrix run has invalid overrides: {run_id}")
        seen_run_ids.add(run_id)
        validated.append(
            {
                "run_id": run_id,
                "seed": seed,
                "total_timesteps": total_timesteps,
                "status": run.get("status"),
                "overrides": overrides,
            }
        )
    return experiment_name, validated


def _aggregate_configuration(
    experiment_name: str,
    config_id: str,
    runs: list[dict[str, Any]],
) -> dict[str, Any]:
    eligible_rows = []
    ineligibility: dict[str, str] = {}
    for run in runs:
        registry_row = run["registry_row"]
        if registry_row is None:
            ineligibility[run["run_id"]] = "missing_registry_run"
            continue
        if run["status"] != "completed":
            ineligibility[run["run_id"]] = (
                f"matrix_status:{run['status'] or 'missing'}"
            )
            continue
        if not _registry_eligible(registry_row["selection_eligible"]):
            issues = registry_row["eligibility_issues"]
            ineligibility[run["run_id"]] = (
                str(issues) if pd.notna(issues) and str(issues) else "ineligible"
            )
            continue
        _validate_eligible_metrics(registry_row, run["run_id"])
        eligible_rows.append((run, registry_row))

    planned_count = len(runs)
    eligible_count = len(eligible_rows)
    checkpoint_counts: dict[str, int] = {}
    for _run, registry_row in eligible_rows:
        checkpoint = str(registry_row["selection_checkpoint"])
        checkpoint_counts[checkpoint] = checkpoint_counts.get(checkpoint, 0) + 1
    row = {
        "configuration_id": config_id,
        "experiment_name": experiment_name,
        "metric_source": SELECTION_METRIC_SOURCE,
        "selection_checkpoint_counts": _canonical_json(checkpoint_counts),
        "total_timesteps": runs[0]["total_timesteps"],
        "overrides": _canonical_json(runs[0]["overrides"]),
        "planned_seed_count": planned_count,
        "eligible_seed_count": eligible_count,
        "ineligible_seed_count": planned_count - eligible_count,
        "coverage_ratio": eligible_count / planned_count,
        "planned_seeds": _canonical_json(
            sorted(run["seed"] for run in runs)
        ),
        "eligible_seeds": _canonical_json(
            sorted(run["seed"] for run, _registry_row in eligible_rows)
        ),
        "ineligible_run_ids": _canonical_json(sorted(ineligibility)),
        "ineligibility_issues": _canonical_json(ineligibility),
        "ranking_ready": eligible_count == planned_count,
    }
    eligible_frame = pd.DataFrame(
        [registry_row for _run, registry_row in eligible_rows]
    )
    row.update(_metric_statistics(eligible_frame))
    return row


def _metric_statistics(frame: pd.DataFrame) -> dict[str, float | None]:
    output: dict[str, float | None] = {}
    sharpe = _numeric_series(
        frame,
        SELECTION_METRIC_COLUMNS["validation_sharpe_ratio"],
    )
    for statistic in ("mean", "median", "std", "min", "max"):
        output[f"validation_sharpe_ratio_{statistic}"] = _statistic(
            sharpe,
            statistic,
        )
    for metric in AGGREGATE_METRICS:
        values = _numeric_series(frame, SELECTION_METRIC_COLUMNS[metric])
        for statistic in ("median", "std"):
            output[f"{metric}_{statistic}"] = _statistic(values, statistic)
    return output


def _numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce")


def _statistic(values: pd.Series, statistic: str) -> float | None:
    if values.empty:
        return None
    result = getattr(values, statistic)()
    return float(result) if pd.notna(result) else None


def _validate_registry(registry: pd.DataFrame) -> None:
    missing = [
        column
        for column in REQUIRED_REGISTRY_COLUMNS
        if column not in registry.columns
    ]
    if missing:
        raise ValueError(f"registry missing required columns: {missing}")
    duplicated = registry[registry["run_id"].duplicated(keep=False)]
    if not duplicated.empty:
        raise ValueError(
            "registry contains duplicate run_ids: "
            + ", ".join(sorted(duplicated["run_id"].astype(str).unique()))
        )


def _validate_eligible_metrics(row: pd.Series, run_id: str) -> None:
    if row["selection_checkpoint"] not in SELECTION_CHECKPOINTS:
        raise ValueError(
            f"eligible registry run has unsupported selection checkpoint "
            f"({run_id}): {row['selection_checkpoint']!r}"
        )
    if not isinstance(row["selection_model_path"], str) or not row[
        "selection_model_path"
    ]:
        raise ValueError(
            f"eligible registry run has no selection model path: {run_id}"
        )
    metric_columns = list(SELECTION_METRIC_COLUMNS.values())
    invalid = [
        column for column in metric_columns if not _finite_number(row[column])
    ]
    if invalid:
        raise ValueError(
            f"eligible registry run has invalid metrics ({run_id}): {invalid}"
        )


def _registry_eligible(value: Any) -> bool:
    if pd.api.types.is_bool(value):
        return bool(value)
    if isinstance(value, str) and value.lower() in {"true", "false"}:
        return value.lower() == "true"
    raise ValueError(f"selection_eligible must be boolean: {value!r}")


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and isfinite(float(value))
    )


def _configuration_id(
    total_timesteps: int,
    overrides: dict[str, Any],
) -> str:
    payload = {
        "total_timesteps": total_timesteps,
        "overrides": overrides,
    }
    digest = hashlib.sha256(
        _canonical_json(payload).encode("utf-8")
    ).hexdigest()
    return digest[:12]


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _read_registry(path: Path) -> pd.DataFrame:
    if path.suffix == ".csv":
        return pd.read_csv(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError("registry must be a CSV or Parquet file")


def _read_json(path: Path) -> dict[str, Any]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"expected JSON object: {path}")
    return loaded


def _validate_stability(stability: pd.DataFrame) -> None:
    missing = [
        column
        for column in REQUIRED_STABILITY_COLUMNS
        if column not in stability.columns
    ]
    if missing:
        raise ValueError(f"seed stability missing required columns: {missing}")
    if stability.empty:
        raise ValueError("seed stability must contain at least one configuration")
    duplicated = stability[
        stability["configuration_id"].duplicated(keep=False)
    ]
    if not duplicated.empty:
        raise ValueError(
            "seed stability contains duplicate configuration_ids: "
            + ", ".join(
                sorted(duplicated["configuration_id"].astype(str).unique())
            )
        )
    if stability["experiment_name"].nunique(dropna=False) != 1:
        raise ValueError("seed stability must contain exactly one experiment")


def _validate_baselines(
    baselines: dict[str, dict[str, float | None]],
) -> None:
    missing = [
        strategy for strategy in REQUIRED_BASELINES if strategy not in baselines
    ]
    if missing:
        raise ValueError(f"missing required validation baselines: {missing}")
    invalid = [
        f"{strategy}.{metric}"
        for strategy in REQUIRED_BASELINES
        for metric in BASELINE_METRICS
        if not _finite_number(baselines[strategy].get(metric))
    ]
    if invalid:
        raise ValueError(f"invalid validation baseline metrics: {invalid}")


def _validate_selection_config(config: SelectionConfig) -> None:
    if config.primary_metric != "sharpe_ratio" or not config.higher_is_better:
        raise ValueError(
            "candidate ranking currently requires higher-is-better sharpe_ratio"
        )
    supported_tie_breakers = {
        "max_drawdown",
        "average_weekly_turnover",
        "transaction_cost_drag",
    }
    unsupported = [
        value for value in config.tie_breakers if value not in supported_tie_breakers
    ]
    if unsupported:
        raise ValueError(f"unsupported selection tie breakers: {unsupported}")


def _candidate_metrics(candidate: dict[str, Any]) -> dict[str, float | None]:
    return {
        "sharpe_ratio": _optional_finite(
            candidate["validation_sharpe_ratio_median"]
        ),
        "total_return": _optional_finite(
            candidate["validation_total_return_median"]
        ),
        "max_drawdown": _optional_finite(
            candidate["validation_max_drawdown_median"]
        ),
        "average_weekly_turnover": _optional_finite(
            candidate["validation_average_weekly_turnover_median"]
        ),
        "transaction_cost_drag": _optional_finite(
            candidate["validation_transaction_cost_drag_median"]
        ),
    }


def _optional_finite(value: Any) -> float | None:
    return float(value) if _finite_number(value) else None


def _coerce_bool(value: Any) -> bool:
    if pd.api.types.is_bool(value):
        return bool(value)
    if isinstance(value, str) and value.lower() in {"true", "false"}:
        return value.lower() == "true"
    raise ValueError(f"expected boolean value, received: {value!r}")


def _selected_configuration(
    candidate: pd.Series,
    selection_config: SelectionConfig,
    baselines: dict[str, dict[str, float | None]],
    stability_path: Path,
    baseline_root: Path,
    config_path: Path,
) -> dict[str, Any]:
    candidate_metrics = {
        "sharpe_ratio_median": float(
            candidate["validation_sharpe_ratio_median"]
        ),
        "sharpe_ratio_std": _json_optional_float(
            candidate["validation_sharpe_ratio_std"]
        ),
        "total_return_median": float(
            candidate["validation_total_return_median"]
        ),
        "max_drawdown_median": float(
            candidate["validation_max_drawdown_median"]
        ),
        "average_weekly_turnover_median": float(
            candidate["validation_average_weekly_turnover_median"]
        ),
        "transaction_cost_drag_median": float(
            candidate["validation_transaction_cost_drag_median"]
        ),
    }
    baseline_deltas = {
        strategy: {
            metric: candidate_metrics[f"{metric}_median"]
            - float(values[metric])
            for metric in [
                "total_return",
                "max_drawdown",
                "average_weekly_turnover",
                "transaction_cost_drag",
            ]
        }
        | {
            "sharpe_ratio": candidate_metrics["sharpe_ratio_median"]
            - float(values["sharpe_ratio"])
        }
        for strategy, values in baselines.items()
        if strategy in REQUIRED_BASELINES
    }
    gate_names = [
        "gate_finite_metrics",
        "gate_seed_coverage",
        "gate_shy_total_return",
        "gate_equal_weight_materiality",
        "gate_weekly_turnover",
        "gate_transaction_cost_drag",
    ]
    return {
        "schema_version": 1,
        "selected_at": datetime.now(UTC).isoformat(),
        "experiment_name": str(candidate["experiment_name"]),
        "configuration_id": str(candidate["configuration_id"]),
        "metric_source": str(candidate["metric_source"]),
        "selection_checkpoint_counts": json.loads(
            str(candidate["selection_checkpoint_counts"])
        ),
        "total_timesteps": int(candidate["total_timesteps"]),
        "overrides": json.loads(str(candidate["overrides"])),
        "planned_seeds": json.loads(str(candidate["planned_seeds"])),
        "eligible_seeds": json.loads(str(candidate["eligible_seeds"])),
        "validation_metrics": candidate_metrics,
        "baseline_deltas": baseline_deltas,
        "gate_results": {
            name: _coerce_bool(candidate[name]) for name in gate_names
        },
        "selection_config": selection_config.model_dump(mode="json"),
        "sources": {
            "seed_stability": _source_reference(stability_path),
            "evaluation_config": _source_reference(config_path),
            "baselines": {
                strategy: _source_reference(
                    baseline_root / strategy / "metrics.json"
                )
                for strategy in REQUIRED_BASELINES
            },
        },
        "validation_only": True,
        "test_split_used": False,
    }


def _json_optional_float(value: Any) -> float | None:
    return float(value) if _finite_number(value) else None


def _source_reference(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": _sha256_file(path)}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _candidate_ranking_markdown(ranked: pd.DataFrame) -> str:
    experiment_name = ranked.iloc[0]["experiment_name"]
    passing_count = int(ranked["passes_all_gates"].sum())
    lines = [
        f"# Candidate Ranking: {experiment_name}",
        "",
        (
            "Validation-only ranking using the best available checkpoint, "
            "including the final endpoint; the test split was not accessed."
        ),
        "",
        f"Configurations: {len(ranked)}",
        f"Passed all gates: {passing_count}",
        f"Failed gates: {len(ranked) - passing_count}",
        "",
    ]
    visible = [
        "rank",
        "configuration_id",
        "eligible_seed_count",
        "validation_sharpe_ratio_median",
        "validation_sharpe_ratio_std",
        "validation_total_return_median",
        "validation_max_drawdown_median",
        "validation_average_weekly_turnover_median",
        "validation_transaction_cost_drag_median",
        "passes_all_gates",
        "failed_gates",
    ]
    lines.extend(_markdown_table(ranked.loc[:, visible]))
    return "\n".join(lines) + "\n"


def _validation_selection_report(
    ranked: pd.DataFrame,
    baselines: dict[str, dict[str, float | None]],
    config: SelectionConfig,
    selected: dict[str, Any] | None,
) -> str:
    lines = [
        "# Validation Configuration Selection",
        "",
        "Selection used validation metrics only. The test split was not accessed.",
        f"Metric source: `{ranked.iloc[0]['metric_source']}`.",
        "The final training endpoint participates in checkpoint selection.",
        "",
        "## Gate Policy",
        "",
        f"- Minimum eligible seeds: {config.min_eligible_seeds}",
        "- Must equal or beat SHY validation total return: "
        f"{config.require_beats_shy_total_return}",
        "- Equal-weight Sharpe tolerance: "
        f"{config.equal_weight_sharpe_tolerance:.4f}",
        "- Equal-weight drawdown tolerance: "
        f"{config.equal_weight_drawdown_tolerance:.4f}",
        "- Maximum median weekly turnover: "
        f"{config.max_median_weekly_turnover:.4f}",
        "- Maximum median transaction-cost drag: "
        f"{config.max_median_transaction_cost_drag:.4f}",
        "",
        "## Baseline Validation Metrics",
        "",
    ]
    baseline_frame = pd.DataFrame(
        [
            {"strategy": strategy, **baselines[strategy]}
            for strategy in REQUIRED_BASELINES
        ]
    ).loc[:, ["strategy", *BASELINE_METRICS]]
    lines.extend(_markdown_table(baseline_frame))
    lines.extend(["", "## Selection Result", ""])
    if selected is None:
        lines.append(
            "No configuration passed all validation gates; no selection was frozen."
        )
    else:
        lines.extend(
            [
                "Selected configuration: "
                f"`{selected['configuration_id']}`.",
                "",
                f"Experiment: `{selected['experiment_name']}`.",
            ]
        )
    lines.extend(
        [
            "",
            f"Passing configurations: {int(ranked['passes_all_gates'].sum())}",
            f"Total configurations: {len(ranked)}",
        ]
    )
    return "\n".join(lines) + "\n"


def _seed_stability_markdown(stability: pd.DataFrame) -> str:
    experiment_name = (
        stability.iloc[0]["experiment_name"] if not stability.empty else "unknown"
    )
    ready_count = (
        int(stability["ranking_ready"].sum()) if not stability.empty else 0
    )
    lines = [
        f"# Seed Stability: {experiment_name}",
        "",
        (
            "Validation-only best-available-checkpoint statistics, including "
            "the final endpoint; the test split was not accessed."
        ),
        "",
        f"Configurations: {len(stability)}",
        f"Ranking ready: {ready_count}",
        f"Incomplete: {len(stability) - ready_count}",
        "",
    ]
    if stability.empty:
        return "\n".join(lines) + "\n"
    visible = [
        "configuration_id",
        "overrides",
        "selection_checkpoint_counts",
        "planned_seed_count",
        "eligible_seed_count",
        "coverage_ratio",
        "ranking_ready",
        "validation_sharpe_ratio_median",
        "validation_sharpe_ratio_std",
        "validation_total_return_median",
        "validation_max_drawdown_median",
        "validation_average_weekly_turnover_median",
        "validation_transaction_cost_drag_median",
        "ineligibility_issues",
    ]
    lines.extend(_markdown_table(stability.loc[:, visible]))
    return "\n".join(lines) + "\n"


def _markdown_table(frame: pd.DataFrame) -> list[str]:
    headers = list(frame.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in frame.itertuples(index=False):
        values = [
            _format_markdown_cell(value).replace("|", "\\|") for value in row
        ]
        lines.append("| " + " | ".join(values) + " |")
    return lines


def _format_markdown_cell(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)

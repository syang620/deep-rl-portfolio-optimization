"""Validation-only seed-stability aggregation for Phase 3 experiments."""

from __future__ import annotations

import hashlib
import json
from math import isfinite
from pathlib import Path
from typing import Any

import pandas as pd


REQUIRED_REGISTRY_COLUMNS = [
    "run_id",
    "experiment_name",
    "selection_eligible",
    "eligibility_issues",
    "validation_sharpe_ratio",
    "validation_total_return",
    "validation_max_drawdown",
    "validation_average_weekly_turnover",
    "validation_transaction_cost_drag",
]
AGGREGATE_METRICS = [
    "validation_total_return",
    "validation_max_drawdown",
    "validation_average_weekly_turnover",
    "validation_transaction_cost_drag",
]
SEED_STABILITY_COLUMNS = [
    "configuration_id",
    "experiment_name",
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
    row = {
        "configuration_id": config_id,
        "experiment_name": experiment_name,
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
    sharpe = _numeric_series(frame, "validation_sharpe_ratio")
    for statistic in ("mean", "median", "std", "min", "max"):
        output[f"validation_sharpe_ratio_{statistic}"] = _statistic(
            sharpe,
            statistic,
        )
    for metric in AGGREGATE_METRICS:
        values = _numeric_series(frame, metric)
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
    metric_columns = ["validation_sharpe_ratio", *AGGREGATE_METRICS]
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
        "Validation-only statistics; the test split was not accessed.",
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

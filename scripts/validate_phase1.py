"""Validate Phase 1 data artifacts without rerunning ETL."""

from __future__ import annotations

import argparse
import json
from math import isfinite
from pathlib import Path

import duckdb
import pandas as pd


REQUIRED_ARTIFACTS = (
    Path("data/raw/prices_daily.parquet"),
    Path("data/raw/macro_daily.parquet"),
    Path("data/processed/model_matrix_daily.parquet"),
    Path("data/duckdb/portfolio.duckdb"),
    Path("artifacts/scalers/feature_scaler_v1.pkl"),
    Path("artifacts/feature_specs/feature_spec_v1.json"),
    Path("artifacts/reports/data_quality_report_v1.json"),
)
REQUIRED_DUCKDB_TABLES = {
    "prices_daily",
    "macro_daily",
    "features_daily",
    "global_features_daily",
    "features_normalized_daily",
    "global_features_normalized_daily",
    "model_matrix_daily",
}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Validate Phase 1 artifacts.")
    parser.add_argument(
        "--root",
        default=".",
        help="Repository root containing data/, artifacts/, and configs/.",
    )
    args = parser.parse_args(argv)

    root = Path(args.root)
    summary = validate_phase1_artifacts(root)
    print(json.dumps(summary, indent=2))


def validate_phase1_artifacts(root: Path) -> dict[str, object]:
    """Validate final Phase 1 artifacts and return a summary."""
    _assert_required_artifacts_exist(root)
    feature_spec = _read_json(root / "artifacts/feature_specs/feature_spec_v1.json")
    data_quality_report = _read_json(
        root / "artifacts/reports/data_quality_report_v1.json"
    )
    model_matrix = pd.read_parquet(root / "data/processed/model_matrix_daily.parquet")

    numeric = model_matrix.select_dtypes(include="number")
    nan_count = int(numeric.isna().sum().sum())
    inf_count = int((~numeric.map(isfinite) & numeric.notna()).sum().sum())
    if nan_count:
        raise ValueError(f"model_matrix_daily contains NaN values: {nan_count}")
    if inf_count:
        raise ValueError(f"model_matrix_daily contains infinite values: {inf_count}")

    observation_columns = [
        column for column in model_matrix.columns if column.startswith("obs_")
    ]
    expected_observation_dim = int(feature_spec["observation_dim"])
    if len(observation_columns) != expected_observation_dim:
        raise ValueError(
            "model_matrix_daily observation column count does not match "
            f"feature_spec: {len(observation_columns)} != {expected_observation_dim}"
        )

    expected_row_count = int(data_quality_report["model_matrix_row_count"])
    if len(model_matrix) != expected_row_count:
        raise ValueError(
            "model_matrix_daily row count does not match data quality report: "
            f"{len(model_matrix)} != {expected_row_count}"
        )

    if int(data_quality_report["nan_count_final"]) != nan_count:
        raise ValueError("data quality report nan_count_final is stale")
    if int(data_quality_report["inf_count_final"]) != inf_count:
        raise ValueError("data quality report inf_count_final is stale")

    model_matrix_report = data_quality_report["processed_artifacts"][
        "model_matrix_daily"
    ]
    if int(model_matrix_report["row_count"]) != len(model_matrix):
        raise ValueError("data quality report model_matrix_daily row_count is stale")
    if int(model_matrix_report["missing_cell_count"]) != nan_count:
        raise ValueError(
            "data quality report model_matrix_daily missing_cell_count is stale"
        )
    if int(model_matrix_report["inf_count"]) != inf_count:
        raise ValueError("data quality report model_matrix_daily inf_count is stale")

    duckdb_tables = _duckdb_tables(root / "data/duckdb/portfolio.duckdb")
    missing_tables = sorted(REQUIRED_DUCKDB_TABLES - duckdb_tables)
    if missing_tables:
        raise ValueError(f"DuckDB is missing required tables: {missing_tables}")

    return {
        "status": "ok",
        "required_artifact_count": len(REQUIRED_ARTIFACTS),
        "duckdb_table_count": len(REQUIRED_DUCKDB_TABLES),
        "model_matrix_row_count": len(model_matrix),
        "observation_dim": expected_observation_dim,
        "nan_count_final": nan_count,
        "inf_count_final": inf_count,
    }


def _assert_required_artifacts_exist(root: Path) -> None:
    missing = [str(path) for path in REQUIRED_ARTIFACTS if not (root / path).exists()]
    if missing:
        raise FileNotFoundError(f"Phase 1 required artifacts are missing: {missing}")


def _read_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as json_file:
        return json.load(json_file)


def _duckdb_tables(database_path: Path) -> set[str]:
    with duckdb.connect(str(database_path), read_only=True) as connection:
        rows = connection.execute("SHOW TABLES").fetchall()
    return {str(row[0]) for row in rows}


if __name__ == "__main__":
    main()

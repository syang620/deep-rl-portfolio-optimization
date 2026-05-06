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
    Path("data/interim/aligned_feature_panel_daily.parquet"),
    Path("data/processed/features_daily.parquet"),
    Path("data/processed/global_features_daily.parquet"),
    Path("data/processed/features_normalized_daily.parquet"),
    Path("data/processed/global_features_normalized_daily.parquet"),
    Path("data/processed/model_matrix_daily.parquet"),
    Path("data/duckdb/portfolio.duckdb"),
    Path("artifacts/scalers/feature_scaler_v1.pkl"),
    Path("artifacts/feature_specs/feature_spec_v1.json"),
    Path("artifacts/reports/data_quality_report_v1.json"),
)
REQUIRED_DUCKDB_TABLES = {
    "prices_daily",
    "macro_daily",
    "aligned_feature_panel_daily",
    "features_daily",
    "global_features_daily",
    "features_normalized_daily",
    "global_features_normalized_daily",
    "model_matrix_daily",
}
RAW_ARTIFACTS = {
    "prices_daily": Path("data/raw/prices_daily.parquet"),
    "macro_daily": Path("data/raw/macro_daily.parquet"),
}
CLEAN_ARTIFACTS = {
    "aligned_feature_panel_daily": Path(
        "data/interim/aligned_feature_panel_daily.parquet"
    ),
    "features_daily": Path("data/processed/features_daily.parquet"),
    "global_features_daily": Path("data/processed/global_features_daily.parquet"),
    "features_normalized_daily": Path(
        "data/processed/features_normalized_daily.parquet"
    ),
    "global_features_normalized_daily": Path(
        "data/processed/global_features_normalized_daily.parquet"
    ),
    "model_matrix_daily": Path("data/processed/model_matrix_daily.parquet"),
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
    raw_artifact_summary = _summarize_raw_artifacts(root)
    clean_artifact_summary = _validate_clean_artifacts(root)
    asset_features = pd.read_parquet(root / CLEAN_ARTIFACTS["features_daily"])
    global_features = pd.read_parquet(root / CLEAN_ARTIFACTS["global_features_daily"])
    model_matrix = pd.read_parquet(root / CLEAN_ARTIFACTS["model_matrix_daily"])

    model_matrix_summary = clean_artifact_summary["model_matrix_daily"]
    nan_count = int(model_matrix_summary["missing_cell_count"])
    inf_count = int(model_matrix_summary["inf_count"])

    expected_observation_dim = int(feature_spec["observation_dim"])
    _validate_feature_names_are_backward_looking(feature_spec)
    _validate_model_matrix_dates(model_matrix, global_features)
    _validate_model_matrix_columns(model_matrix, feature_spec)
    _validate_return_alignment(model_matrix, asset_features, feature_spec)

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
        "timing_contract": _timing_contract(),
        "raw_artifacts": raw_artifact_summary,
        "clean_artifacts": clean_artifact_summary,
    }


def _assert_required_artifacts_exist(root: Path) -> None:
    missing = [str(path) for path in REQUIRED_ARTIFACTS if not (root / path).exists()]
    if missing:
        raise FileNotFoundError(f"Phase 1 required artifacts are missing: {missing}")


def _read_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as json_file:
        return json.load(json_file)


def _summarize_raw_artifacts(root: Path) -> dict[str, dict[str, int]]:
    return {
        artifact_name: _artifact_quality_counts(pd.read_parquet(root / artifact_path))
        for artifact_name, artifact_path in RAW_ARTIFACTS.items()
    }


def _validate_clean_artifacts(root: Path) -> dict[str, dict[str, int]]:
    summaries = {}
    for artifact_name, artifact_path in CLEAN_ARTIFACTS.items():
        summary = _artifact_quality_counts(pd.read_parquet(root / artifact_path))
        if summary["missing_cell_count"]:
            raise ValueError(
                f"{artifact_name} contains NaN values: "
                f"{summary['missing_cell_count']}"
            )
        if summary["inf_count"]:
            raise ValueError(
                f"{artifact_name} contains infinite values: {summary['inf_count']}"
            )
        summaries[artifact_name] = summary
    return summaries


def _artifact_quality_counts(frame: pd.DataFrame) -> dict[str, int]:
    numeric = frame.select_dtypes(include="number")
    inf_count = int((~numeric.map(isfinite) & numeric.notna()).sum().sum())
    return {
        "row_count": int(len(frame)),
        "missing_cell_count": int(frame.isna().sum().sum()),
        "inf_count": inf_count,
    }


def _validate_model_matrix_columns(
    model_matrix: pd.DataFrame,
    feature_spec: dict[str, object],
) -> None:
    expected_columns = _expected_model_matrix_columns(feature_spec)
    actual_columns = list(model_matrix.columns)
    if actual_columns != expected_columns:
        missing = [column for column in expected_columns if column not in actual_columns]
        extra = [column for column in actual_columns if column not in expected_columns]
        raise ValueError(
            "model_matrix_daily columns do not match feature_spec order: "
            f"missing={missing}, extra={extra}"
        )


def _validate_feature_names_are_backward_looking(
    feature_spec: dict[str, object],
) -> None:
    forbidden_tokens = ("future", "next", "target", "label")
    feature_names = [
        str(feature_name)
        for field_name in (
            "per_asset_features",
            "global_features",
            "current_weight_features",
        )
        for feature_name in feature_spec.get(field_name, [])
    ]
    forbidden_names = [
        feature_name
        for feature_name in feature_names
        if any(token in feature_name.lower() for token in forbidden_tokens)
    ]
    if forbidden_names:
        raise ValueError(
            "feature_spec contains forward-looking feature names: "
            f"{forbidden_names}"
        )


def _validate_model_matrix_dates(
    model_matrix: pd.DataFrame,
    global_features: pd.DataFrame,
) -> None:
    model_dates = pd.to_datetime(model_matrix["date"])
    if model_dates.duplicated().any():
        raise ValueError("model_matrix_daily contains duplicate dates")
    if not model_dates.is_monotonic_increasing:
        raise ValueError("model_matrix_daily dates must be sorted ascending")

    global_dates = pd.to_datetime(global_features["date"]).sort_values(ignore_index=True)
    if model_dates.reset_index(drop=True).to_list() != global_dates.to_list():
        raise ValueError("model_matrix_daily dates do not align with global_features_daily")


def _validate_return_alignment(
    model_matrix: pd.DataFrame,
    asset_features: pd.DataFrame,
    feature_spec: dict[str, object],
) -> None:
    _require_columns(asset_features, ["date", "ticker", "ret_1d"], "features_daily")
    asset_order = [str(ticker).upper() for ticker in feature_spec["asset_order"]]
    returns = asset_features.loc[:, ["date", "ticker", "ret_1d"]].copy()
    returns["date"] = pd.to_datetime(returns["date"])
    returns["ticker"] = returns["ticker"].str.upper()
    return_frame = returns.pivot(index="date", columns="ticker", values="ret_1d")
    missing_tickers = [ticker for ticker in asset_order if ticker not in return_frame]
    if missing_tickers:
        raise ValueError(f"features_daily ret_1d is missing tickers: {missing_tickers}")

    return_frame = return_frame.loc[:, asset_order]
    model_dates = pd.to_datetime(model_matrix["date"])
    missing_dates = [
        date.date().isoformat()
        for date in model_dates
        if date not in return_frame.index
    ]
    if missing_dates:
        raise ValueError(f"features_daily ret_1d is missing dates: {missing_dates}")

    expected = return_frame.loc[model_dates, asset_order].reset_index(drop=True)
    expected.columns = [f"return_{ticker.lower()}_1d" for ticker in asset_order]
    actual = model_matrix.loc[:, list(expected.columns)].reset_index(drop=True)
    mismatched_columns = [
        column
        for column in expected.columns
        if not actual[column].equals(expected[column])
    ]
    if mismatched_columns:
        raise ValueError(
            "model_matrix_daily return columns do not match same-date "
            f"features_daily ret_1d values: {mismatched_columns}"
        )


def _expected_model_matrix_columns(feature_spec: dict[str, object]) -> list[str]:
    observation_dim = int(feature_spec["observation_dim"])
    asset_order = [str(ticker).upper() for ticker in feature_spec["asset_order"]]
    observation_columns = [f"obs_{index:03d}" for index in range(observation_dim)]
    return_columns = [f"return_{ticker.lower()}_1d" for ticker in asset_order]
    return ["date", "split", "feature_version", *observation_columns, *return_columns]


def _require_columns(frame: pd.DataFrame, columns: list[str], frame_name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{frame_name} missing required columns: {missing}")


def _timing_contract() -> dict[str, str]:
    return {
        "observation_timestamp": "close_of_date",
        "return_columns": "same_date_close_to_close_daily_returns_ending_on_date",
        "phase2_reward_rule": (
            "for a decision at row i, compound return rows after row i across "
            "the rebalance window"
        ),
    }


def _duckdb_tables(database_path: Path) -> set[str]:
    with duckdb.connect(str(database_path), read_only=True) as connection:
        rows = connection.execute("SHOW TABLES").fetchall()
    return {str(row[0]) for row in rows}


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from scripts.validate_phase1 import REQUIRED_DUCKDB_TABLES, validate_phase1_artifacts


def test_validate_phase1_allows_raw_nans_as_diagnostics(tmp_path: Path) -> None:
    _write_phase1_artifacts(tmp_path, raw_price_open=None)

    summary = validate_phase1_artifacts(tmp_path)

    assert summary["status"] == "ok"
    assert summary["raw_artifacts"]["prices_daily"]["missing_cell_count"] == 1
    assert summary["clean_artifacts"]["model_matrix_daily"]["missing_cell_count"] == 0


def test_validate_phase1_rejects_interim_nans(tmp_path: Path) -> None:
    _write_phase1_artifacts(tmp_path, interim_value=None)

    with pytest.raises(
        ValueError,
        match="aligned_feature_panel_daily contains NaN values: 1",
    ):
        validate_phase1_artifacts(tmp_path)


def test_validate_phase1_rejects_processed_nans(tmp_path: Path) -> None:
    _write_phase1_artifacts(tmp_path, feature_value=None)

    with pytest.raises(ValueError, match="features_daily contains NaN values: 1"):
        validate_phase1_artifacts(tmp_path)


def test_validate_phase1_rejects_model_matrix_nans(tmp_path: Path) -> None:
    _write_phase1_artifacts(tmp_path, model_obs=None, report_nan_count=1)

    with pytest.raises(ValueError, match="model_matrix_daily contains NaN values: 1"):
        validate_phase1_artifacts(tmp_path)


def test_validate_phase1_still_rejects_stale_report(tmp_path: Path) -> None:
    _write_phase1_artifacts(tmp_path, report_row_count=2)

    with pytest.raises(
        ValueError,
        match="model_matrix_daily row count does not match data quality report",
    ):
        validate_phase1_artifacts(tmp_path)


def test_validate_phase1_rejects_missing_observation_column(tmp_path: Path) -> None:
    _write_phase1_artifacts(
        tmp_path,
        model_matrix_columns=["date", "split", "feature_version", "return_spy_1d"],
    )

    with pytest.raises(
        ValueError,
        match="model_matrix_daily columns do not match feature_spec order",
    ):
        validate_phase1_artifacts(tmp_path)


def test_validate_phase1_rejects_extra_observation_column(tmp_path: Path) -> None:
    _write_phase1_artifacts(
        tmp_path,
        model_matrix_columns=[
            "date",
            "split",
            "feature_version",
            "obs_000",
            "obs_001",
            "return_spy_1d",
        ],
    )

    with pytest.raises(
        ValueError,
        match="model_matrix_daily columns do not match feature_spec order",
    ):
        validate_phase1_artifacts(tmp_path)


def test_validate_phase1_rejects_reordered_observation_columns(tmp_path: Path) -> None:
    _write_phase1_artifacts(
        tmp_path,
        observation_dim=2,
        model_matrix_columns=[
            "date",
            "split",
            "feature_version",
            "obs_001",
            "obs_000",
            "return_spy_1d",
        ],
    )

    with pytest.raises(
        ValueError,
        match="model_matrix_daily columns do not match feature_spec order",
    ):
        validate_phase1_artifacts(tmp_path)


def test_validate_phase1_rejects_wrong_return_column(tmp_path: Path) -> None:
    _write_phase1_artifacts(
        tmp_path,
        model_matrix_columns=[
            "date",
            "split",
            "feature_version",
            "obs_000",
            "return_qqq_1d",
        ],
    )

    with pytest.raises(
        ValueError,
        match="model_matrix_daily columns do not match feature_spec order",
    ):
        validate_phase1_artifacts(tmp_path)


def test_validate_phase1_rejects_misaligned_return_values(tmp_path: Path) -> None:
    _write_phase1_artifacts(tmp_path, feature_value=0.01, model_return=0.02)

    with pytest.raises(
        ValueError,
        match="return columns do not match same-date features_daily ret_1d",
    ):
        validate_phase1_artifacts(tmp_path)


def test_validate_phase1_rejects_unsorted_model_matrix_dates(tmp_path: Path) -> None:
    _write_phase1_artifacts(
        tmp_path,
        dates=[pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")],
        model_dates=[pd.Timestamp("2024-01-03"), pd.Timestamp("2024-01-02")],
    )

    with pytest.raises(ValueError, match="dates must be sorted ascending"):
        validate_phase1_artifacts(tmp_path)


def test_validate_phase1_rejects_duplicate_model_matrix_dates(tmp_path: Path) -> None:
    _write_phase1_artifacts(
        tmp_path,
        dates=[pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")],
        model_dates=[pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-02")],
    )

    with pytest.raises(ValueError, match="contains duplicate dates"):
        validate_phase1_artifacts(tmp_path)


def test_validate_phase1_rejects_forward_looking_feature_names(tmp_path: Path) -> None:
    _write_phase1_artifacts(tmp_path, per_asset_features=["next_ret_1d"])

    with pytest.raises(
        ValueError,
        match="feature_spec contains forward-looking feature names",
    ):
        validate_phase1_artifacts(tmp_path)


def _write_phase1_artifacts(
    root: Path,
    *,
    dates: list[pd.Timestamp] | None = None,
    model_dates: list[pd.Timestamp] | None = None,
    raw_price_open: float | None = 100.0,
    interim_value: float | None = 1.0,
    feature_value: float | None = 0.01,
    model_obs: float | None = 1.0,
    model_return: float | None = 0.01,
    observation_dim: int = 1,
    model_matrix_columns: list[str] | None = None,
    per_asset_features: list[str] | None = None,
    report_nan_count: int = 0,
    report_row_count: int | None = None,
) -> None:
    if dates is None:
        dates = [pd.Timestamp("2024-01-02")]
    if model_dates is None:
        model_dates = dates
    row_count = len(dates)
    model_row_count = len(model_dates)
    raw_prices = pd.DataFrame(
        {
            "date": dates,
            "ticker": ["SPY"] * row_count,
            "open": [raw_price_open] * row_count,
            "adj_close": [100.0] * row_count,
        }
    )
    raw_macro = pd.DataFrame(
        {
            "date": dates,
            "series_id": ["VIXCLS"] * row_count,
            "value": [15.0] * row_count,
        }
    )
    interim = pd.DataFrame(
        {
            "date": dates,
            "ticker": ["SPY"] * row_count,
            "feature_version": ["v1"] * row_count,
            "ret_1d": [interim_value] * row_count,
        }
    )
    features = pd.DataFrame(
        {
            "date": dates,
            "ticker": ["SPY"] * row_count,
            "split": ["validation"] * row_count,
            "feature_version": ["v1"] * row_count,
            "ret_1d": [feature_value] * row_count,
        }
    )
    global_features = pd.DataFrame(
        {
            "date": dates,
            "split": ["validation"] * row_count,
            "feature_version": ["v1"] * row_count,
            "vix_z_21d": [1.0] * row_count,
        }
    )
    model_matrix_data = {
        "date": model_dates,
        "split": ["validation"] * model_row_count,
        "feature_version": ["v1"] * model_row_count,
        "obs_000": [model_obs] * model_row_count,
        "obs_001": [2.0] * model_row_count,
        "return_spy_1d": [model_return] * model_row_count,
        "return_qqq_1d": [0.02] * model_row_count,
    }
    if model_matrix_columns is None:
        model_matrix_columns = [
            "date",
            "split",
            "feature_version",
            "obs_000",
            "return_spy_1d",
        ]
    model_matrix = pd.DataFrame(
        {column: model_matrix_data[column] for column in model_matrix_columns}
    )

    _write_parquet(root / "data/raw/prices_daily.parquet", raw_prices)
    _write_parquet(root / "data/raw/macro_daily.parquet", raw_macro)
    _write_parquet(root / "data/interim/aligned_feature_panel_daily.parquet", interim)
    _write_parquet(root / "data/processed/features_daily.parquet", features)
    _write_parquet(root / "data/processed/global_features_daily.parquet", global_features)
    _write_parquet(root / "data/processed/features_normalized_daily.parquet", features)
    _write_parquet(
        root / "data/processed/global_features_normalized_daily.parquet",
        global_features,
    )
    _write_parquet(root / "data/processed/model_matrix_daily.parquet", model_matrix)

    _write_json(
        root / "artifacts/feature_specs/feature_spec_v1.json",
        {
            "observation_dim": observation_dim,
            "asset_order": ["SPY"],
            "per_asset_features": per_asset_features or ["ret_1d"],
            "global_features": ["vix_z_21d"],
            "current_weight_features": ["weight_spy"],
        },
    )
    if report_row_count is None:
        report_row_count = len(model_matrix)
    _write_json(
        root / "artifacts/reports/data_quality_report_v1.json",
        {
            "model_matrix_row_count": report_row_count,
            "nan_count_final": report_nan_count,
            "inf_count_final": 0,
            "processed_artifacts": {
                "model_matrix_daily": {
                    "row_count": report_row_count,
                    "missing_cell_count": report_nan_count,
                    "inf_count": 0,
                }
            },
        },
    )
    scaler_path = root / "artifacts/scalers/feature_scaler_v1.pkl"
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    scaler_path.write_bytes(b"test")
    _write_duckdb(root / "data/duckdb/portfolio.duckdb")


def _write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_duckdb(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(path)) as connection:
        for table_name in REQUIRED_DUCKDB_TABLES:
            connection.execute(f'CREATE TABLE "{table_name}" (id INTEGER)')

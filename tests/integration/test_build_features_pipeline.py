from __future__ import annotations

from datetime import datetime, timezone
import json
from math import isfinite, sin
from pathlib import Path

import pandas as pd

from portfolio_rl.config.schemas import DataConfig, FeaturesConfig, UniverseConfig
from portfolio_rl.data.storage import (
    read_duckdb_table,
    read_parquet,
    write_parquet,
)
from portfolio_rl.features.feature_spec import load_feature_spec
from portfolio_rl.features.normalization import (
    NormalizationArtifactBundle,
    load_normalization_artifact,
)
from portfolio_rl.features.pipeline import build_feature_artifacts


def test_build_feature_artifacts_writes_processed_parquet_and_duckdb(
    tmp_path: Path,
) -> None:
    data_config = _data_config(tmp_path)
    feature_config = _feature_config()
    universe_config = _universe_config()
    write_parquet(
        _prices_fixture(),
        data_config.storage.raw_parquet_dir / "prices_daily.parquet",
    )
    write_parquet(
        _macro_fixture(),
        data_config.storage.raw_parquet_dir / "macro_daily.parquet",
    )
    scaler_artifact_path = tmp_path / "artifacts" / "scalers" / "feature_scaler_v1.pkl"
    feature_spec_path = tmp_path / "artifacts" / "feature_specs" / "feature_spec_v1.json"
    data_quality_report_path = (
        tmp_path / "artifacts" / "reports" / "data_quality_report_v1.json"
    )

    result = build_feature_artifacts(
        data_config=data_config,
        feature_config=feature_config,
        universe_config=universe_config,
        scaler_artifact_path=scaler_artifact_path,
        feature_spec_path=feature_spec_path,
        data_quality_report_path=data_quality_report_path,
    )

    features = read_parquet(result.features_parquet_path)
    global_features = read_parquet(result.global_features_parquet_path)
    normalized_features = read_parquet(result.normalized_features_parquet_path)
    normalized_global_features = read_parquet(
        result.normalized_global_features_parquet_path
    )
    model_matrix = read_parquet(result.model_matrix_parquet_path)
    features_table = read_duckdb_table(result.duckdb_path, "features_daily")
    global_features_table = read_duckdb_table(
        result.duckdb_path,
        "global_features_daily",
    )
    normalized_features_table = read_duckdb_table(
        result.duckdb_path,
        "features_normalized_daily",
    )
    normalized_global_features_table = read_duckdb_table(
        result.duckdb_path,
        "global_features_normalized_daily",
    )
    model_matrix_table = read_duckdb_table(result.duckdb_path, "model_matrix_daily")
    scaler_artifact = load_normalization_artifact(result.scaler_artifact_path)
    feature_spec = load_feature_spec(result.feature_spec_path)
    data_quality_report = json.loads(result.data_quality_report_path.read_text())

    assert result.features_parquet_path == (
        tmp_path / "processed" / "features_daily.parquet"
    )
    assert result.global_features_parquet_path == (
        tmp_path / "processed" / "global_features_daily.parquet"
    )
    assert result.normalized_features_parquet_path == (
        tmp_path / "processed" / "features_normalized_daily.parquet"
    )
    assert result.normalized_global_features_parquet_path == (
        tmp_path / "processed" / "global_features_normalized_daily.parquet"
    )
    assert result.model_matrix_parquet_path == (
        tmp_path / "processed" / "model_matrix_daily.parquet"
    )
    assert result.scaler_artifact_path == scaler_artifact_path
    assert result.feature_spec_path == feature_spec_path
    assert result.feature_spec_path.exists()
    assert result.data_quality_report_path == data_quality_report_path
    assert result.data_quality_report_path.exists()
    assert result.features_row_count == len(features)
    assert result.global_features_row_count == len(global_features)
    assert result.normalized_features_row_count == len(normalized_features)
    assert result.normalized_global_features_row_count == len(
        normalized_global_features
    )
    assert result.model_matrix_row_count == len(model_matrix)
    assert len(features_table) == len(features)
    assert len(global_features_table) == len(global_features)
    assert len(normalized_features_table) == len(normalized_features)
    assert len(normalized_global_features_table) == len(normalized_global_features)
    assert len(model_matrix_table) == len(model_matrix)
    assert set(features["date"]) == set(global_features["date"])
    assert pd.to_datetime(features["date"]).dt.date.min() >= data_config.model_start_date
    assert pd.to_datetime(global_features["date"]).dt.date.min() >= (
        data_config.model_start_date
    )
    assert set(features["split"]) == {"train", "validation", "test"}
    assert set(global_features["split"]) == {"train", "validation", "test"}
    assert set(features_table["split"]) == {"train", "validation", "test"}
    assert set(global_features_table["split"]) == {"train", "validation", "test"}
    assert set(normalized_features["split"]) == {"train", "validation", "test"}
    assert set(normalized_global_features["split"]) == {
        "train",
        "validation",
        "test",
    }
    assert {"ret_252d", "corr_to_spy_63d"}.issubset(features.columns)
    assert "spy_drawdown_63d" in global_features.columns
    assert not features.drop(columns=["date", "ticker", "feature_version"]).isna().any().any()
    assert not global_features.drop(columns=["date", "feature_version"]).isna().any().any()
    assert not normalized_features.drop(columns=["date", "ticker", "feature_version"]).isna().any().any()
    assert not normalized_global_features.drop(columns=["date", "feature_version"]).isna().any().any()
    assert isinstance(scaler_artifact, NormalizationArtifactBundle)
    assert scaler_artifact.asset_features.fit_split == "train"
    assert scaler_artifact.global_features.fit_split == "train"
    assert feature_spec.feature_version == feature_config.feature_version
    assert feature_spec.asset_order == universe_config.tickers
    assert set(feature_spec.per_asset_features).issubset(normalized_features.columns)
    assert set(feature_spec.global_features).issubset(normalized_global_features.columns)
    assert feature_spec.observation_dim == (
        len(feature_spec.asset_order) * len(feature_spec.per_asset_features)
        + len(feature_spec.global_features)
        + len(feature_spec.current_weight_features)
    )
    assert data_quality_report["universe_name"] == universe_config.universe_name
    assert data_quality_report["feature_version"] == feature_config.feature_version
    assert data_quality_report["n_assets"] == len(universe_config.tickers)
    assert data_quality_report["model_start_date"] == (
        data_config.model_start_date.isoformat()
    )
    assert data_quality_report["train_end_date"] == (
        data_config.train_end_date.isoformat()
    )
    assert data_quality_report["validation_start_date"] == (
        data_config.validation_start_date.isoformat()
    )
    assert data_quality_report["test_start_date"] == (
        data_config.test_start_date.isoformat()
    )
    assert data_quality_report["nan_count_final"] == 0
    assert data_quality_report["inf_count_final"] == 0
    assert data_quality_report["normalization_fit_split"] == "train"
    assert data_quality_report["model_matrix_row_count"] == len(model_matrix)
    assert data_quality_report["observation_dim"] == feature_spec.observation_dim
    assert data_quality_report["raw_prices"]["missing_cell_count"] == 1
    assert data_quality_report["raw_prices"]["missing_count_by_ticker"] == {"SPY": 1}
    assert data_quality_report["raw_prices"]["missing_count_by_column"] == {"open": 1}
    assert data_quality_report["raw_macro"]["missing_cell_count"] == 1
    assert data_quality_report["raw_macro"]["missing_value_count_by_series"] == {
        "VIXCLS": 1
    }
    assert data_quality_report["processed_artifacts"]["features_daily"][
        "missing_cell_count"
    ] == 0
    assert data_quality_report["processed_artifacts"]["model_matrix_daily"][
        "missing_cell_count"
    ] == 0
    assert data_quality_report["processed_artifacts"]["model_matrix_daily"][
        "inf_count"
    ] == 0
    obs_columns = [column for column in model_matrix.columns if column.startswith("obs_")]
    return_columns = [
        f"return_{ticker.lower()}_1d" for ticker in feature_spec.asset_order
    ]
    assert len(model_matrix) == len(normalized_global_features)
    assert set(model_matrix["date"]) == set(normalized_global_features["date"])
    assert len(obs_columns) == feature_spec.observation_dim
    assert return_columns == [
        column for column in model_matrix.columns if column.startswith("return_")
    ]
    assert model_matrix[["date", "split", "feature_version"]].equals(
        normalized_global_features[["date", "split", "feature_version"]]
    )
    assert not model_matrix[obs_columns + return_columns].isna().any().any()
    assert model_matrix[obs_columns + return_columns].map(isfinite).all().all()
    assert not features[scaler_artifact.asset_features.feature_columns].equals(
        normalized_features[scaler_artifact.asset_features.feature_columns]
    )
    assert normalized_features[["date", "ticker", "split", "feature_version"]].equals(
        features[["date", "ticker", "split", "feature_version"]]
    )
    assert normalized_global_features[["date", "split", "feature_version"]].equals(
        global_features[["date", "split", "feature_version"]]
    )
    asset_train = normalized_features.loc[normalized_features["split"] == "train"]
    global_train = normalized_global_features.loc[
        normalized_global_features["split"] == "train"
    ]
    assert (
        asset_train[scaler_artifact.asset_features.feature_columns].mean().abs().max()
        < 1e-10
    )
    assert (
        global_train[scaler_artifact.global_features.feature_columns].mean().abs().max()
        < 1e-10
    )


def _data_config(tmp_path: Path) -> DataConfig:
    return DataConfig.model_validate(
        {
            "raw_start_date": "2007-01-01",
            "model_start_date": "2010-01-01",
            "train_start_date": "2010-01-01",
            "train_end_date": "2023-12-31",
            "validation_start_date": "2024-01-01",
            "validation_end_date": "2024-12-31",
            "test_start_date": "2025-01-01",
            "test_end_date": None,
            "market_data_source": "yfinance",
            "macro_data_source": "fred",
            "macro_series": [
                {
                    "series_id": "VIXCLS",
                    "description": "CBOE Volatility Index",
                    "frequency": "daily",
                }
            ],
            "storage": {
                "duckdb_path": tmp_path / "duckdb" / "portfolio.duckdb",
                "raw_parquet_dir": tmp_path / "raw",
                "interim_parquet_dir": tmp_path / "interim",
                "processed_parquet_dir": tmp_path / "processed",
            },
        }
    )


def _feature_config() -> FeaturesConfig:
    return FeaturesConfig.model_validate(
        {
            "feature_version": "v1",
            "return_windows": [1, 5, 21, 63, 126, 252],
            "volatility_windows": [21, 63],
            "drawdown_windows": [63],
            "rsi_windows": [14],
            "price_z_windows": [20, 50],
            "correlation_windows": [63],
            "winsorization": {
                "enabled": True,
                "lower_quantile": 0.005,
                "upper_quantile": 0.995,
            },
            "normalization": {
                "method": "standard_scaler",
                "fit_split": "train",
            },
        }
    )


def _universe_config() -> UniverseConfig:
    return UniverseConfig.model_validate(
        {
            "universe_name": "test_universe",
            "assets": [
                {"ticker": "SPY", "asset_class": "us_large_cap_equity"},
                {"ticker": "QQQ", "asset_class": "us_growth_equity"},
            ],
        }
    )


def _prices_fixture() -> pd.DataFrame:
    downloaded_at = datetime(2026, 4, 27, tzinfo=timezone.utc)
    dates = pd.bdate_range("2022-01-03", periods=900)
    frames = []
    for ticker, base, phase in (("SPY", 100.0, 0.0), ("QQQ", 150.0, 1.7)):
        rows = []
        for index, date in enumerate(dates):
            adj_close = base + index * 0.35 + sin(index / 3.0 + phase) * 2.0
            close = adj_close * 1.001
            rows.append(
                {
                    "date": date.date(),
                    "ticker": ticker,
                    "open": None if ticker == "SPY" and index == 5 else close * 0.99,
                    "high": close * 1.01,
                    "low": close * 0.98,
                    "close": close,
                    "adj_close": adj_close,
                    "volume": 1_000_000 + index * 1_000,
                    "dividends": 0.0,
                    "stock_splits": 0.0,
                    "source": "test",
                    "downloaded_at": downloaded_at,
                }
            )
        frames.append(pd.DataFrame(rows))
    return pd.concat(frames, ignore_index=True)


def _macro_fixture() -> pd.DataFrame:
    downloaded_at = datetime(2026, 4, 27, tzinfo=timezone.utc)
    dates = pd.bdate_range("2022-01-03", periods=900)
    series_offsets = {
        "VIXCLS": 20.0,
        "DGS2": 4.0,
        "DGS10": 4.5,
        "T10Y2Y": 0.5,
        "BAMLH0A0HYM2": 3.5,
    }
    rows = []
    for series_id, base in series_offsets.items():
        for index, date in enumerate(dates):
            value = base + sin(index / 7.0) * 0.2 + index * 0.001
            rows.append(
                {
                    "date": date.date(),
                    "series_id": series_id,
                    "value": None if series_id == "VIXCLS" and index == 10 else value,
                    "source": "test",
                    "downloaded_at": downloaded_at,
                }
            )
    return pd.DataFrame(rows)

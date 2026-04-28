from __future__ import annotations

from datetime import datetime, timezone
from math import sin
from pathlib import Path

import pandas as pd

from portfolio_rl.config.schemas import DataConfig, FeaturesConfig, UniverseConfig
from portfolio_rl.data.storage import (
    read_duckdb_table,
    read_parquet,
    write_parquet,
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

    result = build_feature_artifacts(
        data_config=data_config,
        feature_config=feature_config,
        universe_config=universe_config,
    )

    features = read_parquet(result.features_parquet_path)
    global_features = read_parquet(result.global_features_parquet_path)
    features_table = read_duckdb_table(result.duckdb_path, "features_daily")
    global_features_table = read_duckdb_table(
        result.duckdb_path,
        "global_features_daily",
    )

    assert result.features_parquet_path == (
        tmp_path / "processed" / "features_daily.parquet"
    )
    assert result.global_features_parquet_path == (
        tmp_path / "processed" / "global_features_daily.parquet"
    )
    assert result.features_row_count == len(features)
    assert result.global_features_row_count == len(global_features)
    assert len(features_table) == len(features)
    assert len(global_features_table) == len(global_features)
    assert set(features["date"]) == set(global_features["date"])
    assert {"ret_252d", "corr_to_spy_63d"}.issubset(features.columns)
    assert "spy_drawdown_63d" in global_features.columns
    assert not features.drop(columns=["date", "ticker", "feature_version"]).isna().any().any()
    assert not global_features.drop(columns=["date", "feature_version"]).isna().any().any()


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
    dates = pd.bdate_range("2023-01-02", periods=340)
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
                    "open": close * 0.99,
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
    dates = pd.bdate_range("2023-01-02", periods=340)
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
            rows.append(
                {
                    "date": date.date(),
                    "series_id": series_id,
                    "value": base + sin(index / 7.0) * 0.2 + index * 0.001,
                    "source": "test",
                    "downloaded_at": downloaded_at,
                }
            )
    return pd.DataFrame(rows)

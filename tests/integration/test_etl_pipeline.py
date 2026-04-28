from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from portfolio_rl.config.schemas import DataConfig, UniverseConfig
from portfolio_rl.data.etl import run_raw_etl
from portfolio_rl.data.storage import read_duckdb_table, read_parquet


def test_run_raw_etl_writes_raw_parquet_and_duckdb_tables(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls: dict[str, object] = {}

    def fake_download_prices(tickers, start_date, end_date):
        calls["price_args"] = (list(tickers), start_date, end_date)
        return _prices_fixture()

    def fake_download_macro_series(series, start_date, end_date):
        calls["macro_args"] = (
            [series_config.series_id for series_config in series],
            start_date,
            end_date,
        )
        return _macro_fixture()

    def fake_validate_prices_daily(prices):
        calls["validated_price_rows"] = len(prices)
        return prices

    monkeypatch.setattr(
        "portfolio_rl.data.etl.download_prices",
        fake_download_prices,
    )
    monkeypatch.setattr(
        "portfolio_rl.data.etl.download_macro_series",
        fake_download_macro_series,
    )
    monkeypatch.setattr(
        "portfolio_rl.data.etl.validate_prices_daily",
        fake_validate_prices_daily,
    )

    data_config = _data_config(tmp_path)
    universe_config = UniverseConfig.model_validate(
        {
            "universe_name": "test_universe",
            "assets": [
                {"ticker": "SPY", "asset_class": "us_large_cap_equity"},
                {"ticker": "QQQ", "asset_class": "us_growth_equity"},
            ],
        }
    )

    result = run_raw_etl(data_config, universe_config)

    assert calls["price_args"] == (
        ["SPY", "QQQ"],
        data_config.raw_start_date,
        data_config.test_end_date,
    )
    assert calls["macro_args"] == (
        ["VIXCLS"],
        data_config.raw_start_date,
        data_config.test_end_date,
    )
    assert calls["validated_price_rows"] == 4
    assert result.prices_row_count == 4
    assert result.macro_row_count == 2
    assert result.prices_parquet_path == tmp_path / "raw" / "prices_daily.parquet"
    assert result.macro_parquet_path == tmp_path / "raw" / "macro_daily.parquet"
    assert result.duckdb_path == tmp_path / "duckdb" / "portfolio.duckdb"

    prices_round_trip = read_parquet(result.prices_parquet_path)
    macro_round_trip = read_parquet(result.macro_parquet_path)
    prices_table = read_duckdb_table(result.duckdb_path, "prices_daily")
    macro_table = read_duckdb_table(result.duckdb_path, "macro_daily")

    assert len(prices_round_trip) == 4
    assert len(macro_round_trip) == 2
    assert len(prices_table) == 4
    assert len(macro_table) == 2


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


def _prices_fixture() -> pd.DataFrame:
    downloaded_at = datetime(2026, 4, 26, tzinfo=timezone.utc)
    return pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-03", "2024-01-02", "2024-01-03"],
            "ticker": ["SPY", "SPY", "QQQ", "QQQ"],
            "open": [100.0, 101.0, 200.0, 201.0],
            "high": [102.0, 103.0, 202.0, 203.0],
            "low": [99.0, 100.0, 199.0, 200.0],
            "close": [101.0, 102.0, 201.0, 202.0],
            "adj_close": [101.0, 102.0, 201.0, 202.0],
            "volume": [1000, 1100, 2000, 2100],
            "dividends": [0.0, 0.0, 0.0, 0.0],
            "stock_splits": [0.0, 0.0, 0.0, 0.0],
            "source": ["test", "test", "test", "test"],
            "downloaded_at": [downloaded_at] * 4,
        }
    )


def _macro_fixture() -> pd.DataFrame:
    downloaded_at = datetime(2026, 4, 26, tzinfo=timezone.utc)
    return pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-03"],
            "series_id": ["VIXCLS", "VIXCLS"],
            "value": [13.2, 14.1],
            "source": ["test", "test"],
            "downloaded_at": [downloaded_at] * 2,
        }
    )

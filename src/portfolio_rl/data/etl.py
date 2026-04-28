"""Raw ETL orchestration for Phase 1 market and macro data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from portfolio_rl.config.schemas import DataConfig, UniverseConfig
from portfolio_rl.data.fred_client import download_macro_series
from portfolio_rl.data.storage import write_duckdb_table, write_parquet
from portfolio_rl.data.validation import validate_prices_daily
from portfolio_rl.data.yfinance_client import download_prices


@dataclass(frozen=True)
class RawEtlResult:
    """Output locations and row counts from raw ETL."""

    prices_parquet_path: Path
    macro_parquet_path: Path
    duckdb_path: Path
    prices_row_count: int
    macro_row_count: int


def run_raw_etl(
    data_config: DataConfig,
    universe_config: UniverseConfig,
) -> RawEtlResult:
    """Download, validate, and persist raw Phase 1 data artifacts."""
    prices = download_prices(
        tickers=universe_config.tickers,
        start_date=data_config.raw_start_date,
        end_date=data_config.test_end_date,
    )
    prices = validate_prices_daily(prices)
    macro = download_macro_series(
        series=data_config.macro_series,
        start_date=data_config.raw_start_date,
        end_date=data_config.test_end_date,
    )

    prices_path = data_config.storage.raw_parquet_dir / "prices_daily.parquet"
    macro_path = data_config.storage.raw_parquet_dir / "macro_daily.parquet"
    write_parquet(prices, prices_path)
    write_parquet(macro, macro_path)

    duckdb_path = data_config.storage.duckdb_path
    write_duckdb_table(prices, duckdb_path, "prices_daily")
    write_duckdb_table(macro, duckdb_path, "macro_daily")

    return RawEtlResult(
        prices_parquet_path=prices_path,
        macro_parquet_path=macro_path,
        duckdb_path=duckdb_path,
        prices_row_count=len(prices),
        macro_row_count=len(macro),
    )

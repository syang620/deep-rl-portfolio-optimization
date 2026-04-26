"""Validation helpers for Phase 1 raw market data."""

from __future__ import annotations

from math import log
from warnings import warn

import pandas as pd

from portfolio_rl.data.yfinance_client import PRICE_COLUMNS


PRICE_DAILY_COLUMNS = tuple(PRICE_COLUMNS)
POSITIVE_PRICE_COLUMNS = ("open", "high", "low", "close", "adj_close")
OUTLIER_RETURN_THRESHOLD = 0.25


def validate_prices_daily(prices: pd.DataFrame) -> pd.DataFrame:
    """Validate raw long-format daily price data without modifying rows."""
    _validate_required_columns(prices)
    _validate_no_duplicate_ticker_dates(prices)
    _validate_dates_sorted_by_ticker(prices)
    _validate_positive_prices(prices)
    _validate_high_low_order(prices)
    _validate_non_negative_volume(prices)
    _warn_on_large_returns(prices)
    return prices


def _validate_required_columns(prices: pd.DataFrame) -> None:
    missing = [column for column in PRICE_DAILY_COLUMNS if column not in prices.columns]
    if missing:
        raise ValueError(f"prices_daily is missing required columns: {missing}")


def _validate_no_duplicate_ticker_dates(prices: pd.DataFrame) -> None:
    duplicates = prices.duplicated(subset=["ticker", "date"], keep=False)
    if duplicates.any():
        raise ValueError("prices_daily contains duplicate ticker-date rows")


def _validate_dates_sorted_by_ticker(prices: pd.DataFrame) -> None:
    dates = pd.to_datetime(prices["date"])
    for ticker, ticker_dates in dates.groupby(prices["ticker"], sort=False):
        if not ticker_dates.is_monotonic_increasing:
            raise ValueError(f"prices_daily dates are not sorted for ticker: {ticker}")


def _validate_positive_prices(prices: pd.DataFrame) -> None:
    invalid_mask = prices.loc[:, POSITIVE_PRICE_COLUMNS].le(0).any(axis=1)
    if invalid_mask.any():
        raise ValueError("prices_daily contains non-positive OHLC or adjusted prices")


def _validate_high_low_order(prices: pd.DataFrame) -> None:
    if (prices["high"] < prices["low"]).any():
        raise ValueError("prices_daily contains rows where high is less than low")


def _validate_non_negative_volume(prices: pd.DataFrame) -> None:
    if (prices["volume"] < 0).any():
        raise ValueError("prices_daily contains negative volume")


def _warn_on_large_returns(prices: pd.DataFrame) -> None:
    sorted_prices = prices.sort_values(["ticker", "date"])
    adj_close = sorted_prices.groupby("ticker", sort=False)["adj_close"]
    log_returns = (adj_close.transform(lambda values: values / values.shift(1))).map(
        lambda value: log(value) if pd.notna(value) else value
    )
    if log_returns.abs().gt(OUTLIER_RETURN_THRESHOLD).any():
        warn(
            "prices_daily contains rows where abs(ret_1d) exceeds 0.25",
            RuntimeWarning,
            stacklevel=2,
        )

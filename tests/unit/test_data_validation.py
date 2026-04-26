from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from portfolio_rl.data.validation import validate_prices_daily


def test_validate_prices_daily_accepts_valid_prices() -> None:
    prices = _valid_prices()

    result = validate_prices_daily(prices)

    assert result is prices


def test_validate_prices_daily_fails_on_missing_required_column() -> None:
    prices = _valid_prices().drop(columns=["adj_close"])

    with pytest.raises(ValueError, match="missing required columns"):
        validate_prices_daily(prices)


def test_validate_prices_daily_fails_on_duplicate_ticker_date() -> None:
    prices = pd.concat([_valid_prices(), _valid_prices().iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="duplicate ticker-date"):
        validate_prices_daily(prices)


def test_validate_prices_daily_fails_on_non_positive_price() -> None:
    prices = _valid_prices()
    prices.loc[0, "close"] = 0.0

    with pytest.raises(ValueError, match="non-positive"):
        validate_prices_daily(prices)


def test_validate_prices_daily_fails_when_dates_unsorted_per_ticker() -> None:
    prices = _valid_prices()
    prices.loc[[0, 1], "date"] = prices.loc[[1, 0], "date"].to_list()

    with pytest.raises(ValueError, match="not sorted"):
        validate_prices_daily(prices)


def test_validate_prices_daily_fails_when_high_is_less_than_low() -> None:
    prices = _valid_prices()
    prices.loc[0, "high"] = prices.loc[0, "low"] - 1.0

    with pytest.raises(ValueError, match="high is less than low"):
        validate_prices_daily(prices)


def test_validate_prices_daily_fails_on_negative_volume() -> None:
    prices = _valid_prices()
    prices.loc[0, "volume"] = -1

    with pytest.raises(ValueError, match="negative volume"):
        validate_prices_daily(prices)


def test_validate_prices_daily_warns_on_large_return_without_deleting_rows() -> None:
    prices = _valid_prices()
    prices.loc[1, "adj_close"] = 150.0

    with pytest.warns(RuntimeWarning, match="abs\\(ret_1d\\) exceeds 0.25"):
        result = validate_prices_daily(prices)

    assert len(result) == len(prices)


def _valid_prices() -> pd.DataFrame:
    downloaded_at = datetime(2026, 4, 26, tzinfo=timezone.utc)
    return pd.DataFrame(
        {
            "date": [
                "2024-01-02",
                "2024-01-03",
                "2024-01-02",
                "2024-01-03",
            ],
            "ticker": ["SPY", "SPY", "QQQ", "QQQ"],
            "open": [100.0, 101.0, 200.0, 201.0],
            "high": [102.0, 103.0, 202.0, 203.0],
            "low": [99.0, 100.0, 199.0, 200.0],
            "close": [101.0, 102.0, 201.0, 202.0],
            "adj_close": [101.0, 102.0, 201.0, 202.0],
            "volume": [1000, 1100, 2000, 2100],
            "dividends": [0.0, 0.0, 0.0, 0.0],
            "stock_splits": [0.0, 0.0, 0.0, 0.0],
            "source": ["yfinance", "yfinance", "yfinance", "yfinance"],
            "downloaded_at": [downloaded_at] * 4,
        }
    )

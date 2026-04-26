"""Client helpers for downloading daily ETF price data from yfinance."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, datetime, timezone

import pandas as pd
import yfinance as yf


PRICE_COLUMNS = [
    "date",
    "ticker",
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",
    "dividends",
    "stock_splits",
    "source",
    "downloaded_at",
]


def download_prices(
    tickers: Sequence[str],
    start_date: date,
    end_date: date | None = None,
) -> pd.DataFrame:
    """Download daily OHLCV data and return the Phase 1 long-format schema."""
    ticker_list = _normalize_tickers(tickers)
    raw = yf.download(
        tickers=ticker_list,
        start=start_date.isoformat(),
        end=end_date.isoformat() if end_date is not None else None,
        actions=True,
        auto_adjust=False,
        group_by="ticker",
        progress=False,
        threads=False,
    )

    if raw.empty:
        raise ValueError("yfinance returned no price data")

    return _standardize_download(raw, ticker_list, datetime.now(timezone.utc))


def _normalize_tickers(tickers: Sequence[str]) -> list[str]:
    normalized = [ticker.strip().upper() for ticker in tickers]
    if not normalized or any(not ticker for ticker in normalized):
        raise ValueError("tickers must contain at least one non-empty ticker")
    if len(normalized) != len(set(normalized)):
        raise ValueError("tickers must not contain duplicates")
    return normalized


def _standardize_download(
    raw: pd.DataFrame,
    tickers: Sequence[str],
    downloaded_at: datetime,
) -> pd.DataFrame:
    frames = []
    for ticker in tickers:
        ticker_frame = _select_ticker_frame(raw, ticker, len(tickers))
        ticker_frame = ticker_frame.rename(
            columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Adj Close": "adj_close",
                "Volume": "volume",
                "Dividends": "dividends",
                "Stock Splits": "stock_splits",
            }
        )
        ticker_frame = ticker_frame.reset_index().rename(columns={"Date": "date"})
        ticker_frame["ticker"] = ticker
        ticker_frame["source"] = "yfinance"
        ticker_frame["downloaded_at"] = downloaded_at
        frames.append(ticker_frame)

    prices = pd.concat(frames, ignore_index=True)
    for column in ("dividends", "stock_splits"):
        if column not in prices:
            prices[column] = 0.0

    prices = prices[PRICE_COLUMNS].copy()
    prices["date"] = pd.to_datetime(prices["date"]).dt.date
    return prices.sort_values(["ticker", "date"], ignore_index=True)


def _select_ticker_frame(
    raw: pd.DataFrame,
    ticker: str,
    ticker_count: int,
) -> pd.DataFrame:
    if isinstance(raw.columns, pd.MultiIndex):
        if ticker not in raw.columns.get_level_values(0):
            raise ValueError(f"missing yfinance data for ticker: {ticker}")
        return raw[ticker].copy()

    if ticker_count != 1:
        raise ValueError("expected multi-index yfinance data for multiple tickers")
    return raw.copy()

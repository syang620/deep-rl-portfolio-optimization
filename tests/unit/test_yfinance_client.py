from __future__ import annotations

from datetime import date, datetime, timezone
from unittest.mock import patch

import pandas as pd
import pytest

from portfolio_rl.data.yfinance_client import PRICE_COLUMNS, download_prices


def test_download_prices_returns_long_format_dataframe() -> None:
    raw = _multi_ticker_download_frame()

    with (
        patch("portfolio_rl.data.yfinance_client.yf.download", return_value=raw) as mock,
        patch(
            "portfolio_rl.data.yfinance_client.datetime",
            wraps=datetime,
        ) as mock_datetime,
    ):
        mock_datetime.now.return_value = datetime(2026, 4, 26, tzinfo=timezone.utc)
        prices = download_prices(
            tickers=["spy", "QQQ"],
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 5),
        )

    mock.assert_called_once_with(
        tickers=["SPY", "QQQ"],
        start="2024-01-01",
        end="2024-01-05",
        actions=True,
        auto_adjust=False,
        group_by="ticker",
        progress=False,
        threads=False,
    )
    assert list(prices.columns) == PRICE_COLUMNS
    assert prices["ticker"].tolist() == ["QQQ", "QQQ", "SPY", "SPY"]
    assert prices["source"].unique().tolist() == ["yfinance"]
    assert prices["downloaded_at"].unique().tolist() == [
        datetime(2026, 4, 26, tzinfo=timezone.utc)
    ]
    assert prices.loc[prices["ticker"] == "SPY", "adj_close"].tolist() == [
        470.5,
        471.5,
    ]


def test_download_prices_rejects_duplicate_tickers() -> None:
    with pytest.raises(ValueError, match="duplicates"):
        download_prices(["SPY", "spy"], date(2024, 1, 1))


def test_download_prices_raises_when_yfinance_returns_empty() -> None:
    with patch(
        "portfolio_rl.data.yfinance_client.yf.download",
        return_value=pd.DataFrame(),
    ):
        with pytest.raises(ValueError, match="no price data"):
            download_prices(["SPY"], date(2024, 1, 1))


def test_download_prices_handles_single_ticker_dataframe() -> None:
    raw = pd.DataFrame(
        {
            "Open": [470.0],
            "High": [472.0],
            "Low": [469.0],
            "Close": [471.0],
            "Adj Close": [470.5],
            "Volume": [1000],
        },
        index=pd.DatetimeIndex(["2024-01-02"], name="Date"),
    )

    with (
        patch("portfolio_rl.data.yfinance_client.yf.download", return_value=raw),
        patch(
            "portfolio_rl.data.yfinance_client.datetime",
            wraps=datetime,
        ) as mock_datetime,
    ):
        mock_datetime.now.return_value = datetime(2026, 4, 26, tzinfo=timezone.utc)
        prices = download_prices(["SPY"], date(2024, 1, 1))

    assert prices.loc[0, "ticker"] == "SPY"
    assert prices.loc[0, "dividends"] == 0.0
    assert prices.loc[0, "stock_splits"] == 0.0


def _multi_ticker_download_frame() -> pd.DataFrame:
    columns = pd.MultiIndex.from_product(
        [
            ["SPY", "QQQ"],
            [
                "Open",
                "High",
                "Low",
                "Close",
                "Adj Close",
                "Volume",
                "Dividends",
                "Stock Splits",
            ],
        ]
    )
    return pd.DataFrame(
        [
            [
                470.0,
                472.0,
                469.0,
                471.0,
                470.5,
                1000,
                0.0,
                0.0,
                400.0,
                402.0,
                399.0,
                401.0,
                400.5,
                2000,
                0.0,
                0.0,
            ],
            [
                471.0,
                473.0,
                470.0,
                472.0,
                471.5,
                1100,
                0.1,
                0.0,
                401.0,
                403.0,
                400.0,
                402.0,
                401.5,
                2100,
                0.0,
                0.0,
            ],
        ],
        index=pd.DatetimeIndex(["2024-01-02", "2024-01-03"], name="Date"),
        columns=columns,
    )

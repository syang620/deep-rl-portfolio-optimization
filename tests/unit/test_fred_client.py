from __future__ import annotations

from datetime import date, datetime, timezone
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from portfolio_rl.config.schemas import MacroSeriesConfig
from portfolio_rl.data.fred_client import MACRO_COLUMNS, download_macro_series


def test_download_macro_series_returns_long_format_dataframe() -> None:
    series = [
        MacroSeriesConfig(
            series_id="VIXCLS",
            description="CBOE Volatility Index",
            frequency="daily",
        ),
        MacroSeriesConfig(
            series_id="DGS10",
            description="10-year Treasury constant maturity rate",
            frequency="daily",
        ),
    ]

    with (
        patch("portfolio_rl.data.fred_client.requests.get") as mock_get,
        patch(
            "portfolio_rl.data.fred_client.datetime",
            wraps=datetime,
        ) as mock_datetime,
    ):
        mock_datetime.now.return_value = datetime(2026, 4, 26, tzinfo=timezone.utc)
        mock_get.side_effect = [
            _fred_response(
                [
                    {"date": "2024-01-02", "value": "13.20"},
                    {"date": "2024-01-03", "value": "14.10"},
                ]
            ),
            _fred_response(
                [
                    {"date": "2024-01-02", "value": "3.95"},
                    {"date": "2024-01-03", "value": "3.91"},
                ]
            ),
        ]

        macro = download_macro_series(
            series=series,
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 5),
            api_key="test-key",
        )

    assert list(macro.columns) == MACRO_COLUMNS
    assert macro["series_id"].tolist() == ["DGS10", "DGS10", "VIXCLS", "VIXCLS"]
    assert macro["value"].tolist() == [3.95, 3.91, 13.20, 14.10]
    assert macro["source"].unique().tolist() == ["fred"]
    assert macro["downloaded_at"].unique().tolist() == [
        datetime(2026, 4, 26, tzinfo=timezone.utc)
    ]
    assert mock_get.call_count == 2
    assert mock_get.call_args_list[0].kwargs["params"]["series_id"] == "VIXCLS"
    assert mock_get.call_args_list[0].kwargs["params"]["observation_start"] == (
        "2024-01-01"
    )
    assert mock_get.call_args_list[0].kwargs["params"]["observation_end"] == (
        "2024-01-05"
    )
    assert mock_get.call_args_list[0].kwargs["params"]["api_key"] == "test-key"


def test_download_macro_series_parses_missing_values_as_nan() -> None:
    series = [
        MacroSeriesConfig(
            series_id="VIXCLS",
            description="CBOE Volatility Index",
            frequency="daily",
        )
    ]

    with patch("portfolio_rl.data.fred_client.requests.get") as mock_get:
        mock_get.return_value = _fred_response(
            [
                {"date": "2024-01-02", "value": "."},
                {"date": "2024-01-03", "value": ""},
                {"date": "2024-01-04", "value": None},
            ]
        )

        macro = download_macro_series(series, date(2024, 1, 1))

    assert macro["value"].isna().tolist() == [True, True, True]


def test_download_macro_series_uses_env_api_key_when_not_passed() -> None:
    series = [
        MacroSeriesConfig(
            series_id="VIXCLS",
            description="CBOE Volatility Index",
            frequency="daily",
        )
    ]

    with (
        patch.dict("os.environ", {"FRED_API_KEY": "env-key"}),
        patch("portfolio_rl.data.fred_client.requests.get") as mock_get,
    ):
        mock_get.return_value = _fred_response(
            [{"date": "2024-01-02", "value": "13.20"}]
        )

        download_macro_series(series, date(2024, 1, 1))

    assert mock_get.call_args.kwargs["params"]["api_key"] == "env-key"


def test_download_macro_series_raises_for_empty_series_list() -> None:
    with pytest.raises(ValueError, match="at least one"):
        download_macro_series([], date(2024, 1, 1))


def test_download_macro_series_raises_when_fred_returns_no_observations() -> None:
    series = [
        MacroSeriesConfig(
            series_id="VIXCLS",
            description="CBOE Volatility Index",
            frequency="daily",
        )
    ]

    with patch("portfolio_rl.data.fred_client.requests.get") as mock_get:
        mock_get.return_value = _fred_response([])

        with pytest.raises(ValueError, match="no observations"):
            download_macro_series(series, date(2024, 1, 1))


def test_download_macro_series_raises_for_http_error() -> None:
    series = [
        MacroSeriesConfig(
            series_id="VIXCLS",
            description="CBOE Volatility Index",
            frequency="daily",
        )
    ]
    response = Mock()
    response.raise_for_status.side_effect = RuntimeError("boom")

    with patch("portfolio_rl.data.fred_client.requests.get", return_value=response):
        with pytest.raises(RuntimeError, match="boom"):
            download_macro_series(series, date(2024, 1, 1))


def _fred_response(observations: list[dict[str, object]]) -> Mock:
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {"observations": observations}
    return response

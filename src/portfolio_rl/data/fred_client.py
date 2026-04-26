"""Client helpers for downloading daily macro series from FRED."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, datetime, timezone
import os

import pandas as pd
import requests

from portfolio_rl.config.schemas import MacroSeriesConfig


FRED_OBSERVATIONS_URL = "https://api.stlouisfed.org/fred/series/observations"
MACRO_COLUMNS = ["date", "series_id", "value", "source", "downloaded_at"]


def download_macro_series(
    series: Sequence[MacroSeriesConfig],
    start_date: date,
    end_date: date | None = None,
    api_key: str | None = None,
) -> pd.DataFrame:
    """Download configured FRED series and return the Phase 1 long-format schema."""
    if not series:
        raise ValueError("series must contain at least one macro series")

    resolved_api_key = api_key or os.getenv("FRED_API_KEY")
    downloaded_at = datetime.now(timezone.utc)
    frames = [
        _download_single_series(
            series_config=series_config,
            start_date=start_date,
            end_date=end_date,
            api_key=resolved_api_key,
            downloaded_at=downloaded_at,
        )
        for series_config in series
    ]
    macro = pd.concat(frames, ignore_index=True)
    macro["date"] = pd.to_datetime(macro["date"]).dt.date
    return macro[MACRO_COLUMNS].sort_values(["series_id", "date"], ignore_index=True)


def _download_single_series(
    series_config: MacroSeriesConfig,
    start_date: date,
    end_date: date | None,
    api_key: str | None,
    downloaded_at: datetime,
) -> pd.DataFrame:
    params = {
        "series_id": series_config.series_id,
        "observation_start": start_date.isoformat(),
        "file_type": "json",
    }
    if end_date is not None:
        params["observation_end"] = end_date.isoformat()
    if api_key is not None:
        params["api_key"] = api_key

    response = requests.get(FRED_OBSERVATIONS_URL, params=params, timeout=30)
    response.raise_for_status()
    observations = response.json().get("observations", [])
    if not observations:
        raise ValueError(f"FRED returned no observations for {series_config.series_id}")

    frame = pd.DataFrame(observations)
    frame["series_id"] = series_config.series_id
    frame["value"] = pd.to_numeric(frame["value"].replace([".", ""], pd.NA))
    frame["source"] = "fred"
    frame["downloaded_at"] = downloaded_at
    return frame[MACRO_COLUMNS]

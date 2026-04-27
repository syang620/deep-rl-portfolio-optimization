"""Return feature calculations for long-format daily prices."""

from __future__ import annotations

from collections.abc import Sequence
from math import log, nan

import pandas as pd


REQUIRED_PRICE_COLUMNS = ("date", "ticker", "adj_close")


def calculate_return_features(
    prices: pd.DataFrame,
    windows: Sequence[int],
) -> pd.DataFrame:
    """Calculate per-ticker log return features for the configured windows."""
    _validate_inputs(prices, windows)
    features = _prepare_prices(prices)

    grouped_prices = features.groupby("ticker", sort=False)["adj_close"]
    for window in windows:
        ratios = grouped_prices.transform(lambda values: values / values.shift(window))
        features[f"ret_{window}d"] = ratios.map(_safe_log)

    return features.drop(columns=["adj_close"])


def _prepare_prices(prices: pd.DataFrame) -> pd.DataFrame:
    features = prices.loc[:, REQUIRED_PRICE_COLUMNS].copy()
    features["date"] = pd.to_datetime(features["date"])
    features["adj_close"] = pd.to_numeric(features["adj_close"])
    return features.sort_values(["ticker", "date"], ignore_index=True)


def _validate_inputs(prices: pd.DataFrame, windows: Sequence[int]) -> None:
    missing = [column for column in REQUIRED_PRICE_COLUMNS if column not in prices.columns]
    if missing:
        raise ValueError(f"prices is missing required columns: {missing}")
    if not windows:
        raise ValueError("windows must contain at least one return window")
    if any(window <= 0 for window in windows):
        raise ValueError("return windows must be positive integers")


def _safe_log(value: float) -> float:
    if pd.isna(value):
        return nan
    return log(value)

"""Global macro and benchmark regime feature calculations."""

from __future__ import annotations

from math import log, nan, sqrt

import pandas as pd

from portfolio_rl.config.schemas import FeaturesConfig


REQUIRED_MACRO_COLUMNS = ("date", "series_id", "value")
REQUIRED_PRICE_COLUMNS = ("date", "ticker", "adj_close")
TRADING_DAYS_PER_YEAR = 252
VIX_SERIES_ID = "VIXCLS"
DGS2_SERIES_ID = "DGS2"
DGS10_SERIES_ID = "DGS10"
YIELD_CURVE_SERIES_ID = "T10Y2Y"
CREDIT_SPREAD_SERIES_IDS = ("BAMLH0A0HYM2", "BAMLC0A0CM")
RATE_CHANGE_WINDOW = 5


def calculate_global_features(
    macro: pd.DataFrame,
    prices: pd.DataFrame,
    feature_config: FeaturesConfig,
    benchmark_ticker: str,
) -> pd.DataFrame:
    """Calculate date-level macro and benchmark regime features."""
    macro_wide = _prepare_macro(macro)
    features = pd.DataFrame(index=macro_wide.index)
    features["date"] = features.index

    _add_vix_features(features, macro_wide, feature_config.volatility_windows)
    _add_rate_features(features, macro_wide)
    _add_credit_spread_features(features, macro_wide)
    _add_benchmark_regime_features(
        features,
        prices,
        feature_config,
        benchmark_ticker.strip().upper(),
    )

    return features


def _prepare_macro(macro: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in REQUIRED_MACRO_COLUMNS if column not in macro.columns]
    if missing:
        raise ValueError(f"macro is missing required columns: {missing}")

    macro_frame = macro.loc[:, REQUIRED_MACRO_COLUMNS].copy()
    macro_frame["date"] = pd.to_datetime(macro_frame["date"])
    macro_frame["series_id"] = macro_frame["series_id"].str.upper()
    macro_frame["value"] = pd.to_numeric(macro_frame["value"])
    macro_wide = macro_frame.pivot(
        index="date",
        columns="series_id",
        values="value",
    ).sort_index()
    return macro_wide.ffill()


def _rolling_z_score(values: pd.Series, window: int) -> pd.Series:
    rolling = values.rolling(window, min_periods=window)
    return (values - rolling.mean()) / rolling.std()


def _add_vix_features(
    features: pd.DataFrame,
    macro_wide: pd.DataFrame,
    volatility_windows: list[int],
) -> None:
    if VIX_SERIES_ID not in macro_wide.columns:
        raise ValueError(f"macro is missing required series: {VIX_SERIES_ID}")
    for window in volatility_windows:
        features[f"vix_z_{window}d"] = _rolling_z_score(macro_wide[VIX_SERIES_ID], window)


def _add_rate_features(features: pd.DataFrame, macro_wide: pd.DataFrame) -> None:
    for series_id, column_name in (
        (DGS2_SERIES_ID, "dgs2_change_5d"),
        (DGS10_SERIES_ID, "dgs10_change_5d"),
    ):
        if series_id not in macro_wide.columns:
            raise ValueError(f"macro is missing required series: {series_id}")
        features[column_name] = macro_wide[series_id] - macro_wide[series_id].shift(
            RATE_CHANGE_WINDOW
        )

    if YIELD_CURVE_SERIES_ID in macro_wide.columns:
        features["yield_curve_10y_2y"] = macro_wide[YIELD_CURVE_SERIES_ID]
    else:
        features["yield_curve_10y_2y"] = (
            macro_wide[DGS10_SERIES_ID] - macro_wide[DGS2_SERIES_ID]
        )


def _add_credit_spread_features(
    features: pd.DataFrame,
    macro_wide: pd.DataFrame,
) -> None:
    spread_series_id = next(
        (
            series_id
            for series_id in CREDIT_SPREAD_SERIES_IDS
            if series_id in macro_wide.columns
        ),
        None,
    )
    if spread_series_id is None:
        raise ValueError("macro is missing a configured credit spread series")
    features["credit_spread_z_63d"] = _rolling_z_score(macro_wide[spread_series_id], 63)


def _add_benchmark_regime_features(
    features: pd.DataFrame,
    prices: pd.DataFrame,
    feature_config: FeaturesConfig,
    benchmark_ticker: str,
) -> None:
    if not benchmark_ticker:
        raise ValueError("benchmark_ticker must not be empty")
    missing = [column for column in REQUIRED_PRICE_COLUMNS if column not in prices.columns]
    if missing:
        raise ValueError(f"prices is missing required columns: {missing}")

    price_frame = prices.loc[:, REQUIRED_PRICE_COLUMNS].copy()
    price_frame["date"] = pd.to_datetime(price_frame["date"])
    price_frame["ticker"] = price_frame["ticker"].str.upper()
    price_frame["adj_close"] = pd.to_numeric(price_frame["adj_close"])
    benchmark_prices = (
        price_frame.loc[price_frame["ticker"] == benchmark_ticker]
        .sort_values("date")
        .set_index("date")["adj_close"]
    )
    if benchmark_prices.empty:
        raise ValueError(f"benchmark_ticker is missing from prices: {benchmark_ticker}")

    benchmark_returns = (benchmark_prices / benchmark_prices.shift(1)).map(
        lambda value: log(value) if pd.notna(value) else nan
    )
    volatility_window = min(feature_config.volatility_windows)
    drawdown_window = max(feature_config.drawdown_windows)
    benchmark_name = benchmark_ticker.lower()
    features[f"{benchmark_name}_vol_{volatility_window}d"] = benchmark_returns.rolling(
        volatility_window,
        min_periods=volatility_window,
    ).std() * sqrt(TRADING_DAYS_PER_YEAR)
    rolling_high = benchmark_prices.rolling(
        drawdown_window,
        min_periods=drawdown_window,
    ).max()
    features[f"{benchmark_name}_drawdown_{drawdown_window}d"] = (
        benchmark_prices / rolling_high - 1.0
    )

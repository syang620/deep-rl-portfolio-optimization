"""Technical feature calculations for long-format daily prices."""

from __future__ import annotations

from collections.abc import Sequence
from math import log, nan, sqrt

import pandas as pd

from portfolio_rl.config.schemas import FeaturesConfig


REQUIRED_PRICE_COLUMNS = ("date", "ticker", "adj_close", "close", "volume")
TRADING_DAYS_PER_YEAR = 252
MACD_FAST_SPAN = 12
MACD_SLOW_SPAN = 26
MACD_SIGNAL_SPAN = 9
VOLUME_Z_WINDOWS = (21,)
DOWNSIDE_VOLATILITY_WINDOWS = (63,)


def calculate_technical_features(
    prices: pd.DataFrame,
    feature_config: FeaturesConfig,
    benchmark_ticker: str,
) -> pd.DataFrame:
    """Calculate per-asset technical features without cross-ticker leakage."""
    _validate_prices(prices)
    benchmark = benchmark_ticker.strip().upper()
    if not benchmark:
        raise ValueError("benchmark_ticker must not be empty")

    features = _prepare_prices(prices)
    features["ret_1d_internal"] = _grouped_log_return(features, 1)

    _add_volatility_features(features, feature_config.volatility_windows)
    _add_downside_volatility_features(features, feature_config.volatility_windows)
    _add_drawdown_features(features, feature_config.drawdown_windows)
    _add_rsi_features(features, feature_config.rsi_windows)
    _add_macd_features(features)
    _add_price_z_features(features, feature_config.price_z_windows)
    _add_volume_z_features(features, feature_config.volatility_windows)
    _add_benchmark_features(features, feature_config.correlation_windows, benchmark)

    return features.drop(
        columns=["adj_close", "close", "volume", "ret_1d_internal"],
    )


def _prepare_prices(prices: pd.DataFrame) -> pd.DataFrame:
    features = prices.loc[:, REQUIRED_PRICE_COLUMNS].copy()
    features["date"] = pd.to_datetime(features["date"])
    for column in ("adj_close", "close", "volume"):
        features[column] = pd.to_numeric(features[column])
    return features.sort_values(["ticker", "date"], ignore_index=True)


def _validate_prices(prices: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_PRICE_COLUMNS if column not in prices.columns]
    if missing:
        raise ValueError(f"prices is missing required columns: {missing}")


def _grouped_log_return(features: pd.DataFrame, window: int) -> pd.Series:
    grouped_prices = features.groupby("ticker", sort=False)["adj_close"]
    ratios = grouped_prices.transform(lambda values: values / values.shift(window))
    return ratios.map(lambda value: log(value) if pd.notna(value) else nan)


def _rolling_z_score(values: pd.Series, window: int) -> pd.Series:
    rolling = values.rolling(window, min_periods=window)
    return (values - rolling.mean()) / rolling.std()


def _add_volatility_features(features: pd.DataFrame, windows: Sequence[int]) -> None:
    grouped_returns = features.groupby("ticker", sort=False)["ret_1d_internal"]
    for window in windows:
        features[f"vol_{window}d"] = grouped_returns.transform(
            lambda values: values.rolling(window, min_periods=window).std()
            * sqrt(TRADING_DAYS_PER_YEAR)
        )


def _add_downside_volatility_features(
    features: pd.DataFrame,
    volatility_windows: Sequence[int],
) -> None:
    windows = [
        window
        for window in DOWNSIDE_VOLATILITY_WINDOWS
        if window in set(volatility_windows)
    ]
    for window in windows:
        column = f"downside_vol_{window}d"
        features[column] = features.groupby("ticker", sort=False)[
            "ret_1d_internal"
        ].transform(
            lambda values: values.where(values < 0, 0.0)
            .rolling(window, min_periods=window)
            .std()
            * sqrt(TRADING_DAYS_PER_YEAR)
        )


def _add_drawdown_features(features: pd.DataFrame, windows: Sequence[int]) -> None:
    grouped_prices = features.groupby("ticker", sort=False)["adj_close"]
    for window in windows:
        rolling_high = grouped_prices.transform(
            lambda values: values.rolling(window, min_periods=window).max()
        )
        features[f"drawdown_{window}d"] = features["adj_close"] / rolling_high - 1.0


def _add_rsi_features(features: pd.DataFrame, windows: Sequence[int]) -> None:
    delta = features.groupby("ticker", sort=False)["adj_close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    tickers = features["ticker"]

    for window in windows:
        average_gain = gain.groupby(tickers, sort=False).transform(
            lambda values: values.rolling(window, min_periods=window).mean()
        )
        average_loss = loss.groupby(tickers, sort=False).transform(
            lambda values: values.rolling(window, min_periods=window).mean()
        )
        relative_strength = average_gain / average_loss
        rsi = 100.0 - (100.0 / (1.0 + relative_strength))
        features[f"rsi_{window}"] = rsi.mask(average_loss == 0.0, 100.0)


def _add_macd_features(features: pd.DataFrame) -> None:
    grouped_prices = features.groupby("ticker", sort=False)["adj_close"]
    fast_ema = grouped_prices.transform(
        lambda values: values.ewm(
            span=MACD_FAST_SPAN,
            adjust=False,
            min_periods=MACD_FAST_SPAN,
        ).mean()
    )
    slow_ema = grouped_prices.transform(
        lambda values: values.ewm(
            span=MACD_SLOW_SPAN,
            adjust=False,
            min_periods=MACD_SLOW_SPAN,
        ).mean()
    )
    features["macd_12_26"] = fast_ema - slow_ema
    features["macd_signal_9"] = features.groupby("ticker", sort=False)[
        "macd_12_26"
    ].transform(
        lambda values: values.ewm(
            span=MACD_SIGNAL_SPAN,
            adjust=False,
            min_periods=MACD_SIGNAL_SPAN,
        ).mean()
    )


def _add_price_z_features(features: pd.DataFrame, windows: Sequence[int]) -> None:
    grouped_prices = features.groupby("ticker", sort=False)["adj_close"]
    for window in windows:
        features[f"price_z_{window}d"] = grouped_prices.transform(
            lambda values: _rolling_z_score(values, window)
        )


def _add_volume_z_features(
    features: pd.DataFrame,
    volatility_windows: Sequence[int],
) -> None:
    windows = [window for window in VOLUME_Z_WINDOWS if window in set(volatility_windows)]
    dollar_volume = features["close"] * features["volume"]
    log_dollar_volume = dollar_volume.map(lambda value: log(1.0 + value))
    tickers = features["ticker"]

    for window in windows:
        features[f"volume_z_{window}d"] = log_dollar_volume.groupby(
            tickers,
            sort=False,
        ).transform(lambda values: _rolling_z_score(values, window))


def _add_benchmark_features(
    features: pd.DataFrame,
    windows: Sequence[int],
    benchmark_ticker: str,
) -> None:
    returns = features.pivot(
        index="date",
        columns="ticker",
        values="ret_1d_internal",
    )
    if benchmark_ticker not in returns.columns:
        raise ValueError(f"benchmark_ticker is missing from prices: {benchmark_ticker}")

    benchmark_returns = returns[benchmark_ticker]
    frames = []
    for ticker in returns.columns:
        ticker_returns = returns[ticker]
        ticker_frame = pd.DataFrame(index=returns.index)
        ticker_frame["date"] = ticker_frame.index
        ticker_frame["ticker"] = ticker
        for window in windows:
            ticker_frame[f"corr_to_{benchmark_ticker.lower()}_{window}d"] = (
                ticker_returns.rolling(
                    window,
                    min_periods=window,
                ).corr(benchmark_returns)
            )
            benchmark_variance = benchmark_returns.rolling(
                window,
                min_periods=window,
            ).var()
            ticker_frame[f"beta_to_{benchmark_ticker.lower()}_{window}d"] = (
                ticker_returns.rolling(window, min_periods=window).cov(
                    benchmark_returns
                )
                / benchmark_variance
            )
        frames.append(ticker_frame)

    benchmark_features = pd.concat(frames, ignore_index=True)
    features_with_benchmark = features.merge(
        benchmark_features,
        on=["date", "ticker"],
        how="left",
        validate="one_to_one",
    )
    for column in benchmark_features.columns:
        if column not in ("date", "ticker"):
            features[column] = features_with_benchmark[column]

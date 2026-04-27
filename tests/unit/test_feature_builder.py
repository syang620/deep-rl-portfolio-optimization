from __future__ import annotations

from datetime import datetime, timezone
from math import isclose, log, sin

import pandas as pd
import pytest

from portfolio_rl.config.schemas import FeaturesConfig
from portfolio_rl.features.builder import build_asset_features, build_features
from portfolio_rl.features.returns import calculate_return_features


ASSET_ORDER = ["SPY", "QQQ"]


def test_calculate_return_features_groups_by_ticker_before_shifting() -> None:
    prices = pd.DataFrame(
        {
            "date": [
                "2024-01-02",
                "2024-01-03",
                "2024-01-02",
                "2024-01-03",
            ],
            "ticker": ["SPY", "SPY", "QQQ", "QQQ"],
            "adj_close": [100.0, 110.0, 200.0, 220.0],
        }
    )

    features = calculate_return_features(prices, windows=[1])

    spy_return = features.loc[
        (features["ticker"] == "SPY") & (features["date"] == "2024-01-03"),
        "ret_1d",
    ].iloc[0]
    qqq_return = features.loc[
        (features["ticker"] == "QQQ") & (features["date"] == "2024-01-03"),
        "ret_1d",
    ].iloc[0]
    assert isclose(spy_return, log(1.1))
    assert isclose(qqq_return, log(1.1))


def test_build_asset_features_drops_warmup_rows_and_excludes_raw_prices() -> None:
    prices = _prices_fixture()

    features = build_asset_features(
        prices,
        _feature_config(),
        benchmark_ticker="SPY",
        asset_order=ASSET_ORDER,
    )

    assert not features.empty
    assert features["date"].min() > pd.Timestamp(prices["date"].min())
    assert {"close", "adj_close", "open", "high", "low", "volume"}.isdisjoint(
        features.columns
    )
    assert {
        "ret_1d",
        "ret_5d",
        "ret_21d",
        "ret_63d",
        "ret_126d",
        "ret_252d",
        "vol_21d",
        "vol_63d",
        "downside_vol_63d",
        "drawdown_63d",
        "rsi_14",
        "macd_12_26",
        "macd_signal_9",
        "price_z_20d",
        "price_z_50d",
        "volume_z_21d",
        "rank_ret_21d",
        "rank_ret_63d",
        "rank_vol_21d",
        "corr_to_spy_63d",
        "beta_to_spy_63d",
    }.issubset(features.columns)
    assert not features.drop(columns=["date", "ticker", "feature_version"]).isna().any().any()


def test_build_features_outputs_clean_asset_and_global_frames() -> None:
    result = build_features(
        prices=_prices_fixture(),
        macro=_macro_fixture(),
        feature_config=_feature_config(),
        benchmark_ticker="SPY",
        asset_order=ASSET_ORDER,
    )

    assert not result.asset_features.empty
    assert not result.global_features.empty
    assert set(result.asset_features["feature_version"]) == {"v1"}
    assert set(result.global_features["feature_version"]) == {"v1"}
    assert set(result.asset_features["date"]) == set(result.global_features["date"])
    assert (
        result.asset_features.groupby("date")["ticker"]
        .apply(list)
        .map(lambda tickers: tickers == ASSET_ORDER)
        .all()
    )
    assert not _forbidden_feature_names(result.asset_features.columns)
    assert not _forbidden_feature_names(result.global_features.columns)
    assert {
        "vix_z_21d",
        "vix_z_63d",
        "dgs2_change_5d",
        "dgs10_change_5d",
        "yield_curve_10y_2y",
        "credit_spread_z_63d",
        "spy_vol_21d",
        "spy_drawdown_63d",
    }.issubset(result.global_features.columns)


def test_build_asset_features_fails_when_benchmark_is_missing() -> None:
    prices = _prices_fixture()

    with pytest.raises(ValueError, match="benchmark_ticker is missing"):
        build_asset_features(
            prices,
            _feature_config(),
            benchmark_ticker="IWM",
            asset_order=ASSET_ORDER,
        )


def test_build_asset_features_fails_when_configured_ticker_is_missing() -> None:
    prices = _prices_fixture().loc[lambda frame: frame["ticker"] != "QQQ"]

    with pytest.raises(ValueError, match="missing configured tickers"):
        build_asset_features(
            prices,
            _feature_config(),
            benchmark_ticker="SPY",
            asset_order=ASSET_ORDER,
        )


def test_build_asset_features_fails_on_incomplete_post_warmup_coverage() -> None:
    prices = _prices_fixture()
    last_qqq_index = prices.loc[prices["ticker"] == "QQQ"].index.max()
    prices = prices.drop(index=last_qqq_index)

    with pytest.raises(ValueError, match="complete ticker coverage"):
        build_asset_features(
            prices,
            _feature_config(),
            benchmark_ticker="SPY",
            asset_order=ASSET_ORDER,
        )


def test_build_features_aligns_asset_and_global_common_dates() -> None:
    result = build_features(
        prices=_prices_fixture(),
        macro=_macro_fixture(periods=330),
        feature_config=_feature_config(),
        benchmark_ticker="SPY",
        asset_order=ASSET_ORDER,
    )

    assert set(result.asset_features["date"]) == set(result.global_features["date"])


def _feature_config() -> FeaturesConfig:
    return FeaturesConfig.model_validate(
        {
            "feature_version": "v1",
            "return_windows": [1, 5, 21, 63, 126, 252],
            "volatility_windows": [21, 63],
            "drawdown_windows": [63],
            "rsi_windows": [14],
            "price_z_windows": [20, 50],
            "correlation_windows": [63],
            "winsorization": {
                "enabled": True,
                "lower_quantile": 0.005,
                "upper_quantile": 0.995,
            },
            "normalization": {
                "method": "standard_scaler",
                "fit_split": "train",
            },
        }
    )


def _prices_fixture() -> pd.DataFrame:
    downloaded_at = datetime(2026, 4, 26, tzinfo=timezone.utc)
    dates = pd.bdate_range("2023-01-02", periods=340)
    frames = []
    for ticker, base, phase in (("SPY", 100.0, 0.0), ("QQQ", 150.0, 1.7)):
        rows = []
        for index, date in enumerate(dates):
            adj_close = base + index * 0.35 + sin(index / 3.0 + phase) * 2.0
            close = adj_close * 1.001
            rows.append(
                {
                    "date": date.date(),
                    "ticker": ticker,
                    "open": close * 0.99,
                    "high": close * 1.01,
                    "low": close * 0.98,
                    "close": close,
                    "adj_close": adj_close,
                    "volume": 1_000_000 + index * 1_000,
                    "dividends": 0.0,
                    "stock_splits": 0.0,
                    "source": "test",
                    "downloaded_at": downloaded_at,
                }
            )
        frames.append(pd.DataFrame(rows))
    return pd.concat(frames, ignore_index=True)


def _macro_fixture(periods: int = 340) -> pd.DataFrame:
    downloaded_at = datetime(2026, 4, 26, tzinfo=timezone.utc)
    dates = pd.bdate_range("2023-01-02", periods=periods)
    series_offsets = {
        "VIXCLS": 20.0,
        "DGS2": 4.0,
        "DGS10": 4.5,
        "T10Y2Y": 0.5,
        "BAMLH0A0HYM2": 3.5,
    }
    rows = []
    for series_id, base in series_offsets.items():
        for index, date in enumerate(dates):
            rows.append(
                {
                    "date": date.date(),
                    "series_id": series_id,
                    "value": base + sin(index / 7.0) * 0.2 + index * 0.001,
                    "source": "test",
                    "downloaded_at": downloaded_at,
                }
            )
    return pd.DataFrame(rows)


def _forbidden_feature_names(columns: pd.Index) -> list[str]:
    forbidden = ("future", "next", "target", "label")
    return [
        column
        for column in columns
        if any(forbidden_name in column.lower() for forbidden_name in forbidden)
    ]

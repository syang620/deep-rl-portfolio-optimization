"""Build Phase 1 asset and global feature frames."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd

from portfolio_rl.config.schemas import FeaturesConfig
from portfolio_rl.features.macro import calculate_global_features
from portfolio_rl.features.returns import calculate_return_features
from portfolio_rl.features.technicals import calculate_technical_features


IDENTIFIER_COLUMNS = ("date", "ticker", "feature_version")
GLOBAL_IDENTIFIER_COLUMNS = ("date", "feature_version")


@dataclass(frozen=True)
class FeatureFrames:
    """Container for the feature outputs consumed by later pipeline stages."""

    asset_features: pd.DataFrame
    global_features: pd.DataFrame


def build_features(
    prices: pd.DataFrame,
    macro: pd.DataFrame,
    feature_config: FeaturesConfig,
    benchmark_ticker: str,
    asset_order: Sequence[str],
) -> FeatureFrames:
    """Build per-asset and date-level feature frames for Phase 1."""
    normalized_asset_order = _normalize_asset_order(asset_order)
    asset_features = build_asset_features(
        prices,
        feature_config,
        benchmark_ticker,
        normalized_asset_order,
    )
    global_features = build_global_features(
        macro,
        prices,
        feature_config,
        benchmark_ticker,
    )
    asset_features, global_features = _align_common_feature_dates(
        asset_features,
        global_features,
    )
    _assert_complete_asset_coverage(asset_features, normalized_asset_order)
    return FeatureFrames(
        asset_features=asset_features,
        global_features=global_features,
    )


def build_asset_features(
    prices: pd.DataFrame,
    feature_config: FeaturesConfig,
    benchmark_ticker: str,
    asset_order: Sequence[str],
) -> pd.DataFrame:
    """Build leakage-safe per-asset feature rows and drop warm-up rows."""
    normalized_asset_order = _normalize_asset_order(asset_order)
    _assert_prices_include_assets(prices, normalized_asset_order)
    returns = calculate_return_features(prices, feature_config.return_windows)
    technicals = calculate_technical_features(prices, feature_config, benchmark_ticker)
    features = returns.merge(
        technicals,
        on=["date", "ticker"],
        how="inner",
        validate="one_to_one",
    )
    _add_rank_features(features, feature_config)
    features.insert(2, "feature_version", feature_config.feature_version)
    cleaned = _drop_invalid_feature_rows(features, IDENTIFIER_COLUMNS)
    cleaned = _drop_initial_incomplete_coverage_dates(cleaned, normalized_asset_order)
    _assert_complete_asset_coverage(cleaned, normalized_asset_order)
    return _sort_asset_features(cleaned, normalized_asset_order).reset_index(drop=True)


def build_global_features(
    macro: pd.DataFrame,
    prices: pd.DataFrame,
    feature_config: FeaturesConfig,
    benchmark_ticker: str,
) -> pd.DataFrame:
    """Build leakage-safe global feature rows and drop warm-up rows."""
    features = calculate_global_features(macro, prices, feature_config, benchmark_ticker)
    features.insert(1, "feature_version", feature_config.feature_version)
    return _drop_invalid_feature_rows(features, GLOBAL_IDENTIFIER_COLUMNS).sort_values(
        "date",
        ignore_index=True,
    )


def _drop_invalid_feature_rows(
    features: pd.DataFrame,
    identifier_columns: tuple[str, ...],
) -> pd.DataFrame:
    feature_columns = [
        column for column in features.columns if column not in set(identifier_columns)
    ]
    cleaned = features.replace([float("inf"), -float("inf")], pd.NA)
    return cleaned.dropna(subset=feature_columns).reset_index(drop=True)


def _add_rank_features(features: pd.DataFrame, feature_config: FeaturesConfig) -> None:
    for window in (21, 63):
        column = f"ret_{window}d"
        rank_column = f"rank_ret_{window}d"
        if (
            window in feature_config.return_windows
            and column in features.columns
            and rank_column not in features.columns
        ):
            features[rank_column] = features.groupby("date", sort=False)[column].rank(
                pct=True
            )

    for window in (21,):
        column = f"vol_{window}d"
        rank_column = f"rank_vol_{window}d"
        if (
            window in feature_config.volatility_windows
            and column in features.columns
            and rank_column not in features.columns
        ):
            features[rank_column] = features.groupby("date", sort=False)[column].rank(
                pct=True
            )


def _normalize_asset_order(asset_order: Sequence[str]) -> list[str]:
    normalized = [ticker.strip().upper() for ticker in asset_order]
    if not normalized or any(not ticker for ticker in normalized):
        raise ValueError("asset_order must contain at least one non-empty ticker")
    if len(normalized) != len(set(normalized)):
        raise ValueError("asset_order must not contain duplicate tickers")
    return normalized


def _assert_prices_include_assets(
    prices: pd.DataFrame,
    asset_order: Sequence[str],
) -> None:
    if "ticker" not in prices.columns:
        raise ValueError("prices is missing required column: ticker")

    present = {ticker.strip().upper() for ticker in prices["ticker"].dropna().unique()}
    missing = [ticker for ticker in asset_order if ticker not in present]
    if missing:
        raise ValueError(f"prices are missing configured tickers: {missing}")


def _assert_complete_asset_coverage(
    asset_features: pd.DataFrame,
    asset_order: Sequence[str],
) -> None:
    expected = list(asset_order)
    expected_set = set(expected)
    if asset_features.empty:
        raise ValueError("asset features are empty after warm-up rows are dropped")

    counts = asset_features.groupby(["date", "ticker"], sort=False).size()
    if counts.gt(1).any():
        raise ValueError("asset features contain duplicate date-ticker rows")

    for date_value, date_frame in asset_features.groupby("date", sort=False):
        present = set(date_frame["ticker"])
        if present != expected_set:
            missing = [ticker for ticker in expected if ticker not in present]
            extra = sorted(present - expected_set)
            raise ValueError(
                "asset features are missing complete ticker coverage "
                f"for {date_value}: missing={missing}, extra={extra}"
            )


def _drop_initial_incomplete_coverage_dates(
    asset_features: pd.DataFrame,
    asset_order: Sequence[str],
) -> pd.DataFrame:
    expected_set = set(asset_order)
    complete_by_date = asset_features.groupby("date", sort=True)["ticker"].apply(
        lambda tickers: set(tickers) == expected_set
    )
    complete_dates = complete_by_date.loc[complete_by_date].index
    if complete_dates.empty:
        raise ValueError("asset features never reach complete ticker coverage")

    first_complete_date = complete_dates.min()
    return asset_features.loc[
        asset_features["date"] >= first_complete_date
    ].reset_index(drop=True)


def _align_common_feature_dates(
    asset_features: pd.DataFrame,
    global_features: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    common_dates = sorted(set(asset_features["date"]).intersection(global_features["date"]))
    if not common_dates:
        raise ValueError("asset and global features have no common dates")

    aligned_assets = asset_features.loc[
        asset_features["date"].isin(common_dates)
    ].copy()
    aligned_global = global_features.loc[
        global_features["date"].isin(common_dates)
    ].copy()
    return (
        aligned_assets.reset_index(drop=True),
        aligned_global.sort_values("date", ignore_index=True),
    )


def _sort_asset_features(
    asset_features: pd.DataFrame,
    asset_order: Sequence[str],
) -> pd.DataFrame:
    sorted_features = asset_features.copy()
    sorted_features["ticker"] = pd.Categorical(
        sorted_features["ticker"],
        categories=list(asset_order),
        ordered=True,
    )
    sorted_features = sorted_features.sort_values(["date", "ticker"])
    sorted_features["ticker"] = sorted_features["ticker"].astype(str)
    return sorted_features

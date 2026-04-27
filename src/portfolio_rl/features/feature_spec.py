"""Feature ordering contract for training and serving."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path

import pandas as pd

from portfolio_rl.config.schemas import FeaturesConfig, UniverseConfig


DEFAULT_FEATURE_SPEC_PATH = Path("artifacts/feature_specs/feature_spec_v1.json")


@dataclass(frozen=True)
class FeatureSpec:
    """Versioned feature ordering contract shared by training and serving."""

    feature_version: str
    asset_order: list[str]
    per_asset_features: list[str]
    global_features: list[str]
    current_weight_features: list[str]
    observation_dim: int
    created_at: str


def build_feature_spec(
    universe_config: UniverseConfig,
    feature_config: FeaturesConfig,
    benchmark_ticker: str = "SPY",
) -> FeatureSpec:
    """Build the v1 feature specification from typed configs."""
    asset_order = universe_config.tickers
    benchmark = _normalize_benchmark(benchmark_ticker)
    per_asset_features = _per_asset_features(feature_config, benchmark)
    global_features = _global_features(feature_config, benchmark)
    current_weight_features = [
        f"weight_{ticker.lower()}" for ticker in asset_order
    ]
    observation_dim = (
        len(asset_order) * len(per_asset_features)
        + len(global_features)
        + len(current_weight_features)
    )
    return FeatureSpec(
        feature_version=feature_config.feature_version,
        asset_order=asset_order,
        per_asset_features=per_asset_features,
        global_features=global_features,
        current_weight_features=current_weight_features,
        observation_dim=observation_dim,
        created_at=datetime.now(timezone.utc).isoformat(),
    )


def save_feature_spec(
    feature_spec: FeatureSpec,
    path: str | Path = DEFAULT_FEATURE_SPEC_PATH,
) -> None:
    """Save a feature spec as stable JSON."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as spec_file:
        json.dump(asdict(feature_spec), spec_file, indent=2)
        spec_file.write("\n")


def load_feature_spec(path: str | Path = DEFAULT_FEATURE_SPEC_PATH) -> FeatureSpec:
    """Load a feature spec from JSON."""
    with Path(path).open("r", encoding="utf-8") as spec_file:
        data = json.load(spec_file)
    return FeatureSpec(**data)


def flatten_features(
    asset_features: pd.DataFrame,
    global_features: pd.DataFrame | pd.Series,
    current_weights: Mapping[str, float],
    feature_spec: FeatureSpec,
) -> list[float]:
    """Flatten one observation using the exact order in feature_spec."""
    asset_frame = _one_asset_row_per_configured_ticker(
        asset_features,
        feature_spec.asset_order,
    )
    global_row = _single_global_feature_row(global_features)
    normalized_weights = {
        ticker.strip().upper(): float(weight)
        for ticker, weight in current_weights.items()
    }

    values: list[float] = []
    for ticker in feature_spec.asset_order:
        row = asset_frame.loc[ticker]
        values.extend(_ordered_values(row, feature_spec.per_asset_features))

    values.extend(_ordered_values(global_row, feature_spec.global_features))

    for ticker in feature_spec.asset_order:
        if ticker not in normalized_weights:
            raise ValueError(f"current_weights is missing ticker: {ticker}")
        values.append(normalized_weights[ticker])

    if len(values) != feature_spec.observation_dim:
        raise ValueError("flattened observation length does not match observation_dim")
    return values


def _per_asset_features(
    feature_config: FeaturesConfig,
    benchmark_ticker: str,
) -> list[str]:
    features = [f"ret_{window}d" for window in feature_config.return_windows]
    features.extend(f"vol_{window}d" for window in feature_config.volatility_windows)
    if 63 in feature_config.volatility_windows:
        features.append("downside_vol_63d")
    features.extend(f"drawdown_{window}d" for window in feature_config.drawdown_windows)
    features.extend(f"rsi_{window}" for window in feature_config.rsi_windows)
    features.extend(["macd_12_26", "macd_signal_9"])
    features.extend(f"price_z_{window}d" for window in feature_config.price_z_windows)
    if 21 in feature_config.volatility_windows:
        features.append("volume_z_21d")
        features.append("rank_vol_21d")
    if 21 in feature_config.return_windows:
        features.append("rank_ret_21d")
    if 63 in feature_config.return_windows:
        features.append("rank_ret_63d")
    features.extend(
        f"corr_to_{benchmark_ticker.lower()}_{window}d"
        for window in feature_config.correlation_windows
    )
    features.extend(
        f"beta_to_{benchmark_ticker.lower()}_{window}d"
        for window in feature_config.correlation_windows
    )
    return features


def _global_features(
    feature_config: FeaturesConfig,
    benchmark_ticker: str,
) -> list[str]:
    benchmark = benchmark_ticker.lower()
    features = [f"vix_z_{window}d" for window in feature_config.volatility_windows]
    features.extend(
        [
            "dgs2_change_5d",
            "dgs10_change_5d",
            "yield_curve_10y_2y",
            "credit_spread_z_63d",
        ]
    )
    volatility_window = min(feature_config.volatility_windows)
    drawdown_window = max(feature_config.drawdown_windows)
    features.extend(
        [
            f"{benchmark}_vol_{volatility_window}d",
            f"{benchmark}_drawdown_{drawdown_window}d",
        ]
    )
    return features


def _normalize_benchmark(benchmark_ticker: str) -> str:
    benchmark = benchmark_ticker.strip().upper()
    if not benchmark:
        raise ValueError("benchmark_ticker must not be empty")
    return benchmark


def _one_asset_row_per_configured_ticker(
    asset_features: pd.DataFrame,
    asset_order: list[str],
) -> pd.DataFrame:
    if "ticker" not in asset_features.columns:
        raise ValueError("asset_features is missing ticker column")
    if asset_features["ticker"].duplicated().any():
        raise ValueError("asset_features must contain one row per ticker")

    indexed = asset_features.copy()
    indexed["ticker"] = indexed["ticker"].str.upper()
    if indexed["ticker"].duplicated().any():
        raise ValueError("asset_features must contain one row per ticker")
    indexed = indexed.set_index("ticker")
    missing = [ticker for ticker in asset_order if ticker not in indexed.index]
    if missing:
        raise ValueError(f"asset_features is missing tickers: {missing}")
    return indexed.loc[asset_order]


def _single_global_feature_row(
    global_features: pd.DataFrame | pd.Series,
) -> pd.Series:
    if isinstance(global_features, pd.Series):
        return global_features
    if len(global_features) != 1:
        raise ValueError("global_features must contain exactly one row")
    return global_features.iloc[0]


def _ordered_values(row: pd.Series, columns: list[str]) -> list[float]:
    missing = [column for column in columns if column not in row.index]
    if missing:
        raise ValueError(f"feature row is missing columns: {missing}")
    return [float(row[column]) for column in columns]

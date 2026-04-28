"""Feature artifact orchestration for Phase 1."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from portfolio_rl.config.schemas import DataConfig, FeaturesConfig, UniverseConfig
from portfolio_rl.data.storage import (
    read_parquet,
    write_duckdb_table,
    write_parquet,
)
from portfolio_rl.features.builder import build_features


@dataclass(frozen=True)
class FeatureArtifactResult:
    """Output locations and row counts from the unnormalized feature build."""

    features_parquet_path: Path
    global_features_parquet_path: Path
    duckdb_path: Path
    features_row_count: int
    global_features_row_count: int


def build_feature_artifacts(
    data_config: DataConfig,
    feature_config: FeaturesConfig,
    universe_config: UniverseConfig,
    benchmark_ticker: str = "SPY",
) -> FeatureArtifactResult:
    """Build and persist unnormalized Phase 1 feature artifacts."""
    prices = read_parquet(data_config.storage.raw_parquet_dir / "prices_daily.parquet")
    macro = read_parquet(data_config.storage.raw_parquet_dir / "macro_daily.parquet")
    feature_frames = build_features(
        prices=prices,
        macro=macro,
        feature_config=feature_config,
        benchmark_ticker=benchmark_ticker,
        asset_order=universe_config.tickers,
    )

    features_path = data_config.storage.processed_parquet_dir / "features_daily.parquet"
    global_features_path = (
        data_config.storage.processed_parquet_dir / "global_features_daily.parquet"
    )
    write_parquet(feature_frames.asset_features, features_path)
    write_parquet(feature_frames.global_features, global_features_path)

    duckdb_path = data_config.storage.duckdb_path
    write_duckdb_table(feature_frames.asset_features, duckdb_path, "features_daily")
    write_duckdb_table(
        feature_frames.global_features,
        duckdb_path,
        "global_features_daily",
    )

    return FeatureArtifactResult(
        features_parquet_path=features_path,
        global_features_parquet_path=global_features_path,
        duckdb_path=duckdb_path,
        features_row_count=len(feature_frames.asset_features),
        global_features_row_count=len(feature_frames.global_features),
    )

"""Feature artifact orchestration for Phase 1."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from portfolio_rl.config.schemas import DataConfig, FeaturesConfig, UniverseConfig
from portfolio_rl.data.splits import assign_chronological_splits
from portfolio_rl.data.storage import (
    read_parquet,
    write_duckdb_table,
    write_parquet,
)
from portfolio_rl.features.builder import build_features
from portfolio_rl.features.data_quality_report import (
    DEFAULT_DATA_QUALITY_REPORT_PATH,
    build_data_quality_report,
    save_data_quality_report,
)
from portfolio_rl.features.feature_spec import (
    DEFAULT_FEATURE_SPEC_PATH,
    FeatureSpec,
    build_feature_spec,
    save_feature_spec,
)
from portfolio_rl.features.model_matrix import build_model_matrix
from portfolio_rl.features.normalization import (
    DEFAULT_ARTIFACT_PATH,
    NormalizationArtifactBundle,
    fit_normalization_artifact,
    save_normalization_artifact,
    transform_features,
)


GLOBAL_IDENTIFIER_COLUMNS = ("date", "split", "feature_version")


@dataclass(frozen=True)
class FeatureArtifactResult:
    """Output locations and row counts from the feature artifact build."""

    features_parquet_path: Path
    global_features_parquet_path: Path
    normalized_features_parquet_path: Path
    normalized_global_features_parquet_path: Path
    interim_aligned_panel_parquet_path: Path
    model_matrix_parquet_path: Path
    scaler_artifact_path: Path
    feature_spec_path: Path
    data_quality_report_path: Path
    duckdb_path: Path
    features_row_count: int
    global_features_row_count: int
    normalized_features_row_count: int
    normalized_global_features_row_count: int
    interim_aligned_panel_row_count: int
    model_matrix_row_count: int


def build_feature_artifacts(
    data_config: DataConfig,
    feature_config: FeaturesConfig,
    universe_config: UniverseConfig,
    scaler_artifact_path: str | Path = DEFAULT_ARTIFACT_PATH,
    feature_spec_path: str | Path = DEFAULT_FEATURE_SPEC_PATH,
    data_quality_report_path: str | Path = DEFAULT_DATA_QUALITY_REPORT_PATH,
) -> FeatureArtifactResult:
    """Build and persist raw and normalized Phase 1 feature artifacts."""
    benchmark_ticker = feature_config.market.benchmark_ticker
    prices = read_parquet(data_config.storage.raw_parquet_dir / "prices_daily.parquet")
    macro = read_parquet(data_config.storage.raw_parquet_dir / "macro_daily.parquet")
    feature_frames = build_features(
        prices=prices,
        macro=macro,
        feature_config=feature_config,
        benchmark_ticker=benchmark_ticker,
        asset_order=universe_config.tickers,
    )
    interim_aligned_panel = _build_interim_aligned_panel(
        feature_frames.asset_features,
        feature_frames.global_features,
    )
    asset_features = _prepare_model_period_features(
        feature_frames.asset_features,
        data_config,
    )
    global_features = _prepare_model_period_features(
        feature_frames.global_features,
        data_config,
    )
    (
        normalized_asset_features,
        normalized_global_features,
    ) = _normalize_feature_frames(
        asset_features,
        global_features,
        feature_config,
        scaler_artifact_path,
    )
    feature_spec = build_feature_spec(
        universe_config,
        feature_config,
        benchmark_ticker=benchmark_ticker,
    )
    _validate_feature_spec_columns(
        feature_spec,
        normalized_asset_features,
        normalized_global_features,
    )
    save_feature_spec(feature_spec, feature_spec_path)
    model_matrix = build_model_matrix(
        normalized_asset_features,
        normalized_global_features,
        asset_features,
        feature_spec,
    )
    data_quality_report = build_data_quality_report(
        model_matrix,
        data_config,
        feature_config,
        universe_config,
        feature_spec,
        raw_prices=prices,
        raw_macro=macro,
        processed_artifacts={
            "features_daily": asset_features,
            "global_features_daily": global_features,
            "features_normalized_daily": normalized_asset_features,
            "global_features_normalized_daily": normalized_global_features,
            "model_matrix_daily": model_matrix,
        },
    )
    save_data_quality_report(data_quality_report, data_quality_report_path)

    features_path = data_config.storage.processed_parquet_dir / "features_daily.parquet"
    global_features_path = (
        data_config.storage.processed_parquet_dir / "global_features_daily.parquet"
    )
    normalized_features_path = (
        data_config.storage.processed_parquet_dir / "features_normalized_daily.parquet"
    )
    normalized_global_features_path = (
        data_config.storage.processed_parquet_dir
        / "global_features_normalized_daily.parquet"
    )
    interim_aligned_panel_path = (
        data_config.storage.interim_parquet_dir / "aligned_feature_panel_daily.parquet"
    )
    model_matrix_path = (
        data_config.storage.processed_parquet_dir / "model_matrix_daily.parquet"
    )
    write_parquet(interim_aligned_panel, interim_aligned_panel_path)
    write_parquet(asset_features, features_path)
    write_parquet(global_features, global_features_path)
    write_parquet(normalized_asset_features, normalized_features_path)
    write_parquet(normalized_global_features, normalized_global_features_path)
    write_parquet(model_matrix, model_matrix_path)

    duckdb_path = data_config.storage.duckdb_path
    write_duckdb_table(
        interim_aligned_panel,
        duckdb_path,
        "aligned_feature_panel_daily",
    )
    write_duckdb_table(asset_features, duckdb_path, "features_daily")
    write_duckdb_table(
        global_features,
        duckdb_path,
        "global_features_daily",
    )
    write_duckdb_table(
        normalized_asset_features,
        duckdb_path,
        "features_normalized_daily",
    )
    write_duckdb_table(
        normalized_global_features,
        duckdb_path,
        "global_features_normalized_daily",
    )
    write_duckdb_table(model_matrix, duckdb_path, "model_matrix_daily")

    return FeatureArtifactResult(
        features_parquet_path=features_path,
        global_features_parquet_path=global_features_path,
        normalized_features_parquet_path=normalized_features_path,
        normalized_global_features_parquet_path=normalized_global_features_path,
        interim_aligned_panel_parquet_path=interim_aligned_panel_path,
        model_matrix_parquet_path=model_matrix_path,
        scaler_artifact_path=Path(scaler_artifact_path),
        feature_spec_path=Path(feature_spec_path),
        data_quality_report_path=Path(data_quality_report_path),
        duckdb_path=duckdb_path,
        features_row_count=len(asset_features),
        global_features_row_count=len(global_features),
        normalized_features_row_count=len(normalized_asset_features),
        normalized_global_features_row_count=len(normalized_global_features),
        interim_aligned_panel_row_count=len(interim_aligned_panel),
        model_matrix_row_count=len(model_matrix),
    )


def _build_interim_aligned_panel(
    asset_features: pd.DataFrame,
    global_features: pd.DataFrame,
) -> pd.DataFrame:
    """Join aligned asset and global features before split labels and normalization."""
    global_columns = [
        column for column in global_features.columns if column != "feature_version"
    ]
    return asset_features.merge(
        global_features.loc[:, global_columns],
        on="date",
        how="inner",
        validate="many_to_one",
    ).sort_values(["date", "ticker"], ignore_index=True)


def _prepare_model_period_features(
    features: pd.DataFrame,
    data_config: DataConfig,
) -> pd.DataFrame:
    model_features = features.loc[
        pd.to_datetime(features["date"]).dt.date >= data_config.model_start_date
    ].reset_index(drop=True)
    return assign_chronological_splits(model_features, data_config)


def _normalize_feature_frames(
    asset_features: pd.DataFrame,
    global_features: pd.DataFrame,
    feature_config: FeaturesConfig,
    scaler_artifact_path: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    asset_artifact = fit_normalization_artifact(asset_features, feature_config)
    global_artifact = fit_normalization_artifact(
        global_features,
        feature_config,
        identifier_columns=GLOBAL_IDENTIFIER_COLUMNS,
    )
    normalized_asset_features = transform_features(asset_features, asset_artifact)
    normalized_global_features = transform_features(
        global_features,
        global_artifact,
        identifier_columns=GLOBAL_IDENTIFIER_COLUMNS,
    )
    save_normalization_artifact(
        NormalizationArtifactBundle(
            asset_features=asset_artifact,
            global_features=global_artifact,
        ),
        scaler_artifact_path,
    )
    return normalized_asset_features, normalized_global_features


def _validate_feature_spec_columns(
    feature_spec: FeatureSpec,
    normalized_asset_features: pd.DataFrame,
    normalized_global_features: pd.DataFrame,
) -> None:
    missing_asset_features = [
        column
        for column in feature_spec.per_asset_features
        if column not in normalized_asset_features.columns
    ]
    if missing_asset_features:
        raise ValueError(
            f"normalized asset features are missing spec columns: {missing_asset_features}"
        )

    missing_global_features = [
        column
        for column in feature_spec.global_features
        if column not in normalized_global_features.columns
    ]
    if missing_global_features:
        raise ValueError(
            f"normalized global features are missing spec columns: {missing_global_features}"
        )

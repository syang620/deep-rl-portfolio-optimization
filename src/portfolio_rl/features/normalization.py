"""Train-only winsorization and standard scaling for Phase 1 features."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
import pickle

import pandas as pd

from portfolio_rl.config.schemas import FeaturesConfig


DEFAULT_IDENTIFIER_COLUMNS = ("date", "ticker", "split", "feature_version")
DEFAULT_ARTIFACT_PATH = Path("artifacts/scalers/feature_scaler_v1.pkl")
TRAIN_SPLIT = "train"


@dataclass(frozen=True)
class NormalizationArtifact:
    """Train-fitted thresholds and scaling statistics."""

    feature_version: str
    fit_split: str
    feature_columns: list[str]
    winsorization_lower: dict[str, float]
    winsorization_upper: dict[str, float]
    means: dict[str, float]
    scales: dict[str, float]


def normalize_features(
    features: pd.DataFrame,
    feature_config: FeaturesConfig,
    artifact_path: str | Path = DEFAULT_ARTIFACT_PATH,
    identifier_columns: Sequence[str] = DEFAULT_IDENTIFIER_COLUMNS,
) -> pd.DataFrame:
    """Fit train-only normalization artifacts, transform all rows, and save them."""
    artifact = fit_normalization_artifact(
        features,
        feature_config,
        identifier_columns=identifier_columns,
    )
    normalized = transform_features(
        features,
        artifact,
        identifier_columns=identifier_columns,
    )
    save_normalization_artifact(artifact, artifact_path)
    return normalized


def fit_normalization_artifact(
    features: pd.DataFrame,
    feature_config: FeaturesConfig,
    identifier_columns: Sequence[str] = DEFAULT_IDENTIFIER_COLUMNS,
) -> NormalizationArtifact:
    """Fit winsorization thresholds and standard-scaler stats on train rows only."""
    _validate_features_for_fit(features, identifier_columns)
    feature_columns = _feature_columns(features, identifier_columns)
    train_features = _train_feature_frame(features, feature_columns)

    if feature_config.winsorization.enabled:
        lower = train_features.quantile(feature_config.winsorization.lower_quantile)
        upper = train_features.quantile(feature_config.winsorization.upper_quantile)
    else:
        lower = pd.Series(float("-inf"), index=feature_columns)
        upper = pd.Series(float("inf"), index=feature_columns)

    clipped_train = train_features.clip(lower=lower, upper=upper, axis="columns")
    means = clipped_train.mean()
    scales = clipped_train.std(ddof=0).replace(0.0, 1.0)

    return NormalizationArtifact(
        feature_version=feature_config.feature_version,
        fit_split=feature_config.normalization.fit_split,
        feature_columns=list(feature_columns),
        winsorization_lower=lower.astype(float).to_dict(),
        winsorization_upper=upper.astype(float).to_dict(),
        means=means.astype(float).to_dict(),
        scales=scales.astype(float).to_dict(),
    )


def transform_features(
    features: pd.DataFrame,
    artifact: NormalizationArtifact,
    identifier_columns: Sequence[str] = DEFAULT_IDENTIFIER_COLUMNS,
) -> pd.DataFrame:
    """Apply train-fitted winsorization and standard scaling to feature columns."""
    missing = [column for column in artifact.feature_columns if column not in features]
    if missing:
        raise ValueError(f"features are missing fitted feature columns: {missing}")

    result = features.copy()
    transformed = result.loc[:, artifact.feature_columns].apply(
        pd.to_numeric,
        errors="raise",
    )
    if not transformed.notna().all().all():
        raise ValueError("features contain NaN values before normalization")

    lower = pd.Series(artifact.winsorization_lower)
    upper = pd.Series(artifact.winsorization_upper)
    means = pd.Series(artifact.means)
    scales = pd.Series(artifact.scales)

    transformed = transformed.clip(lower=lower, upper=upper, axis="columns")
    transformed = (transformed - means) / scales
    result.loc[:, artifact.feature_columns] = transformed

    _validate_preserved_columns(result, features, identifier_columns)
    return result


def save_normalization_artifact(
    artifact: NormalizationArtifact,
    artifact_path: str | Path,
) -> None:
    """Persist the train-fitted normalization artifact."""
    path = Path(artifact_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as artifact_file:
        pickle.dump(artifact, artifact_file)


def load_normalization_artifact(artifact_path: str | Path) -> NormalizationArtifact:
    """Load a persisted normalization artifact."""
    with Path(artifact_path).open("rb") as artifact_file:
        artifact = pickle.load(artifact_file)
    if not isinstance(artifact, NormalizationArtifact):
        raise TypeError("normalization artifact has unexpected type")
    return artifact


def _validate_features_for_fit(
    features: pd.DataFrame,
    identifier_columns: Sequence[str],
) -> None:
    if "split" not in features.columns:
        raise ValueError("features must include a split column")
    missing_identifiers = [
        column for column in identifier_columns if column not in features.columns
    ]
    if missing_identifiers:
        raise ValueError(f"features are missing identifier columns: {missing_identifiers}")
    if not (features["split"] == TRAIN_SPLIT).any():
        raise ValueError("features must contain at least one train row")


def _feature_columns(
    features: pd.DataFrame,
    identifier_columns: Sequence[str],
) -> list[str]:
    identifiers = set(identifier_columns)
    feature_columns = [column for column in features.columns if column not in identifiers]
    if not feature_columns:
        raise ValueError("features must contain at least one feature column")
    return feature_columns


def _train_feature_frame(
    features: pd.DataFrame,
    feature_columns: Sequence[str],
) -> pd.DataFrame:
    train = features.loc[features["split"] == TRAIN_SPLIT, feature_columns].apply(
        pd.to_numeric,
        errors="raise",
    )
    if not train.notna().all().all():
        raise ValueError("train features contain NaN values before normalization")
    if train.isin([float("inf"), float("-inf")]).any().any():
        raise ValueError("train features contain infinite values before normalization")
    return train


def _validate_preserved_columns(
    result: pd.DataFrame,
    original: pd.DataFrame,
    identifier_columns: Sequence[str],
) -> None:
    for column in identifier_columns:
        if not result[column].equals(original[column]):
            raise ValueError(f"identifier column was modified: {column}")

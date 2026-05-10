"""Phase 2 in-memory dataset built from Phase 1 artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from portfolio_rl.features.feature_spec import FeatureSpec, load_feature_spec


DEFAULT_MODEL_MATRIX_PATH = Path("data/processed/model_matrix_daily.parquet")


@dataclass(frozen=True)
class PortfolioDataset:
    """Storage-agnostic arrays consumed by the Phase 2 environment."""

    dates: pd.DatetimeIndex
    splits: np.ndarray
    market_features: np.ndarray
    returns: np.ndarray
    asset_order: list[str]
    feature_version: str
    observation_dim: int

    @property
    def n_assets(self) -> int:
        return len(self.asset_order)

    @property
    def market_feature_dim(self) -> int:
        return self.observation_dim - self.n_assets


def load_portfolio_dataset(
    root: str | Path = ".",
    *,
    model_matrix_path: str | Path = DEFAULT_MODEL_MATRIX_PATH,
    feature_spec_path: str | Path | None = None,
) -> PortfolioDataset:
    """Load Phase 1 model-matrix artifacts into a Phase 2 dataset."""
    root_path = Path(root)
    model_matrix = pd.read_parquet(_resolve_path(root_path, model_matrix_path))
    feature_spec = load_feature_spec(
        _resolve_path(
            root_path,
            feature_spec_path or "artifacts/feature_specs/feature_spec_v1.json",
        )
    )
    return build_portfolio_dataset(model_matrix, feature_spec)


def build_portfolio_dataset(
    model_matrix: pd.DataFrame,
    feature_spec: FeatureSpec,
) -> PortfolioDataset:
    """Build a storage-agnostic dataset from a model matrix and feature spec."""
    _validate_feature_spec(feature_spec)
    expected_columns = _expected_model_matrix_columns(feature_spec)
    actual_columns = list(model_matrix.columns)
    if actual_columns != expected_columns:
        raise ValueError(
            "model_matrix columns do not match feature_spec order: "
            f"expected={expected_columns}, actual={actual_columns}"
        )

    dates = pd.DatetimeIndex(pd.to_datetime(model_matrix["date"]))
    if dates.duplicated().any():
        raise ValueError("model_matrix dates must be unique")
    if not dates.is_monotonic_increasing:
        raise ValueError("model_matrix dates must be sorted ascending")

    feature_versions = set(model_matrix["feature_version"].astype(str))
    if feature_versions != {feature_spec.feature_version}:
        raise ValueError(
            "model_matrix feature_version does not match feature_spec: "
            f"{sorted(feature_versions)} != {feature_spec.feature_version}"
        )

    obs_columns = _observation_columns(feature_spec)
    return_columns = _return_columns(feature_spec)
    obs = model_matrix.loc[:, obs_columns].to_numpy(dtype=np.float32)
    returns = model_matrix.loc[:, return_columns].to_numpy(dtype=np.float32)
    _assert_finite(obs, "model_matrix observation values")
    _assert_finite(returns, "model_matrix return values")

    market_feature_dim = feature_spec.observation_dim - len(feature_spec.asset_order)
    market_features = obs[:, :market_feature_dim]
    return PortfolioDataset(
        dates=dates,
        splits=model_matrix["split"].astype(str).to_numpy(),
        market_features=market_features,
        returns=returns,
        asset_order=list(feature_spec.asset_order),
        feature_version=feature_spec.feature_version,
        observation_dim=feature_spec.observation_dim,
    )


def _resolve_path(root: Path, path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return root / candidate


def _validate_feature_spec(feature_spec: FeatureSpec) -> None:
    if not feature_spec.asset_order:
        raise ValueError("feature_spec asset_order must not be empty")
    if feature_spec.observation_dim <= len(feature_spec.asset_order):
        raise ValueError("feature_spec observation_dim must include market features")


def _expected_model_matrix_columns(feature_spec: FeatureSpec) -> list[str]:
    return [
        "date",
        "split",
        "feature_version",
        *_observation_columns(feature_spec),
        *_return_columns(feature_spec),
    ]


def _observation_columns(feature_spec: FeatureSpec) -> list[str]:
    return [f"obs_{index:03d}" for index in range(feature_spec.observation_dim)]


def _return_columns(feature_spec: FeatureSpec) -> list[str]:
    return [f"return_{ticker.lower()}_1d" for ticker in feature_spec.asset_order]


def _assert_finite(values: np.ndarray, label: str) -> None:
    if not np.isfinite(values).all():
        raise ValueError(f"{label} must be finite")

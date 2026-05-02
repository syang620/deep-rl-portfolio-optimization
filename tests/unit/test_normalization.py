from __future__ import annotations

from math import isclose
import pickle

import pandas as pd
import pytest

from portfolio_rl.config.schemas import FeaturesConfig
from portfolio_rl.features.normalization import (
    NormalizationArtifact,
    fit_normalization_artifact,
    load_normalization_artifact,
    normalize_features,
    transform_features,
)


def test_normalize_features_fits_scaler_on_train_only(tmp_path) -> None:
    artifact_path = tmp_path / "feature_scaler_v1.pkl"
    features = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2020-01-01", "2020-01-02", "2024-01-01", "2025-01-01"]
            ),
            "ticker": ["SPY", "SPY", "SPY", "SPY"],
            "split": ["train", "train", "validation", "test"],
            "feature_version": ["v1"] * 4,
            "ret_1d": [0.0, 10.0, 1_000.0, -1_000.0],
            "vol_21d": [2.0, 4.0, 200.0, -200.0],
        }
    )

    normalized = normalize_features(
        features,
        _feature_config(winsorization_enabled=False),
        artifact_path=artifact_path,
    )
    artifact = load_normalization_artifact(artifact_path)

    assert artifact.means == {"ret_1d": 5.0, "vol_21d": 3.0}
    assert artifact.scales == {"ret_1d": 5.0, "vol_21d": 1.0}
    assert normalized.loc[:1, "ret_1d"].to_list() == [-1.0, 1.0]
    assert normalized.loc[:1, "vol_21d"].to_list() == [-1.0, 1.0]
    assert normalized[["date", "ticker", "split", "feature_version"]].equals(
        features[["date", "ticker", "split", "feature_version"]]
    )


def test_winsorization_thresholds_are_fit_on_train_only() -> None:
    features = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2020-01-02", "2024-01-01"]),
            "ticker": ["SPY", "SPY", "SPY"],
            "split": ["train", "train", "validation"],
            "feature_version": ["v1", "v1", "v1"],
            "ret_1d": [0.0, 10.0, 1_000.0],
        }
    )

    artifact = fit_normalization_artifact(features, _feature_config())
    transformed = transform_features(features, artifact)

    assert artifact.winsorization_upper["ret_1d"] < 10.0
    assert artifact.winsorization_upper["ret_1d"] < 1_000.0
    assert isclose(
        transformed.loc[2, "ret_1d"],
        transformed.loc[1, "ret_1d"],
    )


def test_normalize_features_writes_pickle_artifact(tmp_path) -> None:
    artifact_path = tmp_path / "feature_scaler_v1.pkl"

    normalize_features(
        _features_fixture(),
        _feature_config(),
        artifact_path=artifact_path,
    )

    with artifact_path.open("rb") as artifact_file:
        artifact = pickle.load(artifact_file)
    assert isinstance(artifact, NormalizationArtifact)
    assert artifact.feature_version == "v1"
    assert artifact.fit_split == "train"
    assert artifact.feature_columns == ["ret_1d", "vol_21d"]


def test_fit_normalization_artifact_fails_without_train_rows() -> None:
    features = _features_fixture()
    features["split"] = "validation"

    with pytest.raises(ValueError, match="at least one train row"):
        fit_normalization_artifact(features, _feature_config())


def test_fit_normalization_artifact_fails_on_missing_split_column() -> None:
    features = _features_fixture().drop(columns=["split"])

    with pytest.raises(ValueError, match="split column"):
        fit_normalization_artifact(features, _feature_config())


def test_transform_features_requires_fitted_columns() -> None:
    artifact = fit_normalization_artifact(_features_fixture(), _feature_config())
    features = _features_fixture().drop(columns=["vol_21d"])

    with pytest.raises(ValueError, match="missing fitted feature columns"):
        transform_features(features, artifact)


def _features_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2020-01-02", "2024-01-01"]),
            "ticker": ["SPY", "SPY", "SPY"],
            "split": ["train", "train", "validation"],
            "feature_version": ["v1", "v1", "v1"],
            "ret_1d": [1.0, 2.0, 3.0],
            "vol_21d": [2.0, 4.0, 6.0],
        }
    )


def _feature_config(winsorization_enabled: bool = True) -> FeaturesConfig:
    return FeaturesConfig.model_validate(
        {
            "feature_version": "v1",
            "market": {
                "benchmark_ticker": "SPY",
                "credit_proxy_safe_ticker": "IEF",
                "credit_proxy_risk_ticker": "HYG",
            },
            "return_windows": [1, 5, 21, 63, 126, 252],
            "volatility_windows": [21, 63],
            "drawdown_windows": [63],
            "rsi_windows": [14],
            "price_z_windows": [20, 50],
            "correlation_windows": [63],
            "winsorization": {
                "enabled": winsorization_enabled,
                "lower_quantile": 0.005,
                "upper_quantile": 0.995,
            },
            "normalization": {
                "method": "standard_scaler",
                "fit_split": "train",
            },
        }
    )

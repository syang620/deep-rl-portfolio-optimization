from __future__ import annotations

import pandas as pd
import pytest

from portfolio_rl.features.feature_spec import FeatureSpec, flatten_features
from portfolio_rl.features.model_matrix import build_model_matrix


def test_build_model_matrix_uses_feature_spec_order_and_equal_weights() -> None:
    spec = _feature_spec()
    normalized_asset_features = _normalized_asset_features()
    normalized_global_features = _normalized_global_features()
    raw_asset_features = _raw_asset_features()

    model_matrix = build_model_matrix(
        normalized_asset_features,
        normalized_global_features,
        raw_asset_features,
        spec,
    )
    expected_observation = flatten_features(
        normalized_asset_features,
        normalized_global_features.iloc[0],
        {"SPY": 0.5, "QQQ": 0.5},
        spec,
    )

    assert model_matrix.loc[0, "date"] == pd.Timestamp("2024-01-02")
    assert model_matrix.loc[0, "split"] == "validation"
    assert model_matrix.loc[0, "feature_version"] == "v1"
    assert model_matrix.loc[0, ["obs_000", "obs_001", "obs_002", "obs_003", "obs_004", "obs_005", "obs_006"]].to_list() == expected_observation


def test_build_model_matrix_uses_raw_one_day_returns() -> None:
    model_matrix = build_model_matrix(
        _normalized_asset_features(),
        _normalized_global_features(),
        _raw_asset_features(),
        _feature_spec(),
    )

    assert model_matrix.loc[0, "return_spy_1d"] == 0.01
    assert model_matrix.loc[0, "return_qqq_1d"] == -0.02


def test_build_model_matrix_fails_on_missing_asset_coverage() -> None:
    normalized_asset_features = _normalized_asset_features().loc[
        lambda frame: frame["ticker"] != "QQQ"
    ]

    with pytest.raises(ValueError, match="incomplete coverage"):
        build_model_matrix(
            normalized_asset_features,
            _normalized_global_features(),
            _raw_asset_features(),
            _feature_spec(),
        )


def _feature_spec() -> FeatureSpec:
    return FeatureSpec(
        feature_version="v1",
        asset_order=["SPY", "QQQ"],
        per_asset_features=["ret_1d", "vol_21d"],
        global_features=["vix_z_21d"],
        current_weight_features=["weight_spy", "weight_qqq"],
        observation_dim=7,
        created_at="2026-04-30T00:00:00+00:00",
    )


def _normalized_asset_features() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-02"]),
            "ticker": ["QQQ", "SPY"],
            "split": ["validation", "validation"],
            "feature_version": ["v1", "v1"],
            "ret_1d": [30.0, 10.0],
            "vol_21d": [40.0, 20.0],
        }
    )


def _normalized_global_features() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02"]),
            "split": ["validation"],
            "feature_version": ["v1"],
            "vix_z_21d": [50.0],
        }
    )


def _raw_asset_features() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-02"]),
            "ticker": ["SPY", "QQQ"],
            "ret_1d": [0.01, -0.02],
        }
    )

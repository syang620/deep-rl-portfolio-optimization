from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from portfolio_rl.config.loader import load_features_config, load_universe_config
from portfolio_rl.config.schemas import AssetConfig, FeaturesConfig, UniverseConfig
from portfolio_rl.features.feature_spec import (
    FeatureSpec,
    build_feature_spec,
    flatten_features,
    load_feature_spec,
    save_feature_spec,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"


def test_build_feature_spec_from_repo_configs_preserves_asset_order() -> None:
    features_config = load_features_config(CONFIG_DIR / "features.yaml")
    spec = build_feature_spec(
        load_universe_config(CONFIG_DIR / "universe.yaml"),
        features_config,
    )

    assert features_config.market.benchmark_ticker == "SPY"
    assert spec.asset_order == [
        "SPY",
        "QQQ",
        "IWM",
        "EFA",
        "EEM",
        "TLT",
        "IEF",
        "SHY",
        "LQD",
        "HYG",
        "GLD",
        "DBC",
        "VNQ",
        "XLU",
    ]
    assert spec.per_asset_features[:6] == [
        "ret_1d",
        "ret_5d",
        "ret_21d",
        "ret_63d",
        "ret_126d",
        "ret_252d",
    ]
    assert "corr_to_spy_63d" in spec.per_asset_features
    assert "beta_to_spy_63d" in spec.per_asset_features
    assert spec.global_features == [
        "vix_z_21d",
        "vix_z_63d",
        "dgs2_change_5d",
        "dgs10_change_5d",
        "yield_curve_10y_2y",
        "credit_spread_z_63d",
        "spy_vol_21d",
        "spy_drawdown_63d",
    ]


def test_feature_spec_observation_dim_is_explicit() -> None:
    spec = build_feature_spec(_universe_config(), _feature_config())

    assert spec.current_weight_features == ["weight_spy", "weight_qqq"]
    assert spec.observation_dim == (
        len(spec.asset_order) * len(spec.per_asset_features)
        + len(spec.global_features)
        + len(spec.current_weight_features)
    )


def test_feature_spec_save_load_round_trips_json(tmp_path: Path) -> None:
    path = tmp_path / "feature_spec_v1.json"
    spec = build_feature_spec(_universe_config(), _feature_config())

    save_feature_spec(spec, path)
    loaded = load_feature_spec(path)

    assert loaded == spec


def test_flatten_features_respects_feature_spec_order() -> None:
    spec = FeatureSpec(
        feature_version="v1",
        asset_order=["SPY", "QQQ"],
        per_asset_features=["ret_1d", "vol_21d"],
        global_features=["vix_z_21d"],
        current_weight_features=["weight_spy", "weight_qqq"],
        observation_dim=7,
        created_at="2026-04-26T00:00:00+00:00",
    )
    asset_features = pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-02"],
            "ticker": ["QQQ", "SPY"],
            "ret_1d": [3.0, 1.0],
            "vol_21d": [4.0, 2.0],
        }
    )
    global_features = pd.DataFrame({"date": ["2024-01-02"], "vix_z_21d": [5.0]})

    flattened = flatten_features(
        asset_features,
        global_features,
        {"QQQ": 0.4, "SPY": 0.6},
        spec,
    )

    assert flattened == [1.0, 2.0, 3.0, 4.0, 5.0, 0.6, 0.4]


def test_flatten_features_fails_when_asset_feature_is_missing() -> None:
    spec = FeatureSpec(
        feature_version="v1",
        asset_order=["SPY"],
        per_asset_features=["ret_1d", "vol_21d"],
        global_features=[],
        current_weight_features=["weight_spy"],
        observation_dim=3,
        created_at="2026-04-26T00:00:00+00:00",
    )
    asset_features = pd.DataFrame({"ticker": ["SPY"], "ret_1d": [1.0]})

    with pytest.raises(ValueError, match="missing columns"):
        flatten_features(asset_features, pd.Series(dtype=float), {"SPY": 1.0}, spec)


def test_flatten_features_fails_when_current_weight_is_missing() -> None:
    spec = FeatureSpec(
        feature_version="v1",
        asset_order=["SPY"],
        per_asset_features=["ret_1d"],
        global_features=[],
        current_weight_features=["weight_spy"],
        observation_dim=2,
        created_at="2026-04-26T00:00:00+00:00",
    )
    asset_features = pd.DataFrame({"ticker": ["SPY"], "ret_1d": [1.0]})

    with pytest.raises(ValueError, match="current_weights is missing ticker"):
        flatten_features(asset_features, pd.Series(dtype=float), {}, spec)


def _universe_config() -> UniverseConfig:
    return UniverseConfig(
        universe_name="test_universe",
        assets=[
            AssetConfig(ticker="SPY", asset_class="us_large_cap_equity"),
            AssetConfig(ticker="QQQ", asset_class="us_growth_equity"),
        ],
    )


def _feature_config() -> FeaturesConfig:
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

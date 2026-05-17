from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from portfolio_rl.data.dataset import build_portfolio_dataset, load_portfolio_dataset
from portfolio_rl.data.feature_store import PortfolioFeatureStore
from portfolio_rl.features.feature_spec import FeatureSpec, save_feature_spec


def test_build_portfolio_dataset_removes_static_weight_slice() -> None:
    dataset = build_portfolio_dataset(_model_matrix(), _feature_spec())

    assert dataset.observation_dim == 5
    assert dataset.n_assets == 2
    assert dataset.market_feature_dim == 3
    assert dataset.market_features.shape == (8, 3)
    assert dataset.market_features[0].tolist() == [10.0, 20.0, 30.0]


def test_build_portfolio_dataset_orders_returns_by_feature_spec_assets() -> None:
    dataset = build_portfolio_dataset(_model_matrix(), _feature_spec())

    assert dataset.asset_order == ["SPY", "QQQ"]
    np.testing.assert_allclose(dataset.returns[0], [0.01, 0.11])
    np.testing.assert_allclose(dataset.returns[1], [0.02, 0.12])


def test_load_portfolio_dataset_reads_phase1_artifacts(tmp_path: Path) -> None:
    model_matrix_path = tmp_path / "data/processed/model_matrix_daily.parquet"
    feature_spec_path = tmp_path / "artifacts/feature_specs/feature_spec_v1.json"
    model_matrix_path.parent.mkdir(parents=True)
    _model_matrix().to_parquet(model_matrix_path, index=False)
    save_feature_spec(_feature_spec(), feature_spec_path)

    dataset = load_portfolio_dataset(tmp_path)

    assert dataset.dates[0] == pd.Timestamp("2024-01-02")
    assert dataset.feature_version == "v1"
    assert dataset.market_features.shape == (8, 3)


def test_feature_store_filters_split_and_exposes_market_features() -> None:
    store = PortfolioFeatureStore(
        build_portfolio_dataset(_model_matrix(), _feature_spec()),
        split="train",
    )

    assert store.split == "train"
    assert store.n_rows == 6
    assert store.n_assets == 2
    assert store.observation_dim == 5
    assert store.market_feature_dim == 3
    assert store.date_at(1) == pd.Timestamp("2024-01-03")
    assert store.get_market_features(1).tolist() == [11.0, 21.0, 31.0]


def test_feature_store_forward_returns_skip_same_row() -> None:
    store = PortfolioFeatureStore(
        build_portfolio_dataset(_model_matrix(), _feature_spec()),
        split="train",
    )

    forward = store.get_forward_log_returns(relative_idx=0, horizon=5)

    assert forward.shape == (5, 2)
    np.testing.assert_allclose(
        forward[:, 0],
        [0.02, 0.03, 0.04, 0.05, 0.06],
    )
    np.testing.assert_allclose(
        forward[:, 1],
        [0.12, 0.13, 0.14, 0.15, 0.16],
    )


def test_feature_store_rejects_forward_returns_past_split_boundary() -> None:
    store = PortfolioFeatureStore(
        build_portfolio_dataset(_model_matrix(), _feature_spec()),
        split="train",
    )

    with pytest.raises(IndexError, match="exceeds split boundary"):
        store.get_forward_log_returns(relative_idx=1, horizon=5)


def test_feature_store_trailing_returns_end_at_current_row() -> None:
    store = PortfolioFeatureStore(
        build_portfolio_dataset(_model_matrix(), _feature_spec()),
        split="train",
    )

    trailing = store.get_trailing_log_returns(relative_idx=2, lookback=3)

    assert trailing.shape == (3, 2)
    np.testing.assert_allclose(trailing[:, 0], [0.01, 0.02, 0.03])
    np.testing.assert_allclose(trailing[:, 1], [0.11, 0.12, 0.13])


def test_feature_store_rejects_trailing_returns_past_split_start() -> None:
    store = PortfolioFeatureStore(
        build_portfolio_dataset(_model_matrix(), _feature_spec()),
        split="train",
    )

    with pytest.raises(IndexError, match="exceeds split boundary"):
        store.get_trailing_log_returns(relative_idx=1, lookback=3)


def test_feature_store_max_valid_start_index_requires_full_window() -> None:
    store = PortfolioFeatureStore(
        build_portfolio_dataset(_model_matrix(), _feature_spec()),
        split="train",
    )

    assert store.max_valid_start_index(episode_length_trading_days=5) == 0
    with pytest.raises(ValueError, match="split is too short"):
        store.max_valid_start_index(episode_length_trading_days=6)


def test_feature_store_rejects_unknown_split() -> None:
    dataset = build_portfolio_dataset(_model_matrix(), _feature_spec())

    with pytest.raises(ValueError, match="does not contain split"):
        PortfolioFeatureStore(dataset, split="test")


def test_feature_store_rejects_non_contiguous_split() -> None:
    model_matrix = _model_matrix()
    model_matrix.loc[2, "split"] = "validation"

    with pytest.raises(ValueError, match="not contiguous"):
        PortfolioFeatureStore(
            build_portfolio_dataset(model_matrix, _feature_spec()),
            split="train",
        )


def test_build_portfolio_dataset_rejects_schema_drift() -> None:
    model_matrix = _model_matrix().drop(columns=["obs_004"])

    with pytest.raises(ValueError, match="columns do not match feature_spec order"):
        build_portfolio_dataset(model_matrix, _feature_spec())


def _feature_spec() -> FeatureSpec:
    return FeatureSpec(
        feature_version="v1",
        asset_order=["SPY", "QQQ"],
        per_asset_features=["ret_1d"],
        global_features=["vix_z_21d"],
        current_weight_features=["weight_spy", "weight_qqq"],
        observation_dim=5,
        created_at="2026-05-07T00:00:00+00:00",
    )


def _model_matrix() -> pd.DataFrame:
    dates = pd.date_range("2024-01-02", periods=8, freq="B")
    rows = []
    for index, date in enumerate(dates):
        rows.append(
            {
                "date": date,
                "split": "train" if index < 6 else "validation",
                "feature_version": "v1",
                "obs_000": 10.0 + index,
                "obs_001": 20.0 + index,
                "obs_002": 30.0 + index,
                "obs_003": 0.5,
                "obs_004": 0.5,
                "return_spy_1d": 0.01 + index / 100.0,
                "return_qqq_1d": 0.11 + index / 100.0,
            }
        )
    return pd.DataFrame(rows)

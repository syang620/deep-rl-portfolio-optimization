from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio_rl.data.dataset import build_portfolio_dataset
from portfolio_rl.data.feature_store import PortfolioFeatureStore
from portfolio_rl.env.episode_sampler import (
    FixedStartEpisodeSampler,
    RandomWindowEpisodeSampler,
)
from portfolio_rl.features.feature_spec import FeatureSpec


def test_random_window_sampler_returns_start_within_valid_bounds() -> None:
    store = _feature_store(n_train_rows=10)
    sampler = RandomWindowEpisodeSampler()
    rng = np.random.default_rng(42)

    starts = [
        sampler.sample_start(store, episode_length_trading_days=5, rng=rng)
        for _ in range(100)
    ]

    assert min(starts) >= 0
    assert max(starts) <= 4


def test_random_window_sampler_is_reproducible_with_same_seed() -> None:
    store = _feature_store(n_train_rows=10)
    sampler = RandomWindowEpisodeSampler()
    first_rng = np.random.default_rng(7)
    second_rng = np.random.default_rng(7)

    first = [
        sampler.sample_start(
            store,
            episode_length_trading_days=5,
            rng=first_rng,
        )
        for _ in range(3)
    ]
    second = [
        sampler.sample_start(
            store,
            episode_length_trading_days=5,
            rng=second_rng,
        )
        for _ in range(3)
    ]

    assert first == second


def test_random_window_sampler_can_return_upper_bound() -> None:
    store = _feature_store(n_train_rows=10)
    sampler = RandomWindowEpisodeSampler()

    start = sampler.sample_start(
        store,
        episode_length_trading_days=5,
        rng=_UpperBoundRng(),
    )

    assert start == 4


def test_fixed_start_sampler_returns_zero() -> None:
    store = _feature_store(n_train_rows=10)
    sampler = FixedStartEpisodeSampler()

    start = sampler.sample_start(
        store,
        episode_length_trading_days=5,
        rng=np.random.default_rng(42),
    )

    assert start == 0


@pytest.mark.parametrize(
    "sampler",
    [
        RandomWindowEpisodeSampler(),
        FixedStartEpisodeSampler(),
    ],
)
def test_samplers_reject_split_too_short(
    sampler: RandomWindowEpisodeSampler | FixedStartEpisodeSampler,
) -> None:
    store = _feature_store(n_train_rows=5)

    with pytest.raises(ValueError, match="split is too short"):
        sampler.sample_start(
            store,
            episode_length_trading_days=5,
            rng=np.random.default_rng(42),
        )


class _UpperBoundRng:
    def integers(self, low: int, high: int) -> int:
        assert low == 0
        return high - 1


def _feature_store(n_train_rows: int) -> PortfolioFeatureStore:
    return PortfolioFeatureStore(
        build_portfolio_dataset(_model_matrix(n_train_rows), _feature_spec()),
        split="train",
    )


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


def _model_matrix(n_train_rows: int) -> pd.DataFrame:
    dates = pd.date_range("2024-01-02", periods=n_train_rows + 2, freq="B")
    rows = []
    for index, date in enumerate(dates):
        rows.append(
            {
                "date": date,
                "split": "train" if index < n_train_rows else "validation",
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

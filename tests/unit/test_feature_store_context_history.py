from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio_rl.data.dataset import PortfolioDataset
from portfolio_rl.data.feature_store import PortfolioFeatureStore


def test_pre_window_returns_cross_split_boundary_without_evaluation_rows() -> None:
    dataset = _dataset()
    store = PortfolioFeatureStore(dataset, split="validation")

    context = store.get_pre_window_log_returns(3)

    np.testing.assert_array_equal(context, dataset.returns[3:6])
    assert context.shape == (3, 2)
    assert not np.shares_memory(context, dataset.returns)


def test_pre_window_returns_respect_explicit_evaluation_start() -> None:
    dataset = _dataset()
    store = PortfolioFeatureStore(
        dataset,
        split="validation",
        start_date="2024-01-11",
    )

    context = store.get_pre_window_log_returns(2)

    np.testing.assert_array_equal(context, dataset.returns[6:8])


def test_pre_window_returns_raise_when_earlier_history_is_insufficient() -> None:
    store = PortfolioFeatureStore(_dataset(), split="train")

    with pytest.raises(IndexError, match="insufficient"):
        store.get_pre_window_log_returns(1)


def test_existing_forward_and_trailing_boundaries_remain_split_bounded() -> None:
    store = PortfolioFeatureStore(_dataset(), split="validation")

    with pytest.raises(IndexError, match="forward return window"):
        store.get_forward_log_returns(2, 1)
    with pytest.raises(IndexError, match="trailing return window"):
        store.get_trailing_log_returns(0, 2)


def _dataset() -> PortfolioDataset:
    rows = 9
    return PortfolioDataset(
        dates=pd.date_range("2024-01-01", periods=rows, freq="B"),
        splits=np.array(["train"] * 6 + ["validation"] * 3),
        market_features=np.zeros((rows, 2), dtype=np.float32),
        returns=np.arange(rows * 2, dtype=np.float32).reshape(rows, 2),
        asset_order=["SPY", "SHY"],
        feature_version="v1",
        observation_dim=4,
    )

"""Split-aware accessors for Phase 2 portfolio features and returns."""

from __future__ import annotations

import numpy as np
import pandas as pd

from portfolio_rl.data.dataset import PortfolioDataset


class PortfolioFeatureStore:
    """Provide split-bounded market features and forward return windows."""

    def __init__(self, dataset: PortfolioDataset, split: str) -> None:
        split_name = split.strip()
        if not split_name:
            raise ValueError("split must not be empty")

        mask = dataset.splits == split_name
        if not mask.any():
            raise ValueError(f"dataset does not contain split: {split_name}")

        indices = np.flatnonzero(mask)
        if not np.array_equal(indices, np.arange(indices[0], indices[-1] + 1)):
            raise ValueError(f"dataset split is not contiguous: {split_name}")

        self._split = split_name
        self._dates = dataset.dates[indices]
        self._market_features = dataset.market_features[indices]
        self._returns = dataset.returns[indices]
        self._asset_order = list(dataset.asset_order)
        self._feature_version = dataset.feature_version
        self._observation_dim = dataset.observation_dim

    @property
    def split(self) -> str:
        return self._split

    @property
    def n_assets(self) -> int:
        return len(self._asset_order)

    @property
    def asset_order(self) -> list[str]:
        return list(self._asset_order)

    @property
    def feature_version(self) -> str:
        return self._feature_version

    @property
    def observation_dim(self) -> int:
        return self._observation_dim

    @property
    def market_feature_dim(self) -> int:
        return self._observation_dim - self.n_assets

    @property
    def n_rows(self) -> int:
        return len(self._dates)

    def get_market_features(self, relative_idx: int) -> np.ndarray:
        """Return market-only features for one split-relative row."""
        self._validate_relative_idx(relative_idx)
        return self._market_features[relative_idx].copy()

    def get_forward_log_returns(
        self,
        relative_idx: int,
        horizon: int,
    ) -> np.ndarray:
        """Return rows i+1 through i+horizon inclusive."""
        self._validate_relative_idx(relative_idx)
        if horizon <= 0:
            raise ValueError("horizon must be positive")

        start = relative_idx + 1
        end = start + horizon
        if end > self.n_rows:
            raise IndexError(
                "forward return window exceeds split boundary: "
                f"relative_idx={relative_idx}, horizon={horizon}, n_rows={self.n_rows}"
            )
        return self._returns[start:end].copy()

    def max_valid_start_index(self, episode_length_trading_days: int) -> int:
        """Return the last start index with a full future holding window."""
        if episode_length_trading_days <= 0:
            raise ValueError("episode_length_trading_days must be positive")
        max_start = self.n_rows - episode_length_trading_days - 1
        if max_start < 0:
            raise ValueError(
                "split is too short for requested episode length: "
                f"{self.n_rows} rows < {episode_length_trading_days + 1}"
            )
        return max_start

    def date_at(self, relative_idx: int) -> pd.Timestamp:
        """Return the split-relative date at one row."""
        self._validate_relative_idx(relative_idx)
        return pd.Timestamp(self._dates[relative_idx])

    def _validate_relative_idx(self, relative_idx: int) -> None:
        if relative_idx < 0 or relative_idx >= self.n_rows:
            raise IndexError(
                f"relative_idx out of range: {relative_idx}; n_rows={self.n_rows}"
            )

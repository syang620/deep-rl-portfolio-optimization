"""Episode start samplers for split-relative portfolio environments."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from portfolio_rl.data.feature_store import PortfolioFeatureStore


class EpisodeSampler(Protocol):
    """Protocol for choosing a valid split-relative episode start index."""

    def sample_start(
        self,
        store: PortfolioFeatureStore,
        episode_length_trading_days: int,
        rng: np.random.Generator,
    ) -> int:
        """Return a split-relative start index for one episode."""
        ...


class RandomWindowEpisodeSampler:
    """Sample random contiguous training windows inside a split."""

    def sample_start(
        self,
        store: PortfolioFeatureStore,
        episode_length_trading_days: int,
        rng: np.random.Generator,
    ) -> int:
        """Return a uniformly sampled valid start index."""
        max_start = store.max_valid_start_index(episode_length_trading_days)
        return int(rng.integers(0, max_start + 1))


class FixedStartEpisodeSampler:
    """Use a deterministic first-window start for evaluation."""

    def sample_start(
        self,
        store: PortfolioFeatureStore,
        episode_length_trading_days: int,
        rng: np.random.Generator,
    ) -> int:
        """Return the first valid split-relative start index."""
        del rng
        store.max_valid_start_index(episode_length_trading_days)
        return 0

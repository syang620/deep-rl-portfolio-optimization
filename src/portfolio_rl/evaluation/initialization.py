"""Initial portfolio providers for deterministic evaluation backtests."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

import numpy as np

from portfolio_rl.data.feature_store import PortfolioFeatureStore


class InitialPortfolioProvider(Protocol):
    """Provide the endowed portfolio held at an evaluation window's start."""

    def initial_weights(
        self,
        feature_store: PortfolioFeatureStore,
    ) -> np.ndarray:
        """Return valid long-only weights in feature-store asset order."""


class EqualWeightInitializer:
    """Endow the evaluation with equal asset weights."""

    def initial_weights(
        self,
        feature_store: PortfolioFeatureStore,
    ) -> np.ndarray:
        return np.full(
            feature_store.n_assets,
            1.0 / feature_store.n_assets,
            dtype=np.float64,
        )


class StaticWeightInitializer:
    """Endow the evaluation from an exact ticker-to-weight mapping."""

    def __init__(self, *, weights: Mapping[str, float]) -> None:
        if not isinstance(weights, Mapping):
            raise TypeError("weights must be a ticker-to-weight mapping")
        self._weights = dict(weights)

    def initial_weights(
        self,
        feature_store: PortfolioFeatureStore,
    ) -> np.ndarray:
        expected = set(feature_store.asset_order)
        actual = set(self._weights)
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        if missing or unexpected:
            raise ValueError(
                "static initializer tickers do not match asset_order: "
                f"missing={missing}, unexpected={unexpected}"
            )
        weights = np.asarray(
            [self._weights[ticker] for ticker in feature_store.asset_order],
            dtype=np.float64,
        )
        _validate_initial_weights(weights, feature_store.n_assets)
        return weights


class SHYInitializer:
    """Endow the evaluation with 100% in SHY."""

    def initial_weights(
        self,
        feature_store: PortfolioFeatureStore,
    ) -> np.ndarray:
        if "SHY" not in feature_store.asset_order:
            raise ValueError("SHY initializer requires SHY in asset_order")
        weights = np.zeros(feature_store.n_assets, dtype=np.float64)
        weights[feature_store.asset_order.index("SHY")] = 1.0
        return weights


class InverseVolatilityInitializer:
    """Endow from inverse volatility estimated strictly before the window."""

    def __init__(
        self,
        *,
        lookback: int = 63,
        volatility_floor: float = 1e-8,
    ) -> None:
        if lookback <= 0:
            raise ValueError("lookback must be positive")
        if not np.isfinite(volatility_floor) or volatility_floor <= 0.0:
            raise ValueError("volatility_floor must be positive and finite")
        self._lookback = lookback
        self._volatility_floor = float(volatility_floor)

    def initial_weights(
        self,
        feature_store: PortfolioFeatureStore,
    ) -> np.ndarray:
        returns = feature_store.get_pre_window_log_returns(self._lookback)
        volatility = np.std(returns, axis=0, ddof=0)
        inverse_volatility = 1.0 / np.maximum(
            volatility,
            self._volatility_floor,
        )
        weights = inverse_volatility / inverse_volatility.sum()
        _validate_initial_weights(weights, feature_store.n_assets)
        return weights


def _validate_initial_weights(weights: np.ndarray, n_assets: int) -> None:
    if weights.ndim != 1 or weights.shape != (n_assets,):
        raise ValueError(f"initial_weights must have shape ({n_assets},)")
    if not np.isfinite(weights).all():
        raise ValueError("initial_weights values must be finite")
    if (weights < 0.0).any():
        raise ValueError("initial_weights values must be nonnegative")
    if not np.isclose(weights.sum(), 1.0):
        raise ValueError("initial_weights must sum to one")

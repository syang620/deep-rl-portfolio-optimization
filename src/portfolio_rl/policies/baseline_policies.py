"""Deterministic baseline policies for portfolio backtests."""

from __future__ import annotations

from typing import Any, Mapping, Protocol

import numpy as np


class WeightPolicy(Protocol):
    """Protocol for policies that emit target portfolio weights directly."""

    def target_weights(
        self,
        observation: np.ndarray,
        info: Mapping[str, Any],
    ) -> np.ndarray:
        """Return a long-only target weight vector."""


class EqualWeightWeeklyPolicy:
    """Rebalance to equal weights at every decision."""

    def __init__(self, n_assets: int) -> None:
        self._n_assets = _validate_n_assets(n_assets)

    def target_weights(
        self,
        observation: np.ndarray,
        info: Mapping[str, Any],
    ) -> np.ndarray:
        del observation, info
        return _equal_weight_vector(self._n_assets)


class BuyAndHoldEqualWeightPolicy:
    """Start equal-weight, then target current drifted weights to avoid trading."""

    def __init__(self, n_assets: int) -> None:
        self._n_assets = _validate_n_assets(n_assets)
        self._has_initial_trade = False

    def reset(self) -> None:
        """Reset policy state for a new backtest episode."""
        self._has_initial_trade = False

    def target_weights(
        self,
        observation: np.ndarray,
        info: Mapping[str, Any],
    ) -> np.ndarray:
        del observation
        if not self._has_initial_trade:
            self._has_initial_trade = True
            return _equal_weight_vector(self._n_assets)

        if "current_weights" not in info:
            raise ValueError("info must include current_weights after initial trade")
        weights = np.asarray(info["current_weights"], dtype=np.float64)
        _validate_weight_vector(weights, self._n_assets, "current_weights")
        return weights.astype(np.float32)


class SingleAssetPolicy:
    """Allocate 100% to one configured ticker."""

    def __init__(self, asset_order: list[str], ticker: str) -> None:
        if not asset_order:
            raise ValueError("asset_order must not be empty")
        self._asset_order = [asset.upper() for asset in asset_order]
        self._ticker = ticker.strip().upper()
        if not self._ticker:
            raise ValueError("ticker must not be empty")
        if self._ticker not in self._asset_order:
            raise ValueError(f"ticker is not in asset_order: {self._ticker}")

    def target_weights(
        self,
        observation: np.ndarray,
        info: Mapping[str, Any],
    ) -> np.ndarray:
        del observation, info
        weights = np.zeros(len(self._asset_order), dtype=np.float32)
        weights[self._asset_order.index(self._ticker)] = 1.0
        return weights


class InverseVolatilityPolicy:
    """Allocate target weights proportional to inverse realized volatility."""

    def __init__(self, n_assets: int, volatility_floor: float = 1e-8) -> None:
        self._n_assets = _validate_n_assets(n_assets)
        if not np.isfinite(volatility_floor) or volatility_floor <= 0.0:
            raise ValueError("volatility_floor must be positive and finite")
        self._volatility_floor = float(volatility_floor)

    def target_weights(
        self,
        observation: np.ndarray,
        info: Mapping[str, Any],
    ) -> np.ndarray:
        del observation
        if "trailing_log_returns" not in info:
            raise ValueError("info must include trailing_log_returns")
        trailing_log_returns = np.asarray(
            info["trailing_log_returns"],
            dtype=np.float64,
        )
        _validate_trailing_log_returns(
            trailing_log_returns,
            self._n_assets,
            "trailing_log_returns",
        )
        volatility = np.std(trailing_log_returns, axis=0, ddof=0)
        inverse_volatility = 1.0 / np.maximum(volatility, self._volatility_floor)
        weights = inverse_volatility / inverse_volatility.sum()
        return weights.astype(np.float32)


def _equal_weight_vector(n_assets: int) -> np.ndarray:
    return np.full(n_assets, 1.0 / n_assets, dtype=np.float32)


def _validate_n_assets(n_assets: int) -> int:
    if n_assets <= 0:
        raise ValueError("n_assets must be positive")
    return n_assets


def _validate_weight_vector(
    weights: np.ndarray,
    n_assets: int,
    name: str,
) -> None:
    if weights.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array")
    if weights.shape != (n_assets,):
        raise ValueError(f"{name} must have shape ({n_assets},)")
    if not np.isfinite(weights).all():
        raise ValueError(f"{name} values must be finite")
    if (weights < 0.0).any():
        raise ValueError(f"{name} values must be nonnegative")
    if not np.isclose(weights.sum(), 1.0):
        raise ValueError(f"{name} must sum to one")


def _validate_trailing_log_returns(
    values: np.ndarray,
    n_assets: int,
    name: str,
) -> None:
    if values.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array")
    if values.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one row")
    if values.shape[1] != n_assets:
        raise ValueError(f"{name} asset dimension must match n_assets")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} values must be finite")

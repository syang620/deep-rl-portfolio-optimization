"""Transparent evaluation-only policy overlays."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from portfolio_rl.env.costs import calculate_turnover
from portfolio_rl.policies.baseline_policies import WeightPolicy


@dataclass(frozen=True)
class PartialRebalanceRecord:
    """One raw recommendation and its partially executed target."""

    decision_step: int
    date: Any
    current_weights: tuple[float, ...]
    raw_policy_target: tuple[float, ...]
    executed_target: tuple[float, ...]
    raw_half_l1_turnover: float
    executed_half_l1_turnover: float


class PartialRebalancePolicy:
    """Move an alpha fraction from live weights toward a policy target."""

    def __init__(self, *, base_policy: WeightPolicy, alpha: float) -> None:
        if not np.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be finite and in [0, 1]")
        self._base_policy = base_policy
        self._alpha = float(alpha)
        self._records: list[PartialRebalanceRecord] = []
        self._decision_step = 0

    @property
    def alpha(self) -> float:
        return self._alpha

    @property
    def records(self) -> tuple[PartialRebalanceRecord, ...]:
        """Return immutable audit records from the latest execution."""
        return tuple(self._records)

    def reset(self) -> None:
        """Reset overlay and wrapped policy state before a backtest."""
        self._records.clear()
        self._decision_step = 0
        if hasattr(self._base_policy, "reset"):
            self._base_policy.reset()

    def target_weights(
        self,
        observation: np.ndarray,
        info: Mapping[str, Any],
    ) -> np.ndarray:
        """Return the convex combination of live and raw target weights."""
        if "asset_order" not in info:
            raise ValueError("info must include asset_order")
        if "current_weights" not in info:
            raise ValueError("info must include current_weights")
        n_assets = len(info["asset_order"])
        current_weights = np.asarray(info["current_weights"], dtype=np.float64)
        _validate_weights(current_weights, n_assets, "current_weights")
        raw_target = np.asarray(
            self._base_policy.target_weights(observation, info),
            dtype=np.float64,
        )
        _validate_weights(raw_target, n_assets, "raw_policy_target")
        if self._alpha == 0.0:
            executed_target = current_weights.copy()
        elif self._alpha == 1.0:
            executed_target = raw_target.copy()
        else:
            executed_target = current_weights + self._alpha * (
                raw_target - current_weights
            )
        _validate_weights(executed_target, n_assets, "executed_target")
        self._records.append(
            PartialRebalanceRecord(
                decision_step=self._decision_step,
                date=info.get("date"),
                current_weights=tuple(float(value) for value in current_weights),
                raw_policy_target=tuple(float(value) for value in raw_target),
                executed_target=tuple(float(value) for value in executed_target),
                raw_half_l1_turnover=calculate_turnover(
                    current_weights,
                    raw_target,
                ),
                executed_half_l1_turnover=calculate_turnover(
                    current_weights,
                    executed_target,
                ),
            )
        )
        self._decision_step += 1
        return executed_target


def _validate_weights(weights: np.ndarray, n_assets: int, label: str) -> None:
    if weights.shape != (n_assets,):
        raise ValueError(f"{label} must have shape ({n_assets},)")
    if not np.isfinite(weights).all():
        raise ValueError(f"{label} must be finite")
    if (weights < 0.0).any():
        raise ValueError(f"{label} must be nonnegative")
    if not np.isclose(weights.sum(), 1.0):
        raise ValueError(f"{label} must sum to one")

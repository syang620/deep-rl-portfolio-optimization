"""Stable-Baselines3 policy adapters for deterministic backtests."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from portfolio_rl.env.action import action_to_weights


class SB3WeightPolicy:
    """Adapt an SB3 action policy to the target-weight backtest interface."""

    def __init__(self, model: Any, action_temperature: float) -> None:
        if action_temperature <= 0.0 or not np.isfinite(action_temperature):
            raise ValueError("action_temperature must be positive and finite")
        self._model = model
        self._action_temperature = float(action_temperature)

    def target_weights(
        self,
        observation: np.ndarray,
        info: Mapping[str, Any],
    ) -> np.ndarray:
        """Predict a raw action and convert it to long-only target weights."""
        raw_action, _state = self._model.predict(observation, deterministic=True)
        weights = action_to_weights(
            np.asarray(raw_action),
            temperature=self._action_temperature,
        )
        _validate_weight_shape(weights, info)
        return weights


def load_sb3_weight_policy(
    model_path: str | Path,
    action_temperature: float,
) -> SB3WeightPolicy:
    """Load a saved SB3 PPO model as a deterministic target-weight policy."""
    from stable_baselines3 import PPO

    return SB3WeightPolicy(
        model=PPO.load(Path(model_path)),
        action_temperature=action_temperature,
    )


def _validate_weight_shape(
    weights: np.ndarray,
    info: Mapping[str, Any],
) -> None:
    if "asset_order" not in info:
        return
    n_assets = len(info["asset_order"])
    if weights.shape != (n_assets,):
        raise ValueError(
            "model action dimension does not match asset_order: "
            f"{weights.shape} != ({n_assets},)"
        )

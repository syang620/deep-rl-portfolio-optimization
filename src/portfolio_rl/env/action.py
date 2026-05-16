"""Action transforms for long-only portfolio allocation."""

from __future__ import annotations

import numpy as np


def action_to_weights(action: np.ndarray, temperature: float) -> np.ndarray:
    """Convert normalized raw actions into long-only portfolio weights."""
    action_values = np.asarray(action, dtype=np.float64)
    _validate_action(action_values)
    _validate_temperature(temperature)

    logits = np.clip(action_values, -1.0, 1.0) * float(temperature)
    logits = logits - np.max(logits)
    exp_logits = np.exp(logits)
    weights = exp_logits / exp_logits.sum()
    return weights.astype(np.float32)


def _validate_action(action: np.ndarray) -> None:
    if action.ndim != 1:
        raise ValueError("action must be a one-dimensional array")
    if action.size == 0:
        raise ValueError("action must not be empty")
    if not np.isfinite(action).all():
        raise ValueError("action values must be finite")


def _validate_temperature(temperature: float) -> None:
    if not np.isfinite(temperature):
        raise ValueError("temperature must be finite")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

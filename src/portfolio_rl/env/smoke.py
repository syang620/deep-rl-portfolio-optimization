"""Smoke checks for PortfolioEnv episodes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from portfolio_rl.env.portfolio_env import PortfolioEnv


@dataclass(frozen=True)
class DummyAgentSmokeResult:
    """Summary from one dummy random-agent environment episode."""

    steps: int
    final_nav: float
    cumulative_turnover: float
    final_date: pd.Timestamp
    final_weights: np.ndarray

    def to_dict(self) -> dict[str, Any]:
        return {
            "steps": self.steps,
            "final_nav": self.final_nav,
            "cumulative_turnover": self.cumulative_turnover,
            "final_date": self.final_date.isoformat(),
            "final_weights": self.final_weights.tolist(),
        }


def run_dummy_random_agent_episode(
    env: PortfolioEnv,
    *,
    seed: int = 42,
) -> DummyAgentSmokeResult:
    """Run one full random-action episode and validate numeric health."""
    rng = np.random.default_rng(seed)
    observation, info = env.reset(seed=seed)
    _assert_finite_array(observation, "reset observation")
    _assert_positive_finite(float(info["portfolio_value"]), "reset portfolio_value")

    cumulative_turnover = 0.0
    final_info = info
    steps = 0

    for step_index in range(env.config.max_episode_steps):
        action = rng.uniform(-1.0, 1.0, size=env.n_assets).astype(np.float32)
        observation, reward, terminated, truncated, final_info = env.step(action)
        steps = step_index + 1

        _assert_finite_array(observation, "step observation")
        _assert_finite_scalar(float(reward), "reward")
        _assert_positive_finite(
            float(final_info["portfolio_value"]),
            "portfolio_value",
        )

        turnover = final_info.get("turnover")
        if turnover is not None:
            turnover_value = float(turnover)
            _assert_nonnegative_finite(turnover_value, "turnover")
            cumulative_turnover += turnover_value

        _assert_finite_array(env.current_weights, "current_weights")
        _assert_nonnegative_finite_array(env.current_weights, "current_weights")
        if not np.isclose(env.current_weights.sum(), 1.0):
            raise RuntimeError("current_weights must sum to one")

        if terminated:
            raise RuntimeError("dummy-agent smoke episode terminated unexpectedly")
        if truncated:
            break

    if steps != env.config.max_episode_steps:
        raise RuntimeError(
            "dummy-agent smoke episode did not reach max_episode_steps: "
            f"{steps} != {env.config.max_episode_steps}"
        )
    if not final_info.get("date"):
        raise RuntimeError("final info must include date")

    return DummyAgentSmokeResult(
        steps=steps,
        final_nav=float(final_info["portfolio_value"]),
        cumulative_turnover=float(cumulative_turnover),
        final_date=pd.Timestamp(final_info["date"]),
        final_weights=env.current_weights.copy(),
    )


def _assert_finite_array(values: np.ndarray, name: str) -> None:
    if not np.isfinite(values).all():
        raise RuntimeError(f"{name} must be finite")


def _assert_nonnegative_finite_array(values: np.ndarray, name: str) -> None:
    _assert_finite_array(values, name)
    if (values < 0.0).any():
        raise RuntimeError(f"{name} must be nonnegative")


def _assert_finite_scalar(value: float, name: str) -> None:
    if not np.isfinite(value):
        raise RuntimeError(f"{name} must be finite")


def _assert_positive_finite(value: float, name: str) -> None:
    _assert_finite_scalar(value, name)
    if value <= 0.0:
        raise RuntimeError(f"{name} must be positive")


def _assert_nonnegative_finite(value: float, name: str) -> None:
    _assert_finite_scalar(value, name)
    if value < 0.0:
        raise RuntimeError(f"{name} must be nonnegative")

"""Gymnasium environment for weekly portfolio rebalancing."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from portfolio_rl.config.schemas import EnvConfig
from portfolio_rl.data.feature_store import PortfolioFeatureStore
from portfolio_rl.env.action import action_to_weights
from portfolio_rl.env.costs import (
    calculate_transaction_cost_fraction,
    calculate_turnover,
)
from portfolio_rl.env.drift import simulate_buy_and_hold_period
from portfolio_rl.env.episode_sampler import EpisodeSampler
from portfolio_rl.env.reward import log_growth_reward


class PortfolioEnv(gym.Env[np.ndarray, np.ndarray]):
    """Weekly long-only portfolio allocation environment."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        feature_store: PortfolioFeatureStore,
        env_config: EnvConfig,
        episode_sampler: EpisodeSampler,
        seed: int | None = None,
    ) -> None:
        self.feature_store = feature_store
        self.config = env_config
        self.episode_sampler = episode_sampler
        self._constructor_seed = seed
        self._has_reset = False

        self.n_assets = self.feature_store.n_assets
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.n_assets,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.feature_store.observation_dim,),
            dtype=np.float32,
        )

        self.current_step = 0
        self.current_data_idx = 0
        self.episode_start_idx = 0
        self.current_weights = _equal_weight_vector(self.n_assets)
        self.portfolio_value = 1.0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset to a sampled episode start."""
        del options
        reset_seed = (
            self._constructor_seed if seed is None and not self._has_reset else seed
        )
        super().reset(seed=reset_seed)
        self._has_reset = True

        self.current_step = 0
        self.episode_start_idx = self.episode_sampler.sample_start(
            self.feature_store,
            self.config.episode_length_trading_days,
            self.np_random,
        )
        self.current_data_idx = self.episode_start_idx
        self.current_weights = _equal_weight_vector(self.n_assets)
        self.portfolio_value = 1.0

        observation = self._build_observation()
        info = {
            "date": self.feature_store.date_at(self.current_data_idx),
            "portfolio_value": float(self.portfolio_value),
            "current_weights": self.current_weights.copy(),
        }
        return observation, info

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Rebalance, simulate one holding period, and advance weekly clocks."""
        target_weights = action_to_weights(
            action,
            temperature=self.config.action_temperature,
        )

        try:
            forward_log_returns = self.feature_store.get_forward_log_returns(
                self.current_data_idx,
                horizon=self.config.rebalance_frequency_trading_days,
            )
        except IndexError:
            return self._eof_truncation()

        if len(forward_log_returns) < self.config.rebalance_frequency_trading_days:
            return self._eof_truncation()

        turnover = calculate_turnover(self.current_weights, target_weights)
        cost_fraction = calculate_transaction_cost_fraction(
            turnover,
            self.config.transaction_cost_bps,
        )
        period_gross_return, drifted_weights, daily_portfolio_returns = (
            simulate_buy_and_hold_period(target_weights, forward_log_returns)
        )
        net_gross_return = (1.0 - cost_fraction) * period_gross_return
        reward = log_growth_reward(
            period_gross_return,
            cost_fraction,
            reward_scale=self.config.reward_scale,
            bad_gross_penalty=self.config.terminal_bad_gross_penalty,
        )

        self.portfolio_value *= net_gross_return
        self.current_weights = drifted_weights
        self.current_data_idx += self.config.rebalance_frequency_trading_days
        self.current_step += 1

        terminated = False
        truncated = bool(self.current_step >= self.config.max_episode_steps)
        observation = self._build_observation()
        info = {
            "date": self.feature_store.date_at(self.current_data_idx),
            "turnover": float(turnover),
            "transaction_cost_fraction": float(cost_fraction),
            "period_gross_return": float(period_gross_return),
            "net_gross_return": float(net_gross_return),
            "portfolio_value": float(self.portfolio_value),
            "reward_unscaled_log_growth": (
                float(np.log(net_gross_return))
                if net_gross_return > 0.0
                else float("nan")
            ),
        }

        if self.config.record_arrays_in_info:
            info.update(
                {
                    "target_weights": target_weights.copy(),
                    "current_weights": self.current_weights.copy(),
                    "daily_portfolio_returns": daily_portfolio_returns.copy(),
                }
            )

        return observation, reward, terminated, truncated, info

    def _build_observation(self) -> np.ndarray:
        market_features = self.feature_store.get_market_features(self.current_data_idx)
        observation = np.concatenate(
            [market_features, self.current_weights],
        ).astype(np.float32)
        if observation.shape != self.observation_space.shape:
            raise RuntimeError(
                "observation shape mismatch: "
                f"{observation.shape} != {self.observation_space.shape}"
            )
        if not np.isfinite(observation).all():
            raise RuntimeError("observation values must be finite")
        return observation

    def _eof_truncation(self) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        observation = self._build_observation()
        info = {
            "date": self.feature_store.date_at(self.current_data_idx),
            "portfolio_value": float(self.portfolio_value),
            "eof_truncated": True,
        }
        return observation, 0.0, False, True, info


def _equal_weight_vector(n_assets: int) -> np.ndarray:
    return np.full(n_assets, 1.0 / n_assets, dtype=np.float32)

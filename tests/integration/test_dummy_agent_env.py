from __future__ import annotations

import numpy as np
import pandas as pd

from portfolio_rl.config.schemas import EnvConfig
from portfolio_rl.data.dataset import build_portfolio_dataset
from portfolio_rl.data.feature_store import PortfolioFeatureStore
from portfolio_rl.env.episode_sampler import RandomWindowEpisodeSampler
from portfolio_rl.env.portfolio_env import PortfolioEnv
from portfolio_rl.env.smoke import run_dummy_random_agent_episode
from portfolio_rl.features.feature_spec import FeatureSpec


def test_dummy_random_agent_runs_52_steps_without_nan() -> None:
    env = PortfolioEnv(
        feature_store=PortfolioFeatureStore(
            build_portfolio_dataset(_model_matrix(), _feature_spec()),
            split="train",
        ),
        env_config=_env_config(),
        episode_sampler=RandomWindowEpisodeSampler(),
        seed=42,
    )

    result = run_dummy_random_agent_episode(env, seed=42)

    assert result.steps == 52
    assert np.isfinite(result.final_nav)
    assert result.final_nav > 0.0
    assert np.isfinite(result.cumulative_turnover)
    assert result.cumulative_turnover >= 0.0
    assert np.isfinite(result.final_weights).all()
    assert np.all(result.final_weights >= 0.0)
    np.testing.assert_allclose(result.final_weights.sum(), 1.0)
    assert result.final_date == pd.Timestamp("2024-12-31")


def _env_config() -> EnvConfig:
    return EnvConfig(
        rebalance_frequency_trading_days=5,
        episode_length_trading_days=260,
        max_episode_steps=52,
        action_transform="softmax",
        action_temperature=5.0,
        initial_weights="equal_weight",
        transaction_cost_bps=10.0,
        reward_type="log_growth",
        reward_scale=100.0,
        terminal_bad_gross_penalty=-100.0,
        record_arrays_in_info=False,
    )


def _feature_spec() -> FeatureSpec:
    return FeatureSpec(
        feature_version="v1",
        asset_order=["SPY", "QQQ"],
        per_asset_features=["ret_1d"],
        global_features=["vix_z_21d"],
        current_weight_features=["weight_spy", "weight_qqq"],
        observation_dim=5,
        created_at="2026-05-07T00:00:00+00:00",
    )


def _model_matrix() -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=263)
    rows = []
    for index, date in enumerate(dates):
        rows.append(
            {
                "date": date,
                "split": "train" if index < 261 else "validation",
                "feature_version": "v1",
                "obs_000": float(np.sin(index / 10.0)),
                "obs_001": float(np.cos(index / 10.0)),
                "obs_002": float(index / 100.0),
                "obs_003": 0.5,
                "obs_004": 0.5,
                "return_spy_1d": float(0.0002 * np.sin(index / 7.0)),
                "return_qqq_1d": float(0.0003 * np.cos(index / 11.0)),
            }
        )
    return pd.DataFrame(rows)

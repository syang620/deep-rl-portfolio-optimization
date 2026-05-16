from __future__ import annotations

from math import log

import numpy as np
import pandas as pd
import pytest
from gymnasium.utils.env_checker import check_env

from portfolio_rl.config.schemas import EnvConfig
from portfolio_rl.data.dataset import build_portfolio_dataset
from portfolio_rl.data.feature_store import PortfolioFeatureStore
from portfolio_rl.env.action import action_to_weights
from portfolio_rl.env.episode_sampler import FixedStartEpisodeSampler
from portfolio_rl.env.portfolio_env import PortfolioEnv
from portfolio_rl.features.feature_spec import FeatureSpec


def test_reset_returns_valid_observation_and_equal_weights() -> None:
    env = _env()

    observation, info = env.reset(seed=42)

    assert observation.dtype == np.float32
    assert observation.shape == env.observation_space.shape
    np.testing.assert_allclose(observation[-2:], [0.5, 0.5])
    np.testing.assert_allclose(info["current_weights"], [0.5, 0.5])
    assert info["portfolio_value"] == 1.0
    assert info["date"] == pd.Timestamp("2024-01-02")


def test_reset_observation_uses_dynamic_weights_not_static_matrix_weights() -> None:
    env = _env(static_weight_slice=(0.9, 0.1))

    observation, _ = env.reset(seed=42)

    np.testing.assert_allclose(observation[-2:], [0.5, 0.5])


def test_step_advances_two_clocks() -> None:
    env = _env(n_train_rows=20, env_config=_env_config(episode_length=10))
    env.reset(seed=42)

    _, _, _, _, info = env.step(np.zeros(2, dtype=np.float32))

    assert env.current_data_idx == 5
    assert env.current_step == 1
    assert info["date"] == pd.Timestamp("2024-01-09")


def test_step_reward_uses_future_returns_only() -> None:
    env = _env(
        n_train_rows=8,
        env_config=_env_config(episode_length=5, transaction_cost_bps=0.0),
        return_overrides={0: (log(2.0), log(2.0))},
    )
    env.reset(seed=42)

    _, reward, _, truncated, info = env.step(np.zeros(2, dtype=np.float32))

    assert reward == pytest.approx(0.0)
    assert info["period_gross_return"] == pytest.approx(1.0)
    assert truncated is True


def test_step_observation_appends_drifted_current_weights() -> None:
    env = _env(
        n_train_rows=12,
        env_config=_env_config(episode_length=5, transaction_cost_bps=0.0),
        return_overrides={1: (log(1.10), log(1.0))},
    )
    env.reset(seed=42)

    observation, _, _, _, _ = env.step(np.zeros(2, dtype=np.float32))

    np.testing.assert_allclose(observation[-2:], env.current_weights)
    assert observation[-2] > 0.5
    assert observation[-1] < 0.5


def test_turnover_uses_drifted_current_weights() -> None:
    env = _env(
        n_train_rows=20,
        env_config=_env_config(episode_length=10, transaction_cost_bps=0.0),
        return_overrides={1: (log(1.10), log(1.0))},
    )
    env.reset(seed=42)
    env.step(np.zeros(2, dtype=np.float32))
    drifted_weights = env.current_weights.copy()

    _, _, _, _, info = env.step(np.ones(2, dtype=np.float32))

    expected_turnover = np.abs(
        action_to_weights(np.ones(2, dtype=np.float32), temperature=5.0)
        - drifted_weights
    ).sum()
    assert info["turnover"] == pytest.approx(float(expected_turnover))


def test_record_arrays_in_info_false_keeps_step_info_lightweight() -> None:
    env = _env(env_config=_env_config(record_arrays_in_info=False))
    env.reset(seed=42)

    _, _, _, _, info = env.step(np.zeros(2, dtype=np.float32))

    assert "target_weights" not in info
    assert "current_weights" not in info
    assert "daily_portfolio_returns" not in info


def test_record_arrays_in_info_true_includes_diagnostics() -> None:
    env = _env(env_config=_env_config(record_arrays_in_info=True))
    env.reset(seed=42)

    _, _, _, _, info = env.step(np.zeros(2, dtype=np.float32))

    assert info["target_weights"].shape == (2,)
    assert info["current_weights"].shape == (2,)
    assert info["daily_portfolio_returns"].shape == (5,)


def test_eof_tail_truncates_without_error() -> None:
    env = _env(
        n_train_rows=7,
        env_config=_env_config(episode_length=10),
        episode_sampler=_NoValidationSampler(),
    )
    env.reset(seed=42)
    env.step(np.zeros(2, dtype=np.float32))

    observation, reward, terminated, truncated, info = env.step(
        np.zeros(2, dtype=np.float32),
    )

    assert observation.shape == env.observation_space.shape
    assert reward == 0.0
    assert terminated is False
    assert truncated is True
    assert info["eof_truncated"] is True


def test_portfolio_env_passes_gymnasium_check_env() -> None:
    env = _env(n_train_rows=20, env_config=_env_config(episode_length=10))

    check_env(env, skip_render_check=True)


class _NoValidationSampler:
    def sample_start(
        self,
        store: PortfolioFeatureStore,
        episode_length_trading_days: int,
        rng: np.random.Generator,
    ) -> int:
        del store, episode_length_trading_days, rng
        return 0


def _env(
    *,
    n_train_rows: int = 12,
    env_config: EnvConfig | None = None,
    episode_sampler: FixedStartEpisodeSampler | _NoValidationSampler | None = None,
    static_weight_slice: tuple[float, float] = (0.5, 0.5),
    return_overrides: dict[int, tuple[float, float]] | None = None,
) -> PortfolioEnv:
    return PortfolioEnv(
        feature_store=_feature_store(
            n_train_rows=n_train_rows,
            static_weight_slice=static_weight_slice,
            return_overrides=return_overrides or {},
        ),
        env_config=env_config or _env_config(),
        episode_sampler=episode_sampler or FixedStartEpisodeSampler(),
    )


def _env_config(
    *,
    episode_length: int = 5,
    transaction_cost_bps: float = 10.0,
    record_arrays_in_info: bool = False,
) -> EnvConfig:
    return EnvConfig(
        rebalance_frequency_trading_days=5,
        episode_length_trading_days=episode_length,
        max_episode_steps=episode_length // 5,
        action_transform="softmax",
        action_temperature=5.0,
        initial_weights="equal_weight",
        transaction_cost_bps=transaction_cost_bps,
        reward_type="log_growth",
        reward_scale=100.0,
        terminal_bad_gross_penalty=-100.0,
        record_arrays_in_info=record_arrays_in_info,
    )


def _feature_store(
    *,
    n_train_rows: int,
    static_weight_slice: tuple[float, float],
    return_overrides: dict[int, tuple[float, float]],
) -> PortfolioFeatureStore:
    return PortfolioFeatureStore(
        build_portfolio_dataset(
            _model_matrix(n_train_rows, static_weight_slice, return_overrides),
            _feature_spec(),
        ),
        split="train",
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


def _model_matrix(
    n_train_rows: int,
    static_weight_slice: tuple[float, float],
    return_overrides: dict[int, tuple[float, float]],
) -> pd.DataFrame:
    dates = pd.date_range("2024-01-02", periods=n_train_rows + 2, freq="B")
    rows = []
    for index, date in enumerate(dates):
        spy_return, qqq_return = return_overrides.get(index, (0.0, 0.0))
        rows.append(
            {
                "date": date,
                "split": "train" if index < n_train_rows else "validation",
                "feature_version": "v1",
                "obs_000": 10.0 + index,
                "obs_001": 20.0 + index,
                "obs_002": 30.0 + index,
                "obs_003": static_weight_slice[0],
                "obs_004": static_weight_slice[1],
                "return_spy_1d": spy_return,
                "return_qqq_1d": qqq_return,
            }
        )
    return pd.DataFrame(rows)

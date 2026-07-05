from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("stable_baselines3")

from portfolio_rl.config.loader import load_env_config
from portfolio_rl.data.dataset import load_portfolio_dataset
from portfolio_rl.data.feature_store import PortfolioFeatureStore
from portfolio_rl.evaluation.backtest import run_weight_policy_backtest
from portfolio_rl.policies.sb3_policy import load_sb3_weight_policy
from portfolio_rl.training.train_ppo import run_ppo_training


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_saved_ppo_policy_runs_one_step_validation_backtest(tmp_path: Path) -> None:
    env_config = load_env_config(REPO_ROOT / "configs/env.yaml")
    model_path = run_ppo_training(
        root=REPO_ROOT,
        data_config_path=REPO_ROOT / "configs/data.yaml",
        env_config_path=REPO_ROOT / "configs/env.yaml",
        train_config_path=REPO_ROOT / "configs/train_ppo.yaml",
        total_timesteps_override=1000,
        output_dir_override=tmp_path,
    )
    policy = load_sb3_weight_policy(
        model_path,
        action_temperature=env_config.action_temperature,
    )
    dataset = load_portfolio_dataset(REPO_ROOT)
    validation_store = PortfolioFeatureStore(dataset, split="validation")

    result = run_weight_policy_backtest(
        feature_store=validation_store,
        policy=policy,
        strategy="ppo_smoke",
        rebalance_frequency_trading_days=env_config.rebalance_frequency_trading_days,
        transaction_cost_bps=env_config.transaction_cost_bps,
        max_steps=1,
    )

    assert not result.nav.empty
    assert not result.weights_target.empty
    assert not result.trades.empty
    assert not result.costs.empty

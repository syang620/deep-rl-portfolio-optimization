from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("stable_baselines3")

from stable_baselines3 import PPO

from portfolio_rl.training.train_ppo import run_ppo_training


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_ppo_smoke_training_saves_loadable_model(tmp_path: Path) -> None:
    model_path = run_ppo_training(
        root=REPO_ROOT,
        data_config_path=REPO_ROOT / "configs/data.yaml",
        env_config_path=REPO_ROOT / "configs/env.yaml",
        train_config_path=REPO_ROOT / "configs/train_ppo.yaml",
        total_timesteps_override=1000,
        output_dir_override=tmp_path,
    )

    assert model_path == tmp_path / "model.zip"
    assert model_path.exists()
    assert PPO.load(model_path) is not None


def test_ppo_smoke_training_writes_experiment_artifacts(tmp_path: Path) -> None:
    experiment_dir = tmp_path / "experiment"

    model_path = run_ppo_training(
        root=REPO_ROOT,
        data_config_path=REPO_ROOT / "configs/data.yaml",
        env_config_path=REPO_ROOT / "configs/env.yaml",
        train_config_path=REPO_ROOT / "configs/train_ppo.yaml",
        total_timesteps_override=1000,
        output_dir_override=experiment_dir,
    )

    assert model_path == experiment_dir / "model.zip"
    for artifact_name in [
        "model.zip",
        "config.yaml",
        "env.yaml",
        "train_ppo.yaml",
        "feature_spec_v1.json",
        "data_quality_report_v1.json",
        "metrics_validation.json",
        "validation_nav.parquet",
        "validation_weights.parquet",
        "validation_trades.parquet",
        "validation_costs.parquet",
        "manifest.json",
    ]:
        assert (experiment_dir / artifact_name).is_file()

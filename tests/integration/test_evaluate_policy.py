from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("stable_baselines3")

from portfolio_rl.training.train_ppo import run_ppo_training
from scripts.evaluate_policy import run_policy_evaluation


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_ARTIFACTS = [
    "metrics.json",
    "nav.parquet",
    "costs.parquet",
    "weights_target.parquet",
    "report.md",
]


def test_evaluate_policy_writes_expected_artifacts(tmp_path: Path) -> None:
    model_path = run_ppo_training(
        root=REPO_ROOT,
        data_config_path=REPO_ROOT / "configs/data.yaml",
        env_config_path=REPO_ROOT / "configs/env.yaml",
        train_config_path=REPO_ROOT / "configs/train_ppo.yaml",
        total_timesteps_override=1000,
        output_dir_override=tmp_path / "model",
    )

    output_dir = run_policy_evaluation(
        root=REPO_ROOT,
        env_config_path=REPO_ROOT / "configs/env.yaml",
        model_path=model_path,
        split="validation",
        strategy="ppo_smoke",
        output_dir=tmp_path / "eval",
        max_steps=1,
    )

    assert output_dir == tmp_path / "eval"
    for artifact_name in EXPECTED_ARTIFACTS:
        assert (output_dir / artifact_name).is_file()

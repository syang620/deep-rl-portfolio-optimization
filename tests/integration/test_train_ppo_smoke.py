from __future__ import annotations

import sys
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
        "best_model.zip",
        "best_metrics_validation.json",
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


def test_ppo_smoke_training_logs_to_wandb_when_enabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_wandb = _FakeWandbModule()
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    experiment_dir = tmp_path / "experiment"

    model_path = run_ppo_training(
        root=REPO_ROOT,
        data_config_path=REPO_ROOT / "configs/data.yaml",
        env_config_path=REPO_ROOT / "configs/env.yaml",
        train_config_path=REPO_ROOT / "configs/train_ppo_wandb.yaml",
        total_timesteps_override=1000,
        output_dir_override=experiment_dir,
    )

    assert model_path == experiment_dir / "model.zip"
    assert fake_wandb.init_kwargs["project"] == "rl-dynamic-portfolio-allocation"
    assert fake_wandb.init_kwargs["name"] == "experiment"
    assert fake_wandb.run.logged
    assert fake_wandb.run.logged_artifacts
    assert fake_wandb.run.finished is True
    artifact_names = {
        name
        for artifact in fake_wandb.run.logged_artifacts
        for _path, name in artifact.files
    }
    assert "model.zip" in artifact_names
    assert "metrics_validation.json" in artifact_names
    assert "manifest.json" in artifact_names


class _FakeWandbModule:
    def __init__(self) -> None:
        self.run = _FakeWandbRun()
        self.init_kwargs = {}

    def init(self, **kwargs):
        self.init_kwargs = kwargs
        return self.run

    def Artifact(self, name: str, type: str):
        return _FakeWandbArtifact(name=name, type=type)


class _FakeWandbRun:
    def __init__(self) -> None:
        self.logged = []
        self.logged_artifacts = []
        self.finished = False

    def log(self, payload, step: int) -> None:
        self.logged.append((step, payload))

    def log_artifact(self, artifact) -> None:
        self.logged_artifacts.append(artifact)

    def finish(self) -> None:
        self.finished = True


class _FakeWandbArtifact:
    def __init__(self, *, name: str, type: str) -> None:
        self.name = name
        self.type = type
        self.files = []

    def add_file(self, path: str, name: str) -> None:
        self.files.append((path, name))

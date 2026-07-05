from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from portfolio_rl.config.loader import load_train_ppo_config


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"


def test_valid_train_ppo_config_loads_successfully() -> None:
    config = load_train_ppo_config(CONFIG_DIR / "train_ppo.yaml")

    assert config.algorithm == "PPO"
    assert config.policy == "MlpPolicy"
    assert config.total_timesteps == 500000
    assert config.seed == 42
    assert config.ppo.n_steps == 2080
    assert config.ppo.batch_size == 260
    assert config.network.pi == [256, 256]
    assert config.network.vf == [256, 256]
    assert config.evaluation.metric_for_best_model == "sharpe_ratio"
    assert config.checkpoints.output_dir == Path("artifacts/experiments")
    assert config.wandb.enabled is False


def test_train_ppo_unknown_top_level_field_fails(tmp_path: Path) -> None:
    config_path = tmp_path / "train_ppo.yaml"
    config_path.write_text(
        _train_ppo_config_yaml("unexpected_key: true"),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        load_train_ppo_config(config_path)


def test_train_ppo_invalid_algorithm_fails(tmp_path: Path) -> None:
    config_path = tmp_path / "train_ppo.yaml"
    config_path.write_text(
        _train_ppo_config_yaml("algorithm: SAC"),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="algorithm must be PPO"):
        load_train_ppo_config(config_path)


def test_train_ppo_invalid_policy_fails(tmp_path: Path) -> None:
    config_path = tmp_path / "train_ppo.yaml"
    config_path.write_text(
        _train_ppo_config_yaml("policy: CnnPolicy"),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="policy must be MlpPolicy"):
        load_train_ppo_config(config_path)


def test_train_ppo_n_steps_must_align_to_episode_length(tmp_path: Path) -> None:
    config_path = tmp_path / "train_ppo.yaml"
    config_path.write_text(
        _train_ppo_config_yaml(
            """
ppo:
  learning_rate: 0.0003
  gamma: 0.99
  gae_lambda: 0.95
  n_steps: 500
  batch_size: 50
  n_epochs: 10
  clip_range: 0.2
  ent_coef: 0.01
  vf_coef: 0.5
  max_grad_norm: 0.5
"""
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="n_steps must be divisible by 52"):
        load_train_ppo_config(config_path)


def test_train_ppo_n_steps_must_align_to_batch_size(tmp_path: Path) -> None:
    config_path = tmp_path / "train_ppo.yaml"
    config_path.write_text(
        _train_ppo_config_yaml(
            """
ppo:
  learning_rate: 0.0003
  gamma: 0.99
  gae_lambda: 0.95
  n_steps: 2080
  batch_size: 300
  n_epochs: 10
  clip_range: 0.2
  ent_coef: 0.01
  vf_coef: 0.5
  max_grad_norm: 0.5
"""
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="n_steps must be divisible by batch_size"):
        load_train_ppo_config(config_path)


def test_train_ppo_empty_network_layers_fail(tmp_path: Path) -> None:
    config_path = tmp_path / "train_ppo.yaml"
    config_path.write_text(
        _train_ppo_config_yaml(
            """
network:
  pi: []
  vf: [256, 256]
"""
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="List should have at least 1 item"):
        load_train_ppo_config(config_path)


def test_train_ppo_nonpositive_network_layer_size_fails(tmp_path: Path) -> None:
    config_path = tmp_path / "train_ppo.yaml"
    config_path.write_text(
        _train_ppo_config_yaml(
            """
network:
  pi: [256, 0]
  vf: [256, 256]
"""
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="network layer sizes"):
        load_train_ppo_config(config_path)


def test_train_ppo_invalid_best_model_metric_fails(tmp_path: Path) -> None:
    config_path = tmp_path / "train_ppo.yaml"
    config_path.write_text(
        _train_ppo_config_yaml(
            """
evaluation:
  eval_freq_timesteps: 25000
  deterministic: true
  metric_for_best_model: win_rate
"""
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="metric_for_best_model"):
        load_train_ppo_config(config_path)


def test_train_ppo_negative_seed_fails(tmp_path: Path) -> None:
    config_path = tmp_path / "train_ppo.yaml"
    config_path.write_text(
        _train_ppo_config_yaml("seed: -1"),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="greater than or equal to 0"):
        load_train_ppo_config(config_path)


def _train_ppo_config_yaml(override_yaml: str) -> str:
    config = {
        "algorithm": "PPO",
        "policy": "MlpPolicy",
        "total_timesteps": 500000,
        "seed": 42,
        "ppo": {
            "learning_rate": 0.0003,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "n_steps": 2080,
            "batch_size": 260,
            "n_epochs": 10,
            "clip_range": 0.2,
            "ent_coef": 0.01,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
        },
        "network": {
            "pi": [256, 256],
            "vf": [256, 256],
        },
        "evaluation": {
            "eval_freq_timesteps": 25000,
            "deterministic": True,
            "metric_for_best_model": "sharpe_ratio",
        },
        "checkpoints": {
            "save_freq_timesteps": 50000,
            "output_dir": "artifacts/experiments",
        },
        "wandb": {
            "enabled": False,
            "project": "rl-dynamic-portfolio-allocation",
            "group": "phase2-ppo",
            "tags": ["ppo", "weekly-rebalance", "v1-features"],
        },
    }
    _deep_update(config, _parse_override(override_yaml))
    return _to_yaml(config)


def _parse_override(override_yaml: str) -> dict[str, object]:
    import yaml

    parsed = yaml.safe_load(override_yaml)
    if parsed is None:
        return {}
    if not isinstance(parsed, dict):
        raise ValueError("override_yaml must parse to a mapping")
    return parsed


def _deep_update(
    target: dict[str, object],
    updates: dict[str, object],
) -> None:
    for key, value in updates.items():
        existing_value = target.get(key)
        if isinstance(existing_value, dict) and isinstance(value, dict):
            _deep_update(existing_value, value)
        else:
            target[key] = value


def _to_yaml(config: dict[str, object]) -> str:
    import yaml

    return yaml.safe_dump(config, sort_keys=False)

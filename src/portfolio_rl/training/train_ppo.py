"""Stable-Baselines3 PPO training harness."""

from __future__ import annotations

import hashlib
import importlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from portfolio_rl.config.loader import (
    load_data_config,
    load_env_config,
    load_train_ppo_config,
)
from portfolio_rl.data.dataset import load_portfolio_dataset
from portfolio_rl.data.feature_store import PortfolioFeatureStore
from portfolio_rl.env.episode_sampler import RandomWindowEpisodeSampler
from portfolio_rl.env.portfolio_env import PortfolioEnv
from portfolio_rl.evaluation.backtest import run_weight_policy_backtest
from portfolio_rl.policies.sb3_policy import load_sb3_weight_policy
from portfolio_rl.training.callbacks import ValidationCheckpointCallback


DEFAULT_FEATURE_SPEC_PATH = Path("artifacts/feature_specs/feature_spec_v1.json")
DEFAULT_DATA_QUALITY_REPORT_PATH = Path("artifacts/reports/data_quality_report_v1.json")


def run_ppo_training(
    *,
    root: str | Path = ".",
    data_config_path: str | Path = "configs/data.yaml",
    env_config_path: str | Path = "configs/env.yaml",
    train_config_path: str | Path = "configs/train_ppo.yaml",
    total_timesteps_override: int | None = None,
    output_dir_override: str | Path | None = None,
    run_id: str | None = None,
) -> Path:
    """Train a PPO policy, write experiment artifacts, and return model path."""
    root_path = Path(root)
    resolved_data_config_path = _resolve_path(root_path, data_config_path)
    resolved_env_config_path = _resolve_path(root_path, env_config_path)
    resolved_train_config_path = _resolve_path(root_path, train_config_path)
    resolved_feature_spec_path = root_path / DEFAULT_FEATURE_SPEC_PATH
    resolved_data_quality_report_path = root_path / DEFAULT_DATA_QUALITY_REPORT_PATH

    load_data_config(resolved_data_config_path)
    env_config = load_env_config(resolved_env_config_path)
    train_config = load_train_ppo_config(resolved_train_config_path)

    total_timesteps = (
        train_config.total_timesteps
        if total_timesteps_override is None
        else total_timesteps_override
    )
    if total_timesteps <= 0:
        raise ValueError("total_timesteps must be positive")

    experiment_dir = _resolve_experiment_dir(
        root_path=root_path,
        configured_output_dir=train_config.checkpoints.output_dir,
        output_dir_override=output_dir_override,
        run_id=run_id,
    )
    experiment_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_portfolio_dataset(root_path)
    train_store = PortfolioFeatureStore(dataset, split="train")
    validation_store = PortfolioFeatureStore(dataset, split="validation")
    wandb_module, wandb_run = _start_wandb_run(
        train_config=train_config,
        env_config=env_config,
        total_timesteps=total_timesteps,
        feature_version=dataset.feature_version,
        run_id=experiment_dir.name,
    )

    def make_env() -> PortfolioEnv:
        return PortfolioEnv(
            feature_store=train_store,
            env_config=env_config,
            episode_sampler=RandomWindowEpisodeSampler(),
            seed=train_config.seed,
        )

    vec_env = DummyVecEnv([make_env])
    try:
        model = PPO(
            policy=train_config.policy,
            env=vec_env,
            learning_rate=train_config.ppo.learning_rate,
            gamma=train_config.ppo.gamma,
            gae_lambda=train_config.ppo.gae_lambda,
            n_steps=train_config.ppo.n_steps,
            batch_size=train_config.ppo.batch_size,
            n_epochs=train_config.ppo.n_epochs,
            clip_range=train_config.ppo.clip_range,
            ent_coef=train_config.ppo.ent_coef,
            vf_coef=train_config.ppo.vf_coef,
            max_grad_norm=train_config.ppo.max_grad_norm,
            seed=train_config.seed,
            policy_kwargs={
                "net_arch": {
                    "pi": train_config.network.pi,
                    "vf": train_config.network.vf,
                },
            },
            verbose=0,
        )
        validation_callback = ValidationCheckpointCallback(
            validation_store=validation_store,
            action_temperature=env_config.action_temperature,
            rebalance_frequency_trading_days=(
                env_config.rebalance_frequency_trading_days
            ),
            transaction_cost_bps=env_config.transaction_cost_bps,
            eval_freq_timesteps=train_config.evaluation.eval_freq_timesteps,
            metric_for_best_model=train_config.evaluation.metric_for_best_model,
            output_dir=experiment_dir,
            validation_metrics_callback=(
                (lambda step, metrics: _log_wandb_metrics(wandb_run, step, metrics))
                if wandb_run is not None
                else None
            ),
        )
        model.learn(total_timesteps=total_timesteps, callback=validation_callback)

        model_path = experiment_dir / "model.zip"
        model.save(model_path)
    finally:
        vec_env.close()

    validation_policy = load_sb3_weight_policy(
        model_path,
        action_temperature=env_config.action_temperature,
    )
    validation_result = run_weight_policy_backtest(
        feature_store=validation_store,
        policy=validation_policy,
        strategy="ppo",
        rebalance_frequency_trading_days=env_config.rebalance_frequency_trading_days,
        transaction_cost_bps=env_config.transaction_cost_bps,
    )
    _write_validation_artifacts(validation_result, experiment_dir)
    _copy_experiment_inputs(
        experiment_dir=experiment_dir,
        data_config_path=resolved_data_config_path,
        env_config_path=resolved_env_config_path,
        train_config_path=resolved_train_config_path,
        feature_spec_path=resolved_feature_spec_path,
        data_quality_report_path=resolved_data_quality_report_path,
    )
    _write_manifest(
        experiment_dir=experiment_dir,
        run_id=experiment_dir.name,
        total_timesteps=total_timesteps,
        seed=train_config.seed,
        feature_version=dataset.feature_version,
        data_config_path=resolved_data_config_path,
        env_config_path=resolved_env_config_path,
        train_config_path=resolved_train_config_path,
        feature_spec_path=resolved_feature_spec_path,
        data_quality_report_path=resolved_data_quality_report_path,
    )
    if wandb_run is not None:
        _log_wandb_metrics(wandb_run, total_timesteps, validation_result.metrics)
        _log_wandb_artifacts(wandb_module, wandb_run, experiment_dir)
        wandb_run.finish()
    return model_path


def _resolve_path(root: Path, path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return root / candidate


def _resolve_experiment_dir(
    *,
    root_path: Path,
    configured_output_dir: Path,
    output_dir_override: str | Path | None,
    run_id: str | None,
) -> Path:
    if output_dir_override is not None:
        output_dir = Path(output_dir_override)
        return output_dir if output_dir.is_absolute() else root_path / output_dir

    resolved_run_id = run_id or _default_run_id()
    output_dir = Path(configured_output_dir)
    if not output_dir.is_absolute():
        output_dir = root_path / output_dir
    return output_dir / resolved_run_id


def _default_run_id() -> str:
    return "ppo_" + datetime.now(UTC).strftime("%Y%m%d_%H%M%S")


def _start_wandb_run(
    *,
    train_config,
    env_config,
    total_timesteps: int,
    feature_version: str,
    run_id: str,
) -> tuple[Any | None, Any | None]:
    if not train_config.wandb.enabled:
        return None, None

    try:
        wandb_module = importlib.import_module("wandb")
    except ImportError as exc:
        raise RuntimeError(
            "W&B tracking is enabled but wandb is not installed. "
            "Install project dependencies or set wandb.enabled=false."
        ) from exc

    config = {
        "algorithm": train_config.algorithm,
        "policy": train_config.policy,
        "seed": train_config.seed,
        "total_timesteps": total_timesteps,
        "feature_version": feature_version,
        "env": env_config.model_dump(mode="json"),
        "ppo": train_config.ppo.model_dump(mode="json"),
        "network": train_config.network.model_dump(mode="json"),
        "evaluation": train_config.evaluation.model_dump(mode="json"),
    }
    run = wandb_module.init(
        project=train_config.wandb.project,
        group=train_config.wandb.group,
        tags=train_config.wandb.tags,
        name=run_id,
        config=config,
    )
    return wandb_module, run


def _log_wandb_metrics(
    wandb_run,
    step: int,
    metrics: dict[str, float | None],
) -> None:
    payload = {
        f"validation/{key}": value
        for key, value in metrics.items()
        if value is not None
    }
    if payload:
        wandb_run.log(payload, step=step)


def _log_wandb_artifacts(wandb_module, wandb_run, experiment_dir: Path) -> None:
    artifact = wandb_module.Artifact(
        name=f"{experiment_dir.name}-artifacts",
        type="model",
    )
    for relative_path in [
        "model.zip",
        "best_model.zip",
        "manifest.json",
        "config.yaml",
        "env.yaml",
        "train_ppo.yaml",
        "feature_spec_v1.json",
        "data_quality_report_v1.json",
        "metrics_validation.json",
        "best_metrics_validation.json",
    ]:
        path = experiment_dir / relative_path
        if path.exists():
            artifact.add_file(str(path), name=relative_path)
    wandb_run.log_artifact(artifact)


def _write_validation_artifacts(
    validation_result,
    experiment_dir: Path,
) -> None:
    validation_result.nav.to_parquet(
        experiment_dir / "validation_nav.parquet",
        index=False,
    )
    validation_result.weights_target.to_parquet(
        experiment_dir / "validation_weights.parquet",
        index=False,
    )
    validation_result.trades.to_parquet(
        experiment_dir / "validation_trades.parquet",
        index=False,
    )
    validation_result.costs.to_parquet(
        experiment_dir / "validation_costs.parquet",
        index=False,
    )
    (experiment_dir / "metrics_validation.json").write_text(
        json.dumps(validation_result.metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _copy_experiment_inputs(
    *,
    experiment_dir: Path,
    data_config_path: Path,
    env_config_path: Path,
    train_config_path: Path,
    feature_spec_path: Path,
    data_quality_report_path: Path,
) -> None:
    copies = {
        data_config_path: "config.yaml",
        env_config_path: "env.yaml",
        train_config_path: "train_ppo.yaml",
        feature_spec_path: "feature_spec_v1.json",
    }
    if data_quality_report_path.exists():
        copies[data_quality_report_path] = "data_quality_report_v1.json"

    for source_path, target_name in copies.items():
        shutil.copy2(source_path, experiment_dir / target_name)


def _write_manifest(
    *,
    experiment_dir: Path,
    run_id: str,
    total_timesteps: int,
    seed: int,
    feature_version: str,
    data_config_path: Path,
    env_config_path: Path,
    train_config_path: Path,
    feature_spec_path: Path,
    data_quality_report_path: Path,
) -> None:
    manifest = {
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "algorithm": "PPO",
        "feature_version": feature_version,
        "seed": seed,
        "total_timesteps": total_timesteps,
        "data_config_hash": _sha256_file(data_config_path),
        "env_config_hash": _sha256_file(env_config_path),
        "train_config_hash": _sha256_file(train_config_path),
        "feature_spec_hash": _sha256_file(feature_spec_path),
        "data_quality_report_hash": (
            _sha256_file(data_quality_report_path)
            if data_quality_report_path.exists()
            else None
        ),
    }
    (experiment_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

"""Leakage-safe PPO selection for nested walk-forward folds."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from portfolio_rl.config.loader import load_env_config, load_train_ppo_config, load_yaml
from portfolio_rl.data.feature_store import PortfolioFeatureStore
from portfolio_rl.data.walk_forward import load_training_selection_dataset
from portfolio_rl.env.episode_sampler import RandomWindowEpisodeSampler
from portfolio_rl.env.portfolio_env import PortfolioEnv
from portfolio_rl.evaluation.backtest import (
    run_weight_policy_backtest,
    write_backtest_artifacts,
)
from portfolio_rl.policies.sb3_policy import load_sb3_weight_policy
from portfolio_rl.training.callbacks import ValidationCheckpointCallback

EXPECTED_FOLDS = ("WF1", "WF2", "WF3", "WF4")
EXPECTED_SEEDS = (7, 42, 101, 202, 999)


@dataclass(frozen=True)
class WalkForwardCampaignConfig:
    """Frozen PR17 campaign contract."""

    campaign_id: str
    config_path: Path
    config_sha256: str
    data_root: Path
    data_manifest: Path
    data_manifest_sha256: str
    env_config: Path
    env_config_sha256: str
    train_config: Path
    train_config_sha256: str
    output_root: Path
    folds: tuple[str, ...]
    seeds: tuple[int, ...]
    total_timesteps: int
    eval_freq_timesteps: int
    pilot_fold: str
    pilot_seed: int
    pilot_timesteps: int
    rebalance_days: int
    transaction_cost_bps: float
    inverse_vol_lookback: int
    momentum_lookback: int
    momentum_top_k: int
    alphas: tuple[float, ...]

    @property
    def campaign_root(self) -> Path:
        return self.output_root / self.campaign_id


@dataclass(frozen=True)
class SelectionResult:
    """One immutable fold/seed selection result."""

    fold_id: str
    seed: int
    run_id: str
    output_dir: Path
    freeze_path: Path
    selected_model_path: Path


def load_walk_forward_campaign_config(
    path: str | Path,
    *,
    root: str | Path = ".",
) -> WalkForwardCampaignConfig:
    """Load and verify the exact PR17 campaign declaration."""
    root_path = Path(root).resolve()
    config_path = _resolve(root_path, path)
    raw = load_yaml(config_path)
    if int(_required(raw, "schema_version")) != 1:
        raise ValueError("walk-forward campaign schema_version must be 1")
    data = _mapping(raw, "walk_forward_data")
    env = _mapping(raw, "env_config")
    train = _mapping(raw, "train_config")
    selection = _mapping(raw, "selection")
    pilot = _mapping(raw, "pilot")
    execution = _mapping(raw, "execution")
    reporting = _mapping(raw, "reporting")
    folds = tuple(str(value) for value in _list(raw, "folds"))
    seeds = tuple(int(value) for value in _list(raw, "seeds"))
    alphas = tuple(
        float(value) for value in _list(execution, "partial_rebalance_alphas")
    )
    if folds != EXPECTED_FOLDS:
        raise ValueError(f"folds must equal {list(EXPECTED_FOLDS)}")
    if seeds != EXPECTED_SEEDS:
        raise ValueError(f"seeds must equal {list(EXPECTED_SEEDS)}")
    if alphas != (0.25, 0.5, 0.75, 1.0):
        raise ValueError("partial-rebalance alphas must be 0.25, 0.5, 0.75, 1.0")
    if (
        _text(selection, "metric") != "sharpe_ratio"
        or _text(selection, "direction") != "maximize"
        or _text(selection, "tie_break") != "earliest_step"
        or _required(selection, "include_final_endpoint") is not True
    ):
        raise ValueError("selection must maximize Sharpe with earliest-step ties")
    if _text(execution, "initial_portfolio") != "equal_weight":
        raise ValueError("primary walk-forward initialization must be equal weight")
    if _text(reporting, "primary_reference") != "equal_weight_weekly":
        raise ValueError("primary reference must be weekly equal weight")
    if _required(reporting, "select_candidate") is not False:
        raise ValueError("PR17 must not select a candidate")

    config = WalkForwardCampaignConfig(
        campaign_id=_text(raw, "campaign_id"),
        config_path=config_path,
        config_sha256=_sha256(config_path),
        data_root=_resolve(root_path, _text(data, "root")),
        data_manifest=_resolve(root_path, _text(data, "manifest")),
        data_manifest_sha256=_text(data, "manifest_sha256"),
        env_config=_resolve(root_path, _text(env, "path")),
        env_config_sha256=_text(env, "sha256"),
        train_config=_resolve(root_path, _text(train, "path")),
        train_config_sha256=_text(train, "sha256"),
        output_root=_resolve(root_path, _text(raw, "output_root")),
        folds=folds,
        seeds=seeds,
        total_timesteps=int(_required(raw, "total_timesteps")),
        eval_freq_timesteps=int(
            _required(selection, "evaluation_frequency_timesteps")
        ),
        pilot_fold=_text(pilot, "fold"),
        pilot_seed=int(_required(pilot, "seed")),
        pilot_timesteps=int(_required(pilot, "total_timesteps")),
        rebalance_days=int(
            _required(execution, "rebalance_frequency_trading_days")
        ),
        transaction_cost_bps=float(
            _required(execution, "transaction_cost_bps")
        ),
        inverse_vol_lookback=int(
            _required(execution, "inverse_volatility_lookback_trading_days")
        ),
        momentum_lookback=int(
            _required(execution, "momentum_lookback_trading_days")
        ),
        momentum_top_k=int(_required(execution, "momentum_top_k")),
        alphas=alphas,
    )
    _validate_campaign_values(config)
    _verify_file(config.data_manifest, config.data_manifest_sha256)
    _verify_file(config.env_config, config.env_config_sha256)
    _verify_file(config.train_config, config.train_config_sha256)
    _verify_data_campaign(config)
    return config


def train_and_select_on_inner_periods(
    *,
    config: WalkForwardCampaignConfig,
    fold_id: str,
    seed: int,
    total_timesteps: int,
    output_dir: str | Path,
    pilot: bool,
) -> SelectionResult:
    """Fit on inner train and select solely on inner validation."""
    if fold_id not in config.folds:
        raise ValueError(f"unknown walk-forward fold: {fold_id}")
    if seed not in config.seeds:
        raise ValueError(f"unknown walk-forward seed: {seed}")
    if total_timesteps <= 0:
        raise ValueError("total_timesteps must be positive")
    destination = Path(output_dir)
    if destination.exists():
        return verify_selection_freeze(destination, config=config)
    destination.parent.mkdir(parents=True, exist_ok=True)

    fold_dir = config.data_root / fold_id
    fold_manifest_path = fold_dir / "fold_manifest.json"
    fold_manifest = _verify_fold_manifest(config, fold_id, fold_manifest_path)
    dataset = load_training_selection_dataset(fold_dir)
    train_store = PortfolioFeatureStore(dataset, "inner_train")
    validation_store = PortfolioFeatureStore(dataset, "inner_validation")
    env_config = load_env_config(config.env_config)
    train_config = load_train_ppo_config(config.train_config)
    _verify_runtime_contract(config, env_config, train_config)
    run_id = (
        f"{config.campaign_id}_{'pilot_' if pilot else ''}"
        f"{fold_id.lower()}_seed_{seed}_{total_timesteps}"
    )
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )

    def make_env() -> PortfolioEnv:
        return PortfolioEnv(
            feature_store=train_store,
            env_config=env_config,
            episode_sampler=RandomWindowEpisodeSampler(),
            seed=seed,
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
            seed=seed,
            policy_kwargs={
                "net_arch": {
                    "pi": train_config.network.pi,
                    "vf": train_config.network.vf,
                }
            },
            verbose=0,
        )
        callback = ValidationCheckpointCallback(
            validation_store=validation_store,
            action_temperature=env_config.action_temperature,
            rebalance_frequency_trading_days=config.rebalance_days,
            transaction_cost_bps=config.transaction_cost_bps,
            eval_freq_timesteps=config.eval_freq_timesteps,
            metric_for_best_model="sharpe_ratio",
            output_dir=temporary,
        )
        model.learn(total_timesteps=total_timesteps, callback=callback)
        model.save(temporary / "final_model.zip")
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    finally:
        vec_env.close()

    try:
        history = callback.validation_history
        actual_timesteps = _validate_validation_history(
            history,
            total_timesteps=total_timesteps,
            eval_freq_timesteps=config.eval_freq_timesteps,
            rollout_timesteps=train_config.ppo.n_steps,
        )
        if callback.best_step is None or callback.best_metrics is None:
            raise RuntimeError("no finite validation checkpoint was selected")
        best_path = temporary / "best_model.zip"
        selected_path = temporary / "selected_model.zip"
        if not best_path.is_file():
            raise RuntimeError("selected validation checkpoint was not saved")
        shutil.copy2(best_path, selected_path)
        (temporary / "validation_history.json").write_text(
            json.dumps(history, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        selected_policy = load_sb3_weight_policy(
            selected_path,
            action_temperature=env_config.action_temperature,
        )
        selected_validation = run_weight_policy_backtest(
            feature_store=validation_store,
            policy=selected_policy,
            strategy="ppo_selected_inner_validation",
            rebalance_frequency_trading_days=config.rebalance_days,
            transaction_cost_bps=config.transaction_cost_bps,
        )
        write_backtest_artifacts(
            selected_validation,
            temporary / "selected_validation_backtest",
        )
        freeze = {
            "schema_version": 1,
            "campaign_id": config.campaign_id,
            "pilot": pilot,
            "fold_id": fold_id,
            "seed": seed,
            "run_id": run_id,
            "requested_total_timesteps": total_timesteps,
            "actual_total_timesteps": actual_timesteps,
            "training_split": "inner_train",
            "checkpoint_selection_split": "inner_validation",
            "outer_accessed": False,
            "evaluated_checkpoint_steps": [
                int(record["step"]) for record in history
            ],
            "inner_validation_metrics": history,
            "selection_rule": {
                "metric": "sharpe_ratio",
                "direction": "maximize",
                "tie_break": "earliest_step",
                "final_endpoint_participates": True,
            },
            "selected_checkpoint_step": callback.best_step,
            "selected_validation_metrics": callback.best_metrics,
            "selected_model": {
                "path": "selected_model.zip",
                "sha256": _sha256(selected_path),
            },
            "hashes": {
                "campaign_config": config.config_sha256,
                "walk_forward_data_manifest": config.data_manifest_sha256,
                "fold_manifest": _sha256(fold_manifest_path),
                "training_selection_matrix_file": fold_manifest["artifact_hashes"][
                    "training_selection_matrix"
                ]["file_sha256"],
                "training_selection_matrix_logical": fold_manifest[
                    "artifact_hashes"
                ]["training_selection_matrix"]["logical_sha256"],
                "scaler_file": fold_manifest["artifact_hashes"]["scaler"][
                    "file_sha256"
                ],
                "feature_spec_file": fold_manifest["artifact_hashes"][
                    "feature_spec"
                ]["file_sha256"],
                "env_config": config.env_config_sha256,
                "train_config": config.train_config_sha256,
            },
            "git_commit": _git_commit(config.config_path.parent),
            "frozen_at": datetime.now(UTC).isoformat(),
        }
        write_selection_freeze(temporary / "selection_freeze.json", freeze)
        os.replace(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return verify_selection_freeze(destination, config=config)


def write_selection_freeze(path: str | Path, freeze: dict[str, Any]) -> None:
    """Write an immutable selection-freeze manifest."""
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"selection freeze already exists: {destination}")
    destination.write_text(
        json.dumps(freeze, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def verify_selection_freeze(
    output_dir: str | Path,
    *,
    config: WalkForwardCampaignConfig,
) -> SelectionResult:
    """Verify a completed selection bundle without loading outer data."""
    directory = Path(output_dir)
    freeze_path = directory / "selection_freeze.json"
    freeze = _read_json(freeze_path)
    if freeze.get("campaign_id") != config.campaign_id:
        raise ValueError("selection freeze campaign does not match configuration")
    if freeze.get("outer_accessed") is not False:
        raise ValueError("selection freeze indicates outer access")
    hashes = _mapping(freeze, "hashes")
    expected = {
        "campaign_config": config.config_sha256,
        "walk_forward_data_manifest": config.data_manifest_sha256,
        "env_config": config.env_config_sha256,
        "train_config": config.train_config_sha256,
    }
    for name, value in expected.items():
        if hashes.get(name) != value:
            raise ValueError(f"selection freeze hash mismatch: {name}")
    selected = _mapping(freeze, "selected_model")
    model_path = directory / _text(selected, "path")
    _verify_file(model_path, _text(selected, "sha256"))
    fold_id = _text(freeze, "fold_id")
    fold_manifest_path = config.data_root / fold_id / "fold_manifest.json"
    if _sha256(fold_manifest_path) != hashes.get("fold_manifest"):
        raise ValueError("selection freeze fold-manifest hash mismatch")
    return SelectionResult(
        fold_id=fold_id,
        seed=int(_required(freeze, "seed")),
        run_id=_text(freeze, "run_id"),
        output_dir=directory,
        freeze_path=freeze_path,
        selected_model_path=model_path,
    )


def selection_output_dir(
    config: WalkForwardCampaignConfig,
    *,
    fold_id: str,
    seed: int,
    pilot: bool,
) -> Path:
    root = config.campaign_root / ("pilot/selection" if pilot else "selection")
    return root / fold_id / f"seed_{seed}"


def run_selection_stage(
    config: WalkForwardCampaignConfig,
    *,
    pilot: bool,
) -> list[SelectionResult]:
    """Run the pilot or all declared production selections sequentially."""
    if not pilot:
        _verify_pilot_gate(config)
    pairs = (
        [(config.pilot_fold, config.pilot_seed, config.pilot_timesteps)]
        if pilot
        else [
            (fold, seed, config.total_timesteps)
            for fold in config.folds
            for seed in config.seeds
        ]
    )
    results = []
    for fold_id, seed, total_timesteps in pairs:
        results.append(
            train_and_select_on_inner_periods(
                config=config,
                fold_id=fold_id,
                seed=seed,
                total_timesteps=total_timesteps,
                output_dir=selection_output_dir(
                    config,
                    fold_id=fold_id,
                    seed=seed,
                    pilot=pilot,
                ),
                pilot=pilot,
            )
        )
    return results


def _verify_pilot_gate(config: WalkForwardCampaignConfig) -> None:
    path = config.campaign_root / "pilot" / "pilot_verification.json"
    if not path.is_file():
        raise FileNotFoundError("production selection requires a completed pilot")
    payload = _read_json(path)
    if (
        payload.get("campaign_id") != config.campaign_id
        or payload.get("campaign_config_sha256") != config.config_sha256
        or payload.get("passed") is not True
    ):
        raise ValueError("production selection requires a matching passing pilot")


def _validate_campaign_values(config: WalkForwardCampaignConfig) -> None:
    if config.total_timesteps != 500_000:
        raise ValueError("production total_timesteps must remain 500000")
    if (config.pilot_fold, config.pilot_seed, config.pilot_timesteps) != (
        "WF1",
        42,
        50_000,
    ):
        raise ValueError("pilot must be WF1, seed 42, 50000 steps")
    if config.eval_freq_timesteps != 25_000:
        raise ValueError("checkpoint evaluation frequency must remain 25000")
    if (
        config.rebalance_days,
        config.transaction_cost_bps,
        config.inverse_vol_lookback,
        config.momentum_lookback,
        config.momentum_top_k,
    ) != (5, 10.0, 63, 63, 3):
        raise ValueError("walk-forward execution settings do not match PR17")


def _verify_data_campaign(config: WalkForwardCampaignConfig) -> None:
    manifest = _read_json(config.data_manifest)
    if tuple(manifest.get("fold_order", [])) != config.folds:
        raise ValueError("walk-forward data fold order does not match campaign")
    if manifest.get("contains_2024_or_later") is not False:
        raise ValueError("walk-forward data manifest permits 2024 or later")


def _verify_fold_manifest(
    config: WalkForwardCampaignConfig,
    fold_id: str,
    path: Path,
) -> dict[str, Any]:
    campaign = _read_json(config.data_manifest)
    expected = _mapping(campaign, "fold_manifest_sha256").get(fold_id)
    if not isinstance(expected, str):
        raise TypeError(f"data manifest does not declare fold: {fold_id}")
    _verify_file(path, expected)
    manifest = _read_json(path)
    if manifest.get("fold_id") != fold_id:
        raise ValueError("fold manifest identity mismatch")
    if manifest.get("contains_2024_or_later") is not False:
        raise ValueError("fold manifest permits 2024 or later")
    if manifest.get("access_contract", {}).get(
        "outer_accessed_during_training_or_selection"
    ) is not False:
        raise ValueError("fold access contract does not prohibit outer selection")
    return manifest


def _verify_runtime_contract(config, env_config, train_config) -> None:
    if env_config.rebalance_frequency_trading_days != config.rebalance_days:
        raise ValueError("environment rebalance frequency differs from campaign")
    if env_config.transaction_cost_bps != config.transaction_cost_bps:
        raise ValueError("environment transaction cost differs from campaign")
    if env_config.action_temperature != 0.5:
        raise ValueError("action temperature must remain 0.5")
    if train_config.algorithm != "PPO" or train_config.policy != "MlpPolicy":
        raise ValueError("walk-forward training requires PPO MlpPolicy")
    if train_config.evaluation.metric_for_best_model != "sharpe_ratio":
        raise ValueError("base training selection metric must remain Sharpe")


def _validate_validation_history(
    history: list[dict[str, object]],
    *,
    total_timesteps: int,
    eval_freq_timesteps: int,
    rollout_timesteps: int,
) -> int:
    expected = list(range(eval_freq_timesteps, total_timesteps + 1, eval_freq_timesteps))
    observed = [int(record["step"]) for record in history]
    if not observed:
        raise RuntimeError("validation history must not be empty")
    actual_timesteps = observed[-1]
    if actual_timesteps < total_timesteps:
        raise RuntimeError("training ended before the requested timestep budget")
    if actual_timesteps - total_timesteps >= rollout_timesteps:
        raise RuntimeError("training exceeded the requested budget by a full rollout")
    if not expected or expected[-1] != actual_timesteps:
        expected.append(actual_timesteps)
    if observed != expected:
        raise RuntimeError(
            f"validation checkpoint steps do not match contract: {observed} != {expected}"
        )
    return actual_timesteps


def _verify_file(path: Path, expected_sha256: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = _sha256(path)
    if actual != expected_sha256:
        raise ValueError(
            f"file hash mismatch for {path}: expected={expected_sha256}, actual={actual}"
        )


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit(path: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _resolve(root: Path, path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def _read_json(path: Path) -> dict[str, Any]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise TypeError(f"expected JSON object: {path}")
    return loaded


def _required(mapping: dict[str, Any], key: str) -> Any:
    if key not in mapping:
        raise ValueError(f"missing required walk-forward campaign key: {key}")
    return mapping[key]


def _mapping(mapping: dict[str, Any], key: str) -> dict[str, Any]:
    value = _required(mapping, key)
    if not isinstance(value, dict):
        raise TypeError(f"walk-forward campaign key must be a mapping: {key}")
    return value


def _list(mapping: dict[str, Any], key: str) -> list[Any]:
    value = _required(mapping, key)
    if not isinstance(value, list):
        raise TypeError(f"walk-forward campaign key must be a list: {key}")
    return value


def _text(mapping: dict[str, Any], key: str) -> str:
    value = str(_required(mapping, key)).strip()
    if not value:
        raise ValueError(f"walk-forward campaign key must not be empty: {key}")
    return value

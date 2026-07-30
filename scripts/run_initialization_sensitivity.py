"""Run the frozen seed-42 initialization-sensitivity diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from portfolio_rl.config.loader import load_env_config, load_yaml
from portfolio_rl.data.dataset import load_portfolio_dataset
from portfolio_rl.data.feature_store import PortfolioFeatureStore
from portfolio_rl.evaluation.initialization import (
    EqualWeightInitializer,
    InverseVolatilityInitializer,
    SHYInitializer,
)
from portfolio_rl.evaluation.initialization_sensitivity import (
    evaluate_initialization_sensitivity,
    write_initialization_sensitivity_artifacts,
)
from portfolio_rl.policies.sb3_policy import load_sb3_weight_policy


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate frozen seed-42 PPO initialization sensitivity.",
    )
    parser.add_argument(
        "--config",
        default="configs/research/phase3_initialization_sensitivity.yaml",
        help="Strict PR 12 research configuration.",
    )
    parser.add_argument(
        "--root",
        default=".",
        help="Repository root used to resolve paths.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output override; the destination must not exist.",
    )
    args = parser.parse_args(argv)
    outputs = run_initialization_sensitivity(
        config_path=args.config,
        root=args.root,
        output_dir=args.output_dir,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")


def run_initialization_sensitivity(
    *,
    config_path: str | Path,
    root: str | Path = ".",
    output_dir: str | Path | None = None,
) -> dict[str, Path]:
    """Verify the PR 11 freeze and execute the seed-42 diagnostic."""
    root_path = Path(root).resolve()
    resolved_config = _resolve(root_path, config_path)
    config = load_yaml(resolved_config)
    freeze_dir = _resolve(root_path, _required_text(config, "freeze_dir"))
    seed = int(_required(config, "seed"))
    if seed != 42:
        raise ValueError("PR 12 is scoped to the frozen seed-42 policy")
    freeze_manifest, seed_record = _verify_frozen_inputs(
        root_path=root_path,
        freeze_dir=freeze_dir,
        seed=seed,
    )

    model_path = _resolve(root_path, seed_record["model"]["path"])
    data_config_path = _resolve(root_path, seed_record["data_config_hash"]["path"])
    env_config_path = _resolve(root_path, seed_record["env_config_hash"]["path"])
    feature_spec_path = _resolve(
        root_path,
        seed_record["feature_spec_hash"]["path"],
    )
    data_quality_path = _resolve(
        root_path,
        seed_record["data_quality_report_hash"]["path"],
    )
    train_config_path = _resolve(
        root_path,
        seed_record["train_config_hash"]["path"],
    )
    run_manifest_path = _resolve(root_path, seed_record["manifest"]["path"])
    data_config = load_yaml(data_config_path)
    env_config = load_env_config(env_config_path)
    requested_split = _required_text(config, "requested_split")
    evaluation_start = pd.Timestamp(
        _required_text(config, "evaluation_start_date")
    ).normalize()
    evaluation_end = pd.Timestamp(
        _required_text(config, "evaluation_end_date")
    ).normalize()
    configured_test_start = pd.Timestamp(
        _required_text(data_config, "test_start_date")
    ).normalize()

    dataset = load_portfolio_dataset(
        root_path,
        feature_spec_path=feature_spec_path,
    )
    feature_store = PortfolioFeatureStore(
        dataset,
        split=requested_split,
        start_date=evaluation_start,
        end_date=evaluation_end,
    )
    initialization = _required_mapping(config, "initialization")
    convergence = _required_mapping(config, "convergence")
    if _required_text(initialization, "headline") != "equal_weight":
        raise ValueError("PR 12 headline initialization must remain equal_weight")
    if _required(initialization, "diagnostics") != [
        "inverse_volatility_63d",
        "shy_100pct",
    ]:
        raise ValueError("PR 12 diagnostic initializers do not match the contract")
    if _required_text(convergence, "metric") != "half_l1":
        raise ValueError("PR 12 convergence.metric must be half_l1")
    lookback = int(
        _required(initialization, "inverse_volatility_lookback_trading_days")
    )
    threshold = float(_required(convergence, "threshold"))
    consecutive_decisions = int(
        _required(convergence, "consecutive_decisions")
    )
    if lookback != 63:
        raise ValueError("PR 12 inverse-volatility lookback must be exactly 63")
    if threshold != 0.05 or consecutive_decisions != 4:
        raise ValueError(
            "PR 12 convergence requires half_l1 <= 0.05 for four decisions"
        )

    def seed_42_factory() -> Any:
        return load_sb3_weight_policy(
            model_path,
            action_temperature=env_config.action_temperature,
        )

    result = evaluate_initialization_sensitivity(
        feature_store=feature_store,
        candidate_policy_factories={"seed_42": seed_42_factory},
        initializers={
            "equal_weight": EqualWeightInitializer(),
            "inverse_volatility_63d": InverseVolatilityInitializer(
                lookback=lookback
            ),
            "shy_100pct": SHYInitializer(),
        },
        configured_test_start_date=configured_test_start,
        rebalance_frequency_trading_days=(
            env_config.rebalance_frequency_trading_days
        ),
        transaction_cost_bps=env_config.transaction_cost_bps,
        convergence_threshold=threshold,
        convergence_consecutive_decisions=consecutive_decisions,
    )
    destination = (
        _resolve(root_path, output_dir)
        if output_dir is not None
        else root_path
        / "artifacts"
        / "initialization_sensitivity"
        / str(freeze_manifest["campaign_id"])
    )
    source_paths = {
        "pr12_config": resolved_config,
        "freeze_manifest": freeze_dir / "freeze_manifest.json",
        "model": model_path,
        "model_manifest": run_manifest_path,
        "data_config": data_config_path,
        "environment_config": env_config_path,
        "training_config": train_config_path,
        "feature_spec": feature_spec_path,
        "data_quality_report": data_quality_path,
        "model_matrix": root_path / "data/processed/model_matrix_daily.parquet",
    }
    manifest = {
        "campaign_id": freeze_manifest["campaign_id"],
        "candidate": "seed_42",
        "seed": 42,
        "artifact_directory": _manifest_path(destination, root_path),
        "requested_split": requested_split,
        "evaluation_start_date": feature_store.date_at(0).date().isoformat(),
        "evaluation_end_date": feature_store.date_at(
            feature_store.n_rows - 1
        ).date().isoformat(),
        "configured_test_start_date": configured_test_start.date().isoformat(),
        "test_accessed": False,
        "headline_initializer": "equal_weight",
        "diagnostic_initializers": [
            "inverse_volatility_63d",
            "shy_100pct",
        ],
        "inverse_volatility_lookback_rows": lookback,
        "initial_endowment_nav": 1.0,
        "initial_endowment_establishment_cost": 0.0,
        "convergence": {
            "metric": "half_l1",
            "threshold": threshold,
            "consecutive_decisions": consecutive_decisions,
            "primary_path": "target_weights",
        },
        "source_hashes": {
            name: {
                "path": str(path.relative_to(root_path)),
                "sha256": _sha256(path),
            }
            for name, path in source_paths.items()
        },
        "freeze_payload_hashes_verified": True,
        "model_and_training_artifact_hashes_verified": True,
    }
    return write_initialization_sensitivity_artifacts(
        result=result,
        output_dir=destination,
        manifest=manifest,
    )


def _verify_frozen_inputs(
    *,
    root_path: Path,
    freeze_dir: Path,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_path = freeze_dir / "freeze_manifest.json"
    manifest = _read_json(manifest_path)
    if not manifest.get("provenance_passed"):
        raise ValueError("research freeze provenance did not pass")
    if not manifest.get("campaign_test_free"):
        raise ValueError("research freeze campaign is not test-free")
    for payload in manifest.get("files", []):
        path = freeze_dir / str(payload["path"])
        _require_hash(path, str(payload["sha256"]))
    hash_inventory = _read_json(freeze_dir / "hash_inventory.json")
    matching = [
        record for record in hash_inventory["runs"] if int(record["seed"]) == seed
    ]
    if len(matching) != 1:
        raise ValueError(f"research freeze must contain exactly one seed {seed}")
    seed_record = matching[0]
    for field in (
        "model",
        "manifest",
        "data_config_hash",
        "data_quality_report_hash",
        "env_config_hash",
        "feature_spec_hash",
        "train_config_hash",
    ):
        record = seed_record[field]
        if not record.get("matches"):
            raise ValueError(f"frozen seed artifact did not pass: {field}")
        _require_hash(
            _resolve(root_path, record["path"]),
            str(record["expected_sha256"]),
        )
    return manifest, seed_record


def _read_json(path: Path) -> dict[str, Any]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise TypeError(f"expected JSON object: {path}")
    return loaded


def _required(mapping: dict[str, Any], key: str) -> Any:
    if key not in mapping:
        raise ValueError(f"missing required configuration key: {key}")
    return mapping[key]


def _required_text(mapping: dict[str, Any], key: str) -> str:
    value = str(_required(mapping, key)).strip()
    if not value:
        raise ValueError(f"configuration key must not be empty: {key}")
    return value


def _required_mapping(mapping: dict[str, Any], key: str) -> dict[str, Any]:
    value = _required(mapping, key)
    if not isinstance(value, dict):
        raise TypeError(f"configuration key must be a mapping: {key}")
    return value


def _resolve(root: Path, path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def _manifest_path(path: Path, root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _require_hash(path: Path, expected: str) -> None:
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(
            f"artifact hash mismatch: {path}; expected={expected}, actual={actual}"
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()

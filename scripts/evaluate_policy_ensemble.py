"""Run the frozen five-seed mean-target-weight ensemble campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from portfolio_rl.config.loader import load_env_config, load_yaml
from portfolio_rl.data.dataset import load_portfolio_dataset
from portfolio_rl.data.feature_store import PortfolioFeatureStore
from portfolio_rl.evaluation.ensemble import (
    evaluate_ensemble_campaign,
    write_ensemble_artifacts,
)
from portfolio_rl.evaluation.initialization import (
    EqualWeightInitializer,
    InverseVolatilityInitializer,
    SHYInitializer,
)
from portfolio_rl.evaluation.initialization_sensitivity import (
    evaluate_initialization_sensitivity,
)
from portfolio_rl.policies.baseline_policies import (
    BuyAndHoldEqualWeightPolicy,
    EqualWeightWeeklyPolicy,
    InverseVolatilityPolicy,
    SingleAssetPolicy,
)
from portfolio_rl.policies.ensemble_policy import MeanWeightEnsemblePolicy
from portfolio_rl.policies.sb3_policy import load_sb3_weight_policy

EXPECTED_SEEDS = [7, 42, 101, 202, 999]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the frozen five-seed mean-weight PPO ensemble.",
    )
    parser.add_argument(
        "--config",
        default="configs/research/phase3_ensemble.yaml",
        help="Strict PR 13 ensemble research configuration.",
    )
    parser.add_argument("--root", default=".", help="Repository root.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output override; destination must not exist.",
    )
    args = parser.parse_args(argv)
    outputs = run_ensemble_evaluation(
        config_path=args.config,
        root=args.root,
        output_dir=args.output_dir,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")


def run_ensemble_evaluation(
    *,
    config_path: str | Path,
    root: str | Path = ".",
    output_dir: str | Path | None = None,
) -> dict[str, Path]:
    """Verify the research freeze and execute the PR 13 campaign."""
    root_path = Path(root).resolve()
    resolved_config = _resolve(root_path, config_path)
    config = load_yaml(resolved_config)
    freeze_dir = _resolve(root_path, _required_text(config, "freeze_dir"))
    freeze_manifest, model_records = _verify_freeze(
        root_path=root_path,
        freeze_dir=freeze_dir,
    )
    configured_seeds = [int(seed) for seed in _required_list(config, "member_seeds")]
    if configured_seeds != EXPECTED_SEEDS:
        raise ValueError(f"member_seeds must equal {EXPECTED_SEEDS}")
    representative_seed = int(_required(config, "representative_seed"))
    if representative_seed != 42:
        raise ValueError("representative_seed must remain 42")

    first_record = model_records[0]
    data_config_path = _resolve(
        root_path,
        first_record["data_config_hash"]["path"],
    )
    feature_spec_path = _resolve(
        root_path,
        first_record["feature_spec_hash"]["path"],
    )
    data_config = load_yaml(data_config_path)
    configured_test_start = pd.Timestamp(
        _required_text(data_config, "test_start_date")
    ).normalize()
    dataset = load_portfolio_dataset(
        root_path,
        feature_spec_path=feature_spec_path,
    )
    windows = _required_mapping(config, "windows")
    feature_stores = {
        regime: PortfolioFeatureStore(
            dataset,
            split=_required_text(window, "split"),
            start_date=_required_text(window, "start_date"),
            end_date=_required_text(window, "end_date"),
        )
        for regime, window in windows.items()
    }
    if list(feature_stores) != ["validation_2024", "historical_2022"]:
        raise ValueError(
            "windows must be ordered as validation_2024, historical_2022"
        )

    env_configs = {
        int(record["seed"]): load_env_config(
            _resolve(root_path, record["env_config_hash"]["path"])
        )
        for record in model_records
    }
    common_rebalance = {
        env.rebalance_frequency_trading_days for env in env_configs.values()
    }
    common_cost = {env.transaction_cost_bps for env in env_configs.values()}
    if len(common_rebalance) != 1 or len(common_cost) != 1:
        raise ValueError("frozen members must share rebalance and cost settings")
    rebalance_days = common_rebalance.pop()
    transaction_cost_bps = common_cost.pop()
    member_factories = _member_factories(
        root_path=root_path,
        model_records=model_records,
        env_configs=env_configs,
    )
    asset_order = feature_stores["validation_2024"].asset_order
    baseline_factories = {
        "equal_weight_weekly": (
            lambda: EqualWeightWeeklyPolicy(n_assets=len(asset_order))
        ),
        "inverse_volatility": (
            lambda: InverseVolatilityPolicy(n_assets=len(asset_order))
        ),
        "buy_and_hold_equal_weight": (
            lambda: BuyAndHoldEqualWeightPolicy(n_assets=len(asset_order))
        ),
        "spy_only": lambda: SingleAssetPolicy(asset_order, "SPY"),
        "shy_only": lambda: SingleAssetPolicy(asset_order, "SHY"),
    }
    result = evaluate_ensemble_campaign(
        feature_stores=feature_stores,
        member_policy_factories=member_factories,
        baseline_policy_factories=baseline_factories,
        representative_seed=representative_seed,
        configured_test_start_date=configured_test_start,
        rebalance_frequency_trading_days=rebalance_days,
        transaction_cost_bps=transaction_cost_bps,
    )

    initialization = _required_mapping(config, "initialization_sensitivity")
    lookback = int(_required(initialization, "inverse_volatility_lookback_days"))
    threshold = float(_required(initialization, "convergence_threshold"))
    consecutive = int(_required(initialization, "consecutive_decisions"))
    if (lookback, threshold, consecutive) != (63, 0.05, 4):
        raise ValueError("ensemble initialization contract must be 63, 0.05, 4")

    def ensemble_factory() -> MeanWeightEnsemblePolicy:
        return MeanWeightEnsemblePolicy(
            member_policies={
                f"seed_{seed}": factory()
                for seed, factory in member_factories.items()
            }
        )

    initialization_result = evaluate_initialization_sensitivity(
        feature_store=feature_stores["validation_2024"],
        candidate_policy_factories={
            "five_seed_mean_weight_ensemble": ensemble_factory
        },
        initializers={
            "equal_weight": EqualWeightInitializer(),
            "inverse_volatility_63d": InverseVolatilityInitializer(
                lookback=lookback
            ),
            "shy_100pct": SHYInitializer(),
        },
        configured_test_start_date=configured_test_start,
        rebalance_frequency_trading_days=rebalance_days,
        transaction_cost_bps=transaction_cost_bps,
        convergence_threshold=threshold,
        convergence_consecutive_decisions=consecutive,
    )

    destination = (
        _resolve(root_path, output_dir)
        if output_dir is not None
        else root_path
        / "artifacts"
        / "ensemble"
        / str(freeze_manifest["campaign_id"])
    )
    manifest = {
        "campaign_id": freeze_manifest["campaign_id"],
        "artifact_directory": _manifest_path(destination, root_path),
        "policy": "arithmetic_mean_of_member_target_weights",
        "member_seed_order": EXPECTED_SEEDS,
        "representative_seed": 42,
        "primary_initialization": "equal_weight",
        "primary_regime": "validation_2024",
        "development_data_label": "2024 consumed development/selection data",
        "historical_2022_label": "in-sample historical behavior diagnostic",
        "configured_test_start_date": configured_test_start.date().isoformat(),
        "test_accessed": False,
        "rebalance_frequency_trading_days": rebalance_days,
        "transaction_cost_bps": transaction_cost_bps,
        "initialization_sensitivity": {
            "headline": "equal_weight",
            "diagnostics": ["inverse_volatility_63d", "shy_100pct"],
            "inverse_volatility_lookback_days": lookback,
            "convergence_metric": "half_l1",
            "convergence_threshold": threshold,
            "consecutive_decisions": consecutive,
        },
        "members": [
            {
                "order": order,
                "seed": int(record["seed"]),
                "run_id": record["run_id"],
                "selection_checkpoint": "best_checkpoint",
                "model_path": record["model"]["path"],
                "model_sha256": record["model"]["actual_sha256"],
                "action_temperature": env_configs[int(record["seed"])].action_temperature,
            }
            for order, record in enumerate(model_records)
        ],
        "windows": {
            name: {
                "requested_split": store.split,
                "evaluation_start_date": store.date_at(0).date().isoformat(),
                "evaluation_end_date": store.date_at(
                    store.n_rows - 1
                ).date().isoformat(),
            }
            for name, store in feature_stores.items()
        },
        "source_hashes": {
            "pr13_config": _source(resolved_config, root_path),
            "freeze_manifest": _source(
                freeze_dir / "freeze_manifest.json",
                root_path,
            ),
            "feature_spec": _source(feature_spec_path, root_path),
            "data_config": _source(data_config_path, root_path),
            "model_matrix": _source(
                root_path / "data/processed/model_matrix_daily.parquet",
                root_path,
            ),
        },
        "freeze_payload_hashes_verified": True,
        "all_member_hashes_verified": True,
        "inverse_volatility_baseline_boundary_followup_open": True,
    }
    return write_ensemble_artifacts(
        result=result,
        initialization_result=initialization_result,
        output_dir=destination,
        manifest=manifest,
    )


def _member_factories(
    *,
    root_path: Path,
    model_records: list[dict[str, Any]],
    env_configs: Mapping[int, Any],
) -> dict[int, Any]:
    factories = {}
    for record in model_records:
        seed = int(record["seed"])
        model_path = _resolve(root_path, record["model"]["path"])
        temperature = env_configs[seed].action_temperature

        def factory(
            path: Path = model_path,
            action_temperature: float = temperature,
        ) -> Any:
            return load_sb3_weight_policy(path, action_temperature)

        factories[seed] = factory
    return factories


def _verify_freeze(
    *,
    root_path: Path,
    freeze_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = _read_json(freeze_dir / "freeze_manifest.json")
    if manifest.get("provenance_passed") is not True:
        raise ValueError("research freeze provenance did not pass")
    if manifest.get("campaign_test_free") is not True:
        raise ValueError("research freeze campaign is not test-free")
    for payload in manifest.get("files", []):
        _require_hash(
            freeze_dir / str(payload["path"]),
            str(payload["sha256"]),
        )
    inventory = _read_json(freeze_dir / "hash_inventory.json")
    records = inventory.get("runs")
    if not isinstance(records, list):
        raise TypeError("freeze hash inventory runs must be a list")
    records = sorted(records, key=lambda record: EXPECTED_SEEDS.index(int(record["seed"])))
    if [int(record["seed"]) for record in records] != EXPECTED_SEEDS:
        raise ValueError(f"freeze must contain exactly seeds {EXPECTED_SEEDS}")
    fields = (
        "model",
        "manifest",
        "data_config_hash",
        "data_quality_report_hash",
        "env_config_hash",
        "feature_spec_hash",
        "train_config_hash",
    )
    for record in records:
        for field in fields:
            artifact = record[field]
            if artifact.get("matches") is not True:
                raise ValueError(f"frozen artifact did not pass: {field}")
            _require_hash(
                _resolve(root_path, artifact["path"]),
                str(artifact["expected_sha256"]),
            )
    return manifest, records


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


def _required_list(mapping: dict[str, Any], key: str) -> list[Any]:
    value = _required(mapping, key)
    if not isinstance(value, list):
        raise TypeError(f"configuration key must be a list: {key}")
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


def _source(path: Path, root: Path) -> dict[str, str]:
    return {
        "path": _manifest_path(path, root),
        "sha256": _sha256(path),
    }


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

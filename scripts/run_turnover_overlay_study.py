"""Run the predeclared PR 14 partial-rebalancing frontier."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from portfolio_rl.config.loader import load_yaml
from portfolio_rl.data.dataset import load_portfolio_dataset
from portfolio_rl.data.feature_store import PortfolioFeatureStore
from portfolio_rl.evaluation.ensemble import ENSEMBLE_STRATEGY
from portfolio_rl.evaluation.turnover_overlay import (
    evaluate_turnover_overlay_study,
    write_turnover_overlay_artifacts,
)
from portfolio_rl.policies.baseline_policies import EqualWeightWeeklyPolicy
from portfolio_rl.policies.ensemble_policy import MeanWeightEnsemblePolicy
from portfolio_rl.policies.sb3_policy import load_sb3_weight_policy

EXPECTED_ALPHAS = [0.25, 0.5, 0.75, 1.0]
EXPECTED_SEEDS = [7, 42, 101, 202, 999]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the PR 14 partial-rebalancing frontier.",
    )
    parser.add_argument(
        "--config",
        default="configs/research/phase3_turnover_overlay.yaml",
        help="Strict PR 14 research configuration.",
    )
    parser.add_argument("--root", default=".", help="Repository root.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output override; destination must not exist.",
    )
    args = parser.parse_args(argv)
    outputs = run_turnover_overlay_study(
        config_path=args.config,
        root=args.root,
        output_dir=args.output_dir,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")


def run_turnover_overlay_study(
    *,
    config_path: str | Path,
    root: str | Path = ".",
    output_dir: str | Path | None = None,
) -> dict[str, Path]:
    """Verify PR 13 and execute the non-selective PR 14 frontier."""
    root_path = Path(root).resolve()
    config_path = _resolve(root_path, config_path)
    config = load_yaml(config_path)
    pr13_manifest_path = _resolve(
        root_path,
        _required_text(config, "pr13_manifest_path"),
    )
    expected_manifest_hash = _required_text(config, "pr13_manifest_sha256")
    _require_hash(pr13_manifest_path, expected_manifest_hash)
    pr13 = _read_json(pr13_manifest_path)
    _validate_pr13_manifest(pr13, root_path)
    alphas = [float(value) for value in _required_list(config, "alphas")]
    if alphas != EXPECTED_ALPHAS:
        raise ValueError(f"alphas must equal {EXPECTED_ALPHAS}")

    data_config_record = pr13["source_hashes"]["data_config"]
    feature_spec_record = pr13["source_hashes"]["feature_spec"]
    data_config_path = _resolve(root_path, data_config_record["path"])
    feature_spec_path = _resolve(root_path, feature_spec_record["path"])
    data_config = load_yaml(data_config_path)
    dataset = load_portfolio_dataset(
        root_path,
        feature_spec_path=feature_spec_path,
    )
    window = _required_mapping(config, "window")
    feature_store = PortfolioFeatureStore(
        dataset,
        split=_required_text(window, "split"),
        start_date=_required_text(window, "start_date"),
        end_date=_required_text(window, "end_date"),
    )
    configured_test_start = pd.Timestamp(
        _required_text(data_config, "test_start_date")
    ).normalize()
    member_factories = _member_factories(pr13, root_path)

    def ensemble_factory() -> MeanWeightEnsemblePolicy:
        return MeanWeightEnsemblePolicy(
            member_policies={
                f"seed_{seed}": factory()
                for seed, factory in member_factories.items()
            }
        )

    result = evaluate_turnover_overlay_study(
        feature_store=feature_store,
        candidate_policy_factories={
            ENSEMBLE_STRATEGY: ensemble_factory,
            "seed_42": member_factories[42],
        },
        alphas=alphas,
        hurdle_policy_factory=lambda: EqualWeightWeeklyPolicy(
            n_assets=feature_store.n_assets
        ),
        configured_test_start_date=configured_test_start,
        rebalance_frequency_trading_days=int(
            pr13["rebalance_frequency_trading_days"]
        ),
        transaction_cost_bps=float(pr13["transaction_cost_bps"]),
    )
    pr13_dir = _resolve(root_path, pr13["artifact_directory"])
    _reconcile_alpha_one(
        result=result,
        pr13_dir=pr13_dir,
    )

    destination = (
        _resolve(root_path, output_dir)
        if output_dir is not None
        else root_path
        / "artifacts"
        / "turnover_overlays"
        / str(pr13["campaign_id"])
    )
    manifest = {
        "campaign_id": pr13["campaign_id"],
        "artifact_directory": _manifest_path(destination, root_path),
        "study": "partial_rebalancing_frontier",
        "alphas": EXPECTED_ALPHAS,
        "primary_candidate": ENSEMBLE_STRATEGY,
        "secondary_candidate": "seed_42",
        "hurdle": "equal_weight_weekly",
        "initialization": "equal_weight",
        "development_data_label": "2024 consumed development/selection data",
        "requested_split": feature_store.split,
        "evaluation_start_date": feature_store.date_at(0).date().isoformat(),
        "evaluation_end_date": feature_store.date_at(
            feature_store.n_rows - 1
        ).date().isoformat(),
        "configured_test_start_date": configured_test_start.date().isoformat(),
        "test_accessed": False,
        "alpha_selected": None,
        "all_alphas_advance_to_walk_forward": True,
        "pr13_manifest_path": _manifest_path(pr13_manifest_path, root_path),
        "pr13_manifest_sha256": expected_manifest_hash,
        "member_seed_order": EXPECTED_SEEDS,
        "members": pr13["members"],
        "rebalance_frequency_trading_days": int(
            pr13["rebalance_frequency_trading_days"]
        ),
        "transaction_cost_bps": float(pr13["transaction_cost_bps"]),
        "source_hashes": {
            "pr14_config": _source(config_path, root_path),
            "pr13_manifest": _source(pr13_manifest_path, root_path),
            "data_config": _source(data_config_path, root_path),
            "feature_spec": _source(feature_spec_path, root_path),
            "model_matrix": _source(
                root_path / "data/processed/model_matrix_daily.parquet",
                root_path,
            ),
        },
        "alpha_one_reconciled_to_pr13": True,
    }
    return write_turnover_overlay_artifacts(
        result=result,
        output_dir=destination,
        manifest=manifest,
    )


def _validate_pr13_manifest(
    manifest: dict[str, Any],
    root_path: Path,
) -> None:
    if manifest.get("member_seed_order") != EXPECTED_SEEDS:
        raise ValueError("PR 13 member seed order does not match")
    if manifest.get("test_accessed") is not False:
        raise ValueError("PR 13 manifest is not test-free")
    if manifest.get("primary_initialization") != "equal_weight":
        raise ValueError("PR 13 primary initialization must be equal_weight")
    if manifest.get("primary_regime") != "validation_2024":
        raise ValueError("PR 13 primary regime must be validation_2024")
    for source in manifest["source_hashes"].values():
        _require_hash(_resolve(root_path, source["path"]), source["sha256"])
    members = manifest.get("members")
    if not isinstance(members, list) or len(members) != 5:
        raise ValueError("PR 13 manifest must contain five members")
    if [int(member["seed"]) for member in members] != EXPECTED_SEEDS:
        raise ValueError("PR 13 member records are not in frozen seed order")
    for member in members:
        _require_hash(
            _resolve(root_path, member["model_path"]),
            member["model_sha256"],
        )


def _member_factories(
    manifest: dict[str, Any],
    root_path: Path,
) -> dict[int, Any]:
    factories = {}
    for member in manifest["members"]:
        seed = int(member["seed"])
        model_path = _resolve(root_path, member["model_path"])
        temperature = float(member["action_temperature"])

        def factory(
            path: Path = model_path,
            action_temperature: float = temperature,
        ) -> Any:
            return load_sb3_weight_policy(path, action_temperature)

        factories[seed] = factory
    return factories


def _reconcile_alpha_one(
    *,
    result: Any,
    pr13_dir: Path,
) -> None:
    expected_metrics = pd.read_csv(pr13_dir / "comparison_metrics.csv")
    for candidate in (ENSEMBLE_STRATEGY, "seed_42"):
        expected = expected_metrics[
            (expected_metrics["regime"] == "validation_2024")
            & (expected_metrics["strategy"] == candidate)
        ]
        actual = result.overlay_results[
            (result.overlay_results["candidate"] == candidate)
            & np.isclose(result.overlay_results["alpha"], 1.0)
        ]
        if len(expected) != 1 or len(actual) != 1:
            raise ValueError(f"missing alpha-1 reconciliation row: {candidate}")
        for metric in (
            "total_return",
            "sharpe_ratio",
            "max_drawdown",
            "average_weekly_turnover",
            "transaction_cost_drag",
        ):
            if not np.isclose(
                float(actual.iloc[0][metric]),
                float(expected.iloc[0][metric]),
                atol=1e-12,
                rtol=0.0,
            ):
                raise ValueError(
                    f"alpha 1.0 does not reconcile for {candidate}: {metric}"
                )

    expected_ensemble = pd.read_parquet(pr13_dir / "ensemble_targets.parquet")
    expected_ensemble = expected_ensemble[
        expected_ensemble["regime"] == "validation_2024"
    ]
    expected_seed = pd.read_parquet(
        pr13_dir / "backtest/validation_2024/seed_42/weights_target.parquet"
    )
    for candidate, expected in (
        (ENSEMBLE_STRATEGY, expected_ensemble),
        ("seed_42", expected_seed),
    ):
        audit = result.target_audit[
            (result.target_audit["candidate"] == candidate)
            & np.isclose(result.target_audit["alpha"], 1.0)
        ]
        expected_values = expected.sort_values(["date", "ticker"])[
            (
                "ensemble_target_weight"
                if candidate == ENSEMBLE_STRATEGY
                else "target_weight"
            )
        ].to_numpy(dtype=np.float64)
        audit = audit.sort_values(["date", "ticker"])
        if not np.allclose(
            audit["raw_policy_target"].to_numpy(dtype=np.float64),
            expected_values,
            atol=1e-12,
            rtol=0.0,
        ):
            raise ValueError(f"alpha-1 raw targets do not reconcile: {candidate}")
        if not np.array_equal(
            audit["raw_policy_target"].to_numpy(dtype=np.float64),
            audit["executed_target"].to_numpy(dtype=np.float64),
        ):
            raise ValueError(f"alpha-1 execution is not exact: {candidate}")


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

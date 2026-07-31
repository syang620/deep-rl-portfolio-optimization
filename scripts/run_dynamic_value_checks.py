"""Run the predeclared PR 15 dynamic-value diagnostics."""

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
from portfolio_rl.evaluation.dynamic_value import (
    EXPECTED_ALPHAS,
    EXPECTED_SHIFTS,
    evaluate_dynamic_value_checks,
    write_dynamic_value_artifacts,
)
from portfolio_rl.policies.baseline_policies import EqualWeightWeeklyPolicy
from portfolio_rl.policies.ensemble_policy import MeanWeightEnsemblePolicy
from portfolio_rl.policies.sb3_policy import load_sb3_weight_policy

EXPECTED_SEEDS = [7, 42, 101, 202, 999]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate frozen-policy dynamic-value controls.",
    )
    parser.add_argument(
        "--config",
        default="configs/research/phase3_dynamic_value.yaml",
        help="Strict PR 15 research configuration.",
    )
    parser.add_argument("--root", default=".", help="Repository root.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output override; destination must not exist.",
    )
    args = parser.parse_args(argv)
    outputs = run_dynamic_value_study(
        config_path=args.config,
        root=args.root,
        output_dir=args.output_dir,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")


def run_dynamic_value_study(
    *,
    config_path: str | Path,
    root: str | Path = ".",
    output_dir: str | Path | None = None,
) -> dict[str, Path]:
    """Verify PR 14 and execute the non-selective PR 15 study."""
    root_path = Path(root).resolve()
    config_path = _resolve(root_path, config_path)
    config = load_yaml(config_path)
    pr14_manifest_path = _resolve(
        root_path,
        _required_text(config, "pr14_manifest_path"),
    )
    expected_pr14_hash = _required_text(config, "pr14_manifest_sha256")
    _require_hash(pr14_manifest_path, expected_pr14_hash)
    pr14 = _read_json(pr14_manifest_path)
    pr13 = _validate_pr14_manifest(pr14, root_path)

    alphas = [float(value) for value in _required_list(config, "alphas")]
    shifts = [
        int(value)
        for value in _required_list(config, "circular_shift_decisions")
    ]
    if alphas != EXPECTED_ALPHAS:
        raise ValueError(f"alphas must equal {EXPECTED_ALPHAS}")
    if shifts != EXPECTED_SHIFTS:
        raise ValueError(f"circular shifts must equal {EXPECTED_SHIFTS}")
    if _required_text(config, "target_basis") != "executed_target":
        raise ValueError("target_basis must be executed_target")

    data_record = pr14["source_hashes"]["data_config"]
    feature_record = pr14["source_hashes"]["feature_spec"]
    data_path = _resolve(root_path, data_record["path"])
    feature_path = _resolve(root_path, feature_record["path"])
    data_config = load_yaml(data_path)
    configured_test_start = pd.Timestamp(
        _required_text(data_config, "test_start_date")
    ).normalize()
    dataset = load_portfolio_dataset(root_path, feature_spec_path=feature_path)
    prior_store = _build_store(
        dataset=dataset,
        window=_required_mapping(config, "prior_window"),
    )
    evaluation_store = _build_store(
        dataset=dataset,
        window=_required_mapping(config, "evaluation_window"),
    )
    if (
        prior_store.split != "train"
        or prior_store.date_at(0).year != 2023
        or prior_store.date_at(prior_store.n_rows - 1).year != 2023
    ):
        raise ValueError("prior window must be the train-split calendar 2023")
    if (
        evaluation_store.split != "validation"
        or evaluation_store.date_at(0).year != 2024
        or evaluation_store.date_at(evaluation_store.n_rows - 1).year != 2024
    ):
        raise ValueError(
            "evaluation window must be the validation-split calendar 2024"
        )

    member_factories = _member_factories(pr14, root_path)

    def ensemble_factory() -> MeanWeightEnsemblePolicy:
        return MeanWeightEnsemblePolicy(
            member_policies={
                f"seed_{seed}": factory()
                for seed, factory in member_factories.items()
            }
        )

    result = evaluate_dynamic_value_checks(
        prior_feature_store=prior_store,
        evaluation_feature_store=evaluation_store,
        ensemble_policy_factory=ensemble_factory,
        hurdle_policy_factory=lambda: EqualWeightWeeklyPolicy(
            evaluation_store.n_assets
        ),
        alphas=alphas,
        circular_shifts=shifts,
        configured_test_start_date=configured_test_start,
        rebalance_frequency_trading_days=int(
            pr14["rebalance_frequency_trading_days"]
        ),
        transaction_cost_bps=float(pr14["transaction_cost_bps"]),
    )
    pr14_dir = _resolve(root_path, pr14["artifact_directory"])
    _reconcile_pr14(result, pr14_dir)

    destination = (
        _resolve(root_path, output_dir)
        if output_dir is not None
        else root_path / "artifacts" / "dynamic_value" / str(pr14["campaign_id"])
    )
    manifest = {
        "campaign_id": pr14["campaign_id"],
        "artifact_directory": _manifest_path(destination, root_path),
        "study": "dynamic_value_diagnostics",
        "primary_candidate": "five_seed_mean_weight_ensemble",
        "candidates": [
            "ensemble_unmodified",
            "ensemble_alpha_0.25",
            "ensemble_alpha_0.50",
            "ensemble_alpha_0.75",
            "ensemble_alpha_1.00",
        ],
        "alphas": EXPECTED_ALPHAS,
        "circular_shift_decisions": EXPECTED_SHIFTS,
        "target_basis": "executed_target",
        "initialization": "equal_weight",
        "hurdle": "equal_weight_weekly",
        "development_data_label": "2024 consumed development/selection data",
        "prior_window_label": "2023 past-only in-sample calibration",
        "requested_prior_split": prior_store.split,
        "prior_start_date": prior_store.date_at(0).date().isoformat(),
        "prior_end_date": prior_store.date_at(
            prior_store.n_rows - 1
        ).date().isoformat(),
        "requested_evaluation_split": evaluation_store.split,
        "evaluation_start_date": evaluation_store.date_at(0).date().isoformat(),
        "evaluation_end_date": evaluation_store.date_at(
            evaluation_store.n_rows - 1
        ).date().isoformat(),
        "configured_test_start_date": configured_test_start.date().isoformat(),
        "test_accessed": False,
        "alpha_selected": None,
        "alphas_eliminated": [],
        "all_alphas_advance_to_walk_forward": True,
        "pr14_manifest_path": _manifest_path(pr14_manifest_path, root_path),
        "pr14_manifest_sha256": expected_pr14_hash,
        "pr13_manifest_path": pr14["pr13_manifest_path"],
        "pr13_manifest_sha256": pr14["pr13_manifest_sha256"],
        "member_seed_order": EXPECTED_SEEDS,
        "members": pr14["members"],
        "rebalance_frequency_trading_days": int(
            pr14["rebalance_frequency_trading_days"]
        ),
        "transaction_cost_bps": float(pr14["transaction_cost_bps"]),
        "source_hashes": {
            "pr15_config": _source(config_path, root_path),
            "pr14_manifest": _source(pr14_manifest_path, root_path),
            "pr13_manifest": _source(
                _resolve(root_path, pr14["pr13_manifest_path"]),
                root_path,
            ),
            "freeze_manifest": pr13["source_hashes"]["freeze_manifest"],
            "data_config": _source(data_path, root_path),
            "feature_spec": _source(feature_path, root_path),
            "model_matrix": _source(
                root_path / "data/processed/model_matrix_daily.parquet",
                root_path,
            ),
        },
        "pr14_reconciled": True,
        "unmodified_reconciled_to_alpha_1": True,
    }
    return write_dynamic_value_artifacts(
        result=result,
        output_dir=destination,
        manifest=manifest,
    )


def _build_store(*, dataset: Any, window: dict[str, Any]) -> PortfolioFeatureStore:
    return PortfolioFeatureStore(
        dataset,
        split=_required_text(window, "split"),
        start_date=_required_text(window, "start_date"),
        end_date=_required_text(window, "end_date"),
    )


def _validate_pr14_manifest(
    manifest: dict[str, Any],
    root_path: Path,
) -> dict[str, Any]:
    if manifest.get("test_accessed") is not False:
        raise ValueError("PR 14 manifest is not test-free")
    if manifest.get("alpha_selected") is not None:
        raise ValueError("PR 14 must not select an alpha")
    if manifest.get("alphas") != EXPECTED_ALPHAS:
        raise ValueError("PR 14 alpha grid does not match")
    if manifest.get("member_seed_order") != EXPECTED_SEEDS:
        raise ValueError("PR 14 member order does not match")
    for source in manifest["source_hashes"].values():
        _require_hash(_resolve(root_path, source["path"]), source["sha256"])
    for member in manifest["members"]:
        _require_hash(
            _resolve(root_path, member["model_path"]),
            member["model_sha256"],
        )
    pr13_path = _resolve(root_path, manifest["pr13_manifest_path"])
    _require_hash(pr13_path, manifest["pr13_manifest_sha256"])
    pr13 = _read_json(pr13_path)
    if pr13.get("test_accessed") is not False:
        raise ValueError("PR 13 manifest is not test-free")
    for source in pr13["source_hashes"].values():
        _require_hash(_resolve(root_path, source["path"]), source["sha256"])
    return pr13


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
    if list(factories) != EXPECTED_SEEDS:
        raise ValueError("member factories are not in frozen seed order")
    return factories


def _reconcile_pr14(result: Any, pr14_dir: Path) -> None:
    expected_metrics = pd.read_csv(pr14_dir / "overlay_results.csv")
    actual_metrics = result.diagnostic_results[
        result.diagnostic_results["diagnostic"] == "dynamic_live"
    ]
    for alpha in EXPECTED_ALPHAS:
        candidate = f"ensemble_alpha_{alpha:.2f}"
        expected = expected_metrics[
            (expected_metrics["candidate"] == "five_seed_mean_weight_ensemble")
            & np.isclose(expected_metrics["alpha"], alpha)
        ]
        actual = actual_metrics[actual_metrics["candidate"] == candidate]
        if len(expected) != 1 or len(actual) != 1:
            raise ValueError(f"missing PR 14 reconciliation row: alpha={alpha}")
        for metric in (
            "total_return",
            "sharpe_ratio",
            "max_drawdown",
            "average_weekly_turnover",
            "transaction_cost_drag",
        ):
            if not np.isclose(
                float(expected.iloc[0][metric]),
                float(actual.iloc[0][metric]),
                atol=1e-12,
                rtol=0.0,
            ):
                raise ValueError(
                    f"PR 15 does not reconcile to PR 14: alpha={alpha}, "
                    f"metric={metric}"
                )
    expected_targets = pd.read_parquet(
        pr14_dir / "raw_and_executed_targets.parquet"
    )
    expected_targets = expected_targets[
        expected_targets["candidate"] == "five_seed_mean_weight_ensemble"
    ].sort_values(["alpha", "date", "decision_step", "ticker"])
    actual_targets = result.target_sequences[
        (result.target_sequences["diagnostic"] == "dynamic_live")
        & result.target_sequences["alpha"].notna()
    ].sort_values(["alpha", "date", "decision_step", "ticker"])
    if len(expected_targets) != len(actual_targets):
        raise ValueError("PR 14 and PR 15 target audit lengths differ")
    for expected_column, actual_column in (
        ("raw_policy_target", "dynamic_raw_target"),
        ("executed_target", "dynamic_executed_target"),
    ):
        if not np.array_equal(
            expected_targets[expected_column].to_numpy(dtype=np.float64),
            actual_targets[actual_column].to_numpy(dtype=np.float64),
        ):
            raise ValueError(
                f"PR 15 target audit does not reconcile: {actual_column}"
            )


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
    return {"path": _manifest_path(path, root), "sha256": _sha256(path)}


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

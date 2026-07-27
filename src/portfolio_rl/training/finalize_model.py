"""Package one validated Phase 3 checkpoint without accessing the test split."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

_MODEL_VERSION_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(frozen=True)
class FinalizedModel:
    """Paths and identity for a completed final-model package."""

    output_dir: Path
    model_version: str
    configuration_id: str
    run_id: str
    representative_seed: int


def finalize_selected_model(
    *,
    selected_configuration_path: str | Path,
    registry_path: str | Path,
    model_version: str,
    representative_seed: int,
    output_root: str | Path = "artifacts/final_model",
    root: str | Path = ".",
    evaluation_config_path: str | Path = "configs/evaluation.yaml",
    universe_config_path: str | Path = "configs/universe.yaml",
) -> FinalizedModel:
    """Validate and atomically package a representative selected checkpoint."""
    if not _MODEL_VERSION_PATTERN.fullmatch(model_version):
        raise ValueError(
            "model_version must contain only letters, numbers, '.', '_', or '-'"
        )

    repository_root = Path(root).resolve()
    selected_path = _resolve(repository_root, selected_configuration_path)
    registry_csv = _resolve(repository_root, registry_path)
    output_base = _resolve(repository_root, output_root)
    destination = output_base / model_version
    if destination.exists():
        raise FileExistsError(f"Final model package already exists: {destination}")

    selected = _read_json(selected_path)
    experiment_name = _required_text(selected, "experiment_name")
    configuration_id = _required_text(selected, "configuration_id")
    _require_validation_only(selected, selected_path)
    if representative_seed not in {
        int(seed) for seed in selected.get("eligible_seeds", [])
    }:
        raise ValueError(
            f"Seed {representative_seed} is not eligible for the selected configuration"
        )
    gate_results = selected.get("gate_results")
    if (
        not isinstance(gate_results, dict)
        or not gate_results
        or not all(value is True for value in gate_results.values())
    ):
        raise ValueError("Selected configuration does not pass every selection gate")

    registry = pd.read_csv(registry_csv)
    matches = registry.loc[
        (registry["experiment_name"] == experiment_name)
        & (registry["seed"] == representative_seed)
    ]
    if len(matches) != 1:
        raise ValueError(
            "Registry must contain exactly one row for the representative seed"
        )
    registry_row = matches.iloc[0]
    if not _is_true(registry_row["selection_eligible"]):
        raise ValueError("Representative seed is not selection eligible")
    if registry_row["selection_checkpoint"] != "best_checkpoint":
        raise ValueError("Representative seed must use the best validation checkpoint")

    run_id = str(registry_row["run_id"])
    model_path = _resolve(repository_root, str(registry_row["selection_model_path"]))
    run_dir = model_path.parent
    run_manifest_path = run_dir / "manifest.json"
    run_manifest = _read_json(run_manifest_path)
    if run_manifest.get("run_id") != run_id:
        raise ValueError("Run manifest and registry run_id do not match")
    if int(run_manifest.get("seed", -1)) != representative_seed:
        raise ValueError("Run manifest and representative seed do not match")

    evaluation_path = _resolve(repository_root, evaluation_config_path)
    universe_path = _resolve(repository_root, universe_config_path)
    _require_source_hash(selected, "evaluation_config", evaluation_path)
    _verify_run_snapshot_hashes(run_dir, run_manifest)

    selection_dir = selected_path.parent
    artifact_root = selection_dir.parent.parent
    robustness_dir = artifact_root / "robustness" / experiment_name
    diagnostics_dir = artifact_root / "diagnostics" / experiment_name
    sensitivity_dir = artifact_root / "sensitivity" / experiment_name
    statistical_dir = artifact_root / "statistical_validation" / experiment_name

    robustness = _read_json(robustness_dir / "robustness_manifest.json")
    diagnostics = _read_json(diagnostics_dir / "allocation_summary.json")
    sensitivity = _read_json(sensitivity_dir / "sensitivity_manifest.json")
    bootstrap = _read_json(statistical_dir / "bootstrap_summary.json")
    for artifact, path in [
        (robustness, robustness_dir / "robustness_manifest.json"),
        (diagnostics, diagnostics_dir / "allocation_summary.json"),
        (sensitivity, sensitivity_dir / "sensitivity_manifest.json"),
        (bootstrap, statistical_dir / "bootstrap_summary.json"),
    ]:
        _require_validation_only(artifact, path)

    for artifact_name, artifact in [
        ("robustness", robustness),
        ("diagnostics", diagnostics),
        ("sensitivity", sensitivity),
    ]:
        if (
            artifact.get("experiment_name") != experiment_name
            or artifact.get("configuration_id") != configuration_id
        ):
            raise ValueError(
                f"{artifact_name} artifact does not match selected configuration"
            )

    robustness_checks = robustness.get("diagnostics")
    if not isinstance(robustness_checks, dict) or not all(
        value is True for value in robustness_checks.values()
    ):
        raise ValueError("Robustness diagnostics are incomplete or failing")
    if diagnostics.get("all_validation_metrics_reconciled") is not True:
        raise ValueError("Policy diagnostics do not reconcile to validation metrics")
    if sensitivity.get("all_observed_actions_reconciled") is not True:
        raise ValueError("Sensitivity replay does not reconcile observed actions")
    interpretation = sensitivity.get("campaign_interpretation", {})
    if (
        not isinstance(interpretation, dict)
        or interpretation.get("stop_before_packaging") is not False
    ):
        raise ValueError("Sensitivity diagnostics block final packaging")

    model_hash = _sha256_file(model_path)
    _require_selected_model_hash(
        diagnostics, representative_seed, model_hash, "diagnostics"
    )
    _require_selected_model_hash(
        sensitivity, representative_seed, model_hash, "sensitivity"
    )
    _require_robustness_model_hash(robustness, representative_seed, model_hash)

    source_files = {
        "model.zip": model_path,
        "data.yaml": run_dir / "config.yaml",
        "env.yaml": run_dir / "env.yaml",
        "train_ppo.yaml": run_dir / "train_ppo.yaml",
        "feature_spec_v1.json": run_dir / "feature_spec_v1.json",
        "universe.yaml": universe_path,
        "evaluation.yaml": evaluation_path,
        "validation_metrics.json": run_dir / "best_metrics_validation.json",
        "selected_configuration.json": selected_path,
        "validation_selection_report.md": (
            selection_dir / "validation_selection_report.md"
        ),
        "robustness_manifest.json": robustness_dir / "robustness_manifest.json",
        "robustness_report.md": robustness_dir / "robustness_report.md",
        "diagnostics_summary.json": diagnostics_dir / "allocation_summary.json",
        "diagnostics_report.md": diagnostics_dir / "diagnostics_report.md",
        "sensitivity_manifest.json": sensitivity_dir / "sensitivity_manifest.json",
        "sensitivity_report.md": sensitivity_dir / "sensitivity_report.md",
        "bootstrap_summary.json": statistical_dir / "bootstrap_summary.json",
        "bootstrap_report.md": statistical_dir / "bootstrap_report.md",
    }
    for source in source_files.values():
        if not source.is_file():
            raise FileNotFoundError(f"Required packaging artifact is missing: {source}")

    validation_metrics = _validation_metrics(registry_row)
    selected_model = {
        "schema_version": 1,
        "model_version": model_version,
        "packaged_at": datetime.now(UTC).isoformat(),
        "configuration_id": configuration_id,
        "experiment_name": experiment_name,
        "representative_seed": representative_seed,
        "representative_seed_policy": "cross_seed_median_primary_metrics",
        "representative_seed_rationale": (
            "Seed 42 matches the five-seed median total return, Sharpe ratio, "
            "and maximum drawdown for the selected configuration."
        ),
        "run_id": run_id,
        "selection_checkpoint": "best_checkpoint",
        "source_model": {
            "path": _relative_or_absolute(model_path, repository_root),
            "sha256": model_hash,
        },
        "feature_version": str(registry_row["feature_version"]),
        "validation_metrics": validation_metrics,
        "validation_only": True,
        "test_split_used": False,
        "final_test_status": "not_run",
    }
    model_card = _model_card(
        selected_model=selected_model,
        selected=selected,
        diagnostics=diagnostics,
        sensitivity=sensitivity,
        bootstrap=bootstrap,
    )

    output_base.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{model_version}.", dir=output_base))
    try:
        for packaged_name, source in source_files.items():
            shutil.copy2(source, temporary / packaged_name)
        (temporary / "selected_model.json").write_text(
            json.dumps(selected_model, indent=2, sort_keys=True, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
        (temporary / "selected_model_card.md").write_text(model_card, encoding="utf-8")
        payload_files = sorted(
            path for path in temporary.iterdir() if path.name != "manifest.json"
        )
        package_manifest = {
            "schema_version": 1,
            "model_version": model_version,
            "created_at": selected_model["packaged_at"],
            "configuration_id": configuration_id,
            "experiment_name": experiment_name,
            "run_id": run_id,
            "representative_seed": representative_seed,
            "validation_only": True,
            "test_split_used": False,
            "final_test_status": "not_run",
            "files": [
                {
                    "path": path.name,
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256_file(path),
                }
                for path in payload_files
            ],
        }
        (temporary / "manifest.json").write_text(
            json.dumps(package_manifest, indent=2, sort_keys=True, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    return FinalizedModel(
        output_dir=destination,
        model_version=model_version,
        configuration_id=configuration_id,
        run_id=run_id,
        representative_seed=representative_seed,
    )


def _model_card(
    *,
    selected_model: dict[str, Any],
    selected: dict[str, Any],
    diagnostics: dict[str, Any],
    sensitivity: dict[str, Any],
    bootstrap: dict[str, Any],
) -> str:
    metrics = selected_model["validation_metrics"]
    equal_weight = selected.get("baseline_deltas", {}).get("equal_weight_weekly", {})
    triggered_warnings = [
        name
        for name, details in diagnostics.get("campaign_warnings", {}).items()
        if isinstance(details, dict) and details.get("triggered") is True
    ]
    bootstrap_warnings = bootstrap.get("warnings", [])
    return "\n".join(
        [
            f"# Model Card — {selected_model['model_version']}",
            "",
            "## Candidate",
            "",
            f"- Configuration: `{selected_model['configuration_id']}`",
            f"- Representative seed: `{selected_model['representative_seed']}`",
            f"- Run: `{selected_model['run_id']}`",
            "- Checkpoint: best validation checkpoint",
            "- Feature contract: `v1`",
            "",
            (
                f"Seed {selected_model['representative_seed']} is a reproducible "
                "representative checkpoint whose primary "
                "validation metrics match the five-seed medians. It is not an ensemble."
            ),
            "",
            "## Validation results",
            "",
            f"- Total return: {metrics['total_return']:.4%}",
            f"- Sharpe ratio: {metrics['sharpe_ratio']:.4f}",
            f"- Maximum drawdown: {metrics['max_drawdown']:.4%}",
            f"- Average weekly turnover: {metrics['average_weekly_turnover']:.4%}",
            f"- Transaction-cost drag: {metrics['transaction_cost_drag']:.4%}",
            "",
            (
                "Relative to weekly equal weight, the selected configuration's "
                f"five-seed median total return delta was "
                f"{float(equal_weight.get('total_return', 0.0)):.4%} and its Sharpe "
                f"delta was {float(equal_weight.get('sharpe_ratio', 0.0)):.4f}."
            ),
            "",
            "## Robustness and behavior review",
            "",
            "- Transaction-cost and named-regime robustness checks passed.",
            "- Validation metrics and deterministic action replays reconciled.",
            (
                f"- Sensitivity recommendation: "
                f"`{sensitivity['campaign_interpretation']['recommendation']}`."
            ),
            "- Triggered diagnostic warnings: "
            + (", ".join(triggered_warnings) if triggered_warnings else "none")
            + ".",
            "",
            "## Intended use and limitations",
            "",
            (
                "This is a research candidate for controlled final holdout evaluation, "
                "not authorization for live trading. Results are based on one validation "
                "market path; all five policy seeds share that path. Selection and "
                "checkpoint choice used validation data, so the reported performance is "
                "selection-biased. Allocation behavior must be reviewed alongside return "
                "metrics and transaction costs."
            ),
            "",
            (
                "The test split has not been accessed by this packaging workflow. Final "
                "test status: `not_run`."
            ),
            "",
            "Bootstrap cautions:",
            *[f"- {warning}" for warning in bootstrap_warnings],
            "",
        ]
    )


def _validation_metrics(row: pd.Series) -> dict[str, float]:
    columns = {
        "total_return": "selection_validation_total_return",
        "sharpe_ratio": "selection_validation_sharpe_ratio",
        "max_drawdown": "selection_validation_max_drawdown",
        "average_weekly_turnover": ("selection_validation_average_weekly_turnover"),
        "transaction_cost_drag": ("selection_validation_transaction_cost_drag"),
    }
    metrics = {name: float(row[column]) for name, column in columns.items()}
    if not all(pd.notna(value) for value in metrics.values()):
        raise ValueError("Representative seed has incomplete validation metrics")
    return metrics


def _require_validation_only(artifact: dict[str, Any], path: Path) -> None:
    if artifact.get("test_split_used") is not False:
        raise ValueError(f"Artifact is not test-free: {path}")
    if "validation_only" in artifact and artifact["validation_only"] is not True:
        raise ValueError(f"Artifact is not validation-only: {path}")


def _require_selected_model_hash(
    artifact: dict[str, Any],
    seed: int,
    model_hash: str,
    artifact_name: str,
) -> None:
    matches = [
        model
        for model in artifact.get("selected_models", [])
        if int(model.get("seed", -1)) == seed
    ]
    if len(matches) != 1 or matches[0].get("sha256") != model_hash:
        raise ValueError(
            f"{artifact_name} model provenance does not match packaged checkpoint"
        )


def _require_robustness_model_hash(
    artifact: dict[str, Any], seed: int, model_hash: str
) -> None:
    models = artifact.get("models", [])
    matches = [model for model in models if int(model.get("seed", -1)) == seed]
    if len(matches) != 1 or matches[0].get("model", {}).get("sha256") != model_hash:
        raise ValueError(
            "Robustness model provenance does not match packaged checkpoint"
        )


def _verify_run_snapshot_hashes(run_dir: Path, run_manifest: dict[str, Any]) -> None:
    snapshot_hashes = {
        "data_config_hash": run_dir / "config.yaml",
        "env_config_hash": run_dir / "env.yaml",
        "train_config_hash": run_dir / "train_ppo.yaml",
        "feature_spec_hash": run_dir / "feature_spec_v1.json",
    }
    for manifest_key, snapshot_path in snapshot_hashes.items():
        if run_manifest.get(manifest_key) != _sha256_file(snapshot_path):
            raise ValueError(f"Run snapshot hash mismatch: {snapshot_path}")


def _require_source_hash(
    artifact: dict[str, Any], source_name: str, source_path: Path
) -> None:
    source = artifact.get("sources", {}).get(source_name, {})
    if source.get("sha256") != _sha256_file(source_path):
        raise ValueError(f"Source hash mismatch: {source_path}")


def _required_text(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Required text field is missing: {key}")
    return value


def _is_true(value: Any) -> bool:
    return value is True or str(value).strip().lower() == "true"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required artifact is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def _resolve(root: Path, path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def _relative_or_absolute(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

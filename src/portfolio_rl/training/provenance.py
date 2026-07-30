"""Phase 3 campaign provenance and test-access auditing."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from portfolio_rl.config.loader import load_yaml

HASHED_RUN_ARTIFACTS = {
    "data_config_hash": "config.yaml",
    "data_quality_report_hash": "data_quality_report_v1.json",
    "env_config_hash": "env.yaml",
    "feature_spec_hash": "feature_spec_v1.json",
    "train_config_hash": "train_ppo.yaml",
}
MODEL_INVENTORY_COLUMNS = [
    "seed",
    "run_id",
    "selection_checkpoint",
    "model_path",
    "model_sha256",
    "manifest_path",
    "manifest_sha256",
    "training_git_commit",
    "feature_version",
    "total_timesteps",
]


class ProvenanceAuditError(ValueError):
    """Raised when a campaign cannot be frozen safely."""


@dataclass(frozen=True)
class CampaignAudit:
    """Validated in-memory evidence used to write a research freeze."""

    campaign_id: str
    model_inventory: list[dict[str, Any]]
    hash_inventory: dict[str, Any]
    test_access_audit: dict[str, Any]
    freeze_metadata: dict[str, Any]
    provenance_report: str


@dataclass(frozen=True)
class ResearchFreezeResult:
    """Identity and output path for an immutable research freeze."""

    output_dir: Path
    campaign_id: str
    campaign_test_free: bool
    phase3b_authorized: bool


def audit_phase3_campaign(
    *,
    config_path: str | Path,
    root: str | Path = ".",
) -> CampaignAudit:
    """Validate the frozen five-seed campaign without writing artifacts."""
    repository_root = Path(root).resolve()
    qualification_path = _resolve(repository_root, config_path)
    config = load_yaml(qualification_path)
    campaign_id = _required_text(config, "campaign_id")
    experiment_name = _required_text(config, "experiment_name")
    configuration_id = _required_text(config, "configuration_id")

    selected_path = _configured_path(
        repository_root, config, "selected_configuration_path"
    )
    registry_path = _configured_path(repository_root, config, "registry_path")
    artifact_root = _configured_path(repository_root, config, "artifact_root")
    _require_hash(
        selected_path,
        _required_text(config, "selected_configuration_sha256"),
        "selected configuration",
    )
    _require_hash(
        registry_path,
        _required_text(config, "registry_sha256"),
        "experiment registry",
    )
    selected = _read_json(selected_path)
    _require_selected_configuration(
        selected,
        experiment_name=experiment_name,
        configuration_id=configuration_id,
    )

    turnover = _required_mapping(config, "turnover_contract")
    training_git_commit = _required_text(turnover, "training_git_commit")
    turnover_source_path = _required_text(turnover, "source_path")
    turnover_source_hash = _git_file_hash(
        repository_root,
        training_git_commit,
        turnover_source_path,
    )
    expected_turnover_source_hash = _required_text(turnover, "source_sha256")
    if turnover_source_hash != expected_turnover_source_hash:
        raise ProvenanceAuditError(
            "Turnover implementation at the training commit does not match "
            "the frozen turnover-v2 contract"
        )

    registry = pd.read_csv(registry_path)
    configured_models = _required_list(config, "models")
    expected_seeds = [
        int(_required_value(model, "seed")) for model in configured_models
    ]
    if len(expected_seeds) != len(set(expected_seeds)):
        raise ProvenanceAuditError("Configured model seeds must be unique")
    if set(expected_seeds) != {int(seed) for seed in selected["eligible_seeds"]}:
        raise ProvenanceAuditError(
            "Configured seeds do not match selected eligible seeds"
        )

    model_inventory: list[dict[str, Any]] = []
    hash_runs: list[dict[str, Any]] = []
    selected_model_paths: set[str] = set()
    common_hashes: dict[str, set[str]] = {
        "data_config_hash": set(),
        "data_quality_report_hash": set(),
        "env_config_hash": set(),
        "feature_spec_hash": set(),
    }
    for configured_model in configured_models:
        model_record, hash_record = _audit_model(
            configured_model=configured_model,
            registry=registry,
            repository_root=repository_root,
            experiment_name=experiment_name,
            training_git_commit=training_git_commit,
            expected_feature_version=_required_text(config, "expected_feature_version"),
            expected_total_timesteps=int(
                _required_value(config, "expected_total_timesteps")
            ),
            expected_transaction_cost_bps=float(
                _required_value(turnover, "transaction_cost_bps")
            ),
        )
        model_inventory.append(model_record)
        hash_runs.append(hash_record)
        selected_model_paths.add(model_record["model_path"])
        for hash_name, observed_hashes in common_hashes.items():
            observed_hashes.add(hash_record[hash_name]["actual_sha256"])

    inconsistent = [
        name for name, observed in common_hashes.items() if len(observed) != 1
    ]
    if inconsistent:
        raise ProvenanceAuditError(
            "Frozen runs do not share common artifact hashes: "
            + ", ".join(inconsistent)
        )

    test_access_audit = _audit_test_access(
        config=config,
        repository_root=repository_root,
        artifact_root=artifact_root,
        experiment_name=experiment_name,
        configuration_id=configuration_id,
        selected_model_paths=selected_model_paths,
        selected_run_ids={record["run_id"] for record in model_inventory},
    )
    if not test_access_audit["campaign_test_free"]:
        raise ProvenanceAuditError(
            "A frozen turnover-v2 campaign model has accessed the test split"
        )
    if test_access_audit["undeclared_test_access"]:
        raise ProvenanceAuditError(
            "Undeclared repository test access was detected: "
            + ", ".join(test_access_audit["undeclared_test_access"])
        )

    development = _required_mapping(config, "development_period")
    frozen_at = datetime.now(UTC).isoformat()
    repository_baseline_commit = _required_text(config, "repository_baseline_commit")
    _require_git_commit(repository_root, repository_baseline_commit)
    freeze_metadata = {
        "schema_version": 1,
        "campaign_id": campaign_id,
        "experiment_name": experiment_name,
        "configuration_id": configuration_id,
        "frozen_at": frozen_at,
        "repository_baseline_commit": repository_baseline_commit,
        "audit_execution": _git_execution_state(repository_root),
        "qualification_config": _source_record(qualification_path, repository_root),
        "selected_configuration": _source_record(selected_path, repository_root),
        "registry": _source_record(registry_path, repository_root),
        "development_period": {
            "start_date": _required_text(development, "start_date"),
            "end_date": _required_text(development, "end_date"),
            "label": _required_text(development, "label"),
        },
        "turnover_contract": {
            "version": _required_text(turnover, "version"),
            "training_git_commit": training_git_commit,
            "source_path": turnover_source_path,
            "source_sha256": turnover_source_hash,
            "transaction_cost_bps": float(
                _required_value(turnover, "transaction_cost_bps")
            ),
        },
        "seeds": sorted(expected_seeds),
        "provenance_passed": True,
        "campaign_test_free": True,
        "legacy_project_test_access_detected": bool(
            test_access_audit["known_legacy_access"]
        ),
        "phase3b_authorized": False,
        "phase3b_block_reason": (
            "Legacy project-level test access exists; PM/ML approval of a new "
            "independent holdout is required."
        ),
    }
    hash_inventory = {
        "schema_version": 1,
        "campaign_id": campaign_id,
        "common_artifact_hashes": {
            name: next(iter(values)) for name, values in common_hashes.items()
        },
        "runs": hash_runs,
    }
    report = _provenance_report(
        freeze_metadata=freeze_metadata,
        model_inventory=model_inventory,
        test_access_audit=test_access_audit,
    )
    return CampaignAudit(
        campaign_id=campaign_id,
        model_inventory=model_inventory,
        hash_inventory=hash_inventory,
        test_access_audit=test_access_audit,
        freeze_metadata=freeze_metadata,
        provenance_report=report,
    )


def freeze_phase3_campaign(
    *,
    config_path: str | Path,
    output_root: str | Path = "artifacts/research_freeze",
    root: str | Path = ".",
) -> ResearchFreezeResult:
    """Audit and atomically write one immutable campaign research freeze."""
    repository_root = Path(root).resolve()
    audit = audit_phase3_campaign(config_path=config_path, root=repository_root)
    output_base = _resolve(repository_root, output_root)
    destination = output_base / audit.campaign_id
    if destination.exists():
        raise FileExistsError(f"Research freeze already exists: {destination}")

    output_base.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{audit.campaign_id}.", dir=output_base))
    try:
        _write_csv(temporary / "model_inventory.csv", audit.model_inventory)
        _write_json(temporary / "hash_inventory.json", audit.hash_inventory)
        _write_json(temporary / "test_access_audit.json", audit.test_access_audit)
        (temporary / "provenance_report.md").write_text(
            audit.provenance_report, encoding="utf-8"
        )
        payloads = sorted(temporary.iterdir())
        freeze_manifest = {
            **audit.freeze_metadata,
            "files": [
                {
                    "path": path.name,
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256_file(path),
                }
                for path in payloads
            ],
        }
        _write_json(temporary / "freeze_manifest.json", freeze_manifest)
        temporary.replace(destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    return ResearchFreezeResult(
        output_dir=destination,
        campaign_id=audit.campaign_id,
        campaign_test_free=True,
        phase3b_authorized=False,
    )


def _audit_model(
    *,
    configured_model: dict[str, Any],
    registry: pd.DataFrame,
    repository_root: Path,
    experiment_name: str,
    training_git_commit: str,
    expected_feature_version: str,
    expected_total_timesteps: int,
    expected_transaction_cost_bps: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    seed = int(_required_value(configured_model, "seed"))
    run_id = _required_text(configured_model, "run_id")
    checkpoint = _required_text(configured_model, "selection_checkpoint")
    model_path = _configured_path(repository_root, configured_model, "model_path")
    manifest_path = _configured_path(repository_root, configured_model, "manifest_path")
    expected_model_hash = _required_text(configured_model, "model_sha256")
    expected_manifest_hash = _required_text(configured_model, "manifest_sha256")
    _require_hash(model_path, expected_model_hash, f"seed {seed} model")
    _require_hash(manifest_path, expected_manifest_hash, f"seed {seed} manifest")

    matches = registry.loc[
        (registry["experiment_name"] == experiment_name) & (registry["seed"] == seed)
    ]
    if len(matches) != 1:
        raise ProvenanceAuditError(
            f"Registry must contain exactly one row for seed {seed}"
        )
    row = matches.iloc[0]
    expected_registry_values = {
        "run_id": run_id,
        "selection_checkpoint": checkpoint,
        "selection_model_path": _relative_path(model_path, repository_root),
        "manifest_path": _relative_path(manifest_path, repository_root),
    }
    for column, expected in expected_registry_values.items():
        if str(row[column]) != expected:
            raise ProvenanceAuditError(f"Registry {column} mismatch for seed {seed}")
    if not _is_true(row["selection_eligible"]):
        raise ProvenanceAuditError(f"Seed {seed} is not selection eligible")
    if checkpoint != "best_checkpoint":
        raise ProvenanceAuditError(f"Seed {seed} must use the frozen best checkpoint")

    manifest = _read_json(manifest_path)
    required_manifest_values = {
        "run_id": run_id,
        "seed": seed,
        "git_commit": training_git_commit,
        "feature_version": expected_feature_version,
        "total_timesteps": expected_total_timesteps,
    }
    for key, expected in required_manifest_values.items():
        if manifest.get(key) != expected:
            raise ProvenanceAuditError(f"Manifest {key} mismatch for seed {seed}")

    run_dir = manifest_path.parent
    hash_record: dict[str, Any] = {
        "seed": seed,
        "run_id": run_id,
        "model": _verified_hash_record(
            model_path, expected_model_hash, repository_root
        ),
        "manifest": _verified_hash_record(
            manifest_path, expected_manifest_hash, repository_root
        ),
    }
    for manifest_key, filename in HASHED_RUN_ARTIFACTS.items():
        artifact_path = run_dir / filename
        expected_hash = _required_text(manifest, manifest_key)
        hash_record[manifest_key] = _verified_hash_record(
            artifact_path, expected_hash, repository_root
        )

    env_config = _read_yaml(run_dir / "env.yaml")
    if float(env_config.get("transaction_cost_bps", -1.0)) != (
        expected_transaction_cost_bps
    ):
        raise ProvenanceAuditError(
            f"Transaction-cost configuration mismatch for seed {seed}"
        )
    if env_config.get("initial_weights") != "equal_weight":
        raise ProvenanceAuditError(
            f"Initial portfolio configuration mismatch for seed {seed}"
        )
    train_config = _read_yaml(run_dir / "train_ppo.yaml")
    if int(train_config.get("seed", -1)) != seed:
        raise ProvenanceAuditError(f"Training-config seed mismatch for seed {seed}")
    test_named_artifacts = sorted(
        path.name for path in run_dir.iterdir() if "test" in path.name.lower()
    )
    if test_named_artifacts:
        raise ProvenanceAuditError(
            f"Test-named artifacts found for seed {seed}: "
            + ", ".join(test_named_artifacts)
        )

    inventory = {
        "seed": seed,
        "run_id": run_id,
        "selection_checkpoint": checkpoint,
        "model_path": _relative_path(model_path, repository_root),
        "model_sha256": expected_model_hash,
        "manifest_path": _relative_path(manifest_path, repository_root),
        "manifest_sha256": expected_manifest_hash,
        "training_git_commit": training_git_commit,
        "feature_version": expected_feature_version,
        "total_timesteps": expected_total_timesteps,
    }
    return inventory, hash_record


def _audit_test_access(
    *,
    config: dict[str, Any],
    repository_root: Path,
    artifact_root: Path,
    experiment_name: str,
    configuration_id: str,
    selected_model_paths: set[str],
    selected_run_ids: set[str],
) -> dict[str, Any]:
    governance = _required_mapping(config, "test_governance")
    configured_legacy = _required_list(governance, "known_legacy_access")
    known_by_path: dict[str, dict[str, Any]] = {}
    for legacy in configured_legacy:
        metadata_path = _configured_path(repository_root, legacy, "metadata_path")
        relative_metadata_path = _relative_path(metadata_path, repository_root)
        _require_hash(
            metadata_path,
            _required_text(legacy, "metadata_sha256"),
            "known legacy test metadata",
        )
        payload = _read_json(metadata_path)
        if payload.get("split") != _required_text(legacy, "split"):
            raise ProvenanceAuditError(
                f"Known legacy split mismatch: {relative_metadata_path}"
            )
        if payload.get("model_path") != _required_text(legacy, "model_path"):
            raise ProvenanceAuditError(
                f"Known legacy model mismatch: {relative_metadata_path}"
            )
        known_by_path[relative_metadata_path] = legacy

    findings: list[dict[str, Any]] = []
    unreadable_json: list[str] = []
    for json_path in sorted(artifact_root.rglob("*.json")):
        try:
            payload = _read_json(json_path)
        except (json.JSONDecodeError, TypeError):
            unreadable_json.append(_relative_path(json_path, repository_root))
            continue
        if not _declares_test_access(payload):
            continue
        relative_json_path = _relative_path(json_path, repository_root)
        serialized = json.dumps(payload, sort_keys=True)
        model_path = str(payload.get("model_path", ""))
        current_campaign = (
            payload.get("experiment_name") == experiment_name
            or payload.get("configuration_id") == configuration_id
            or model_path in selected_model_paths
            or any(run_id in serialized for run_id in selected_run_ids)
        )
        findings.append(
            {
                "metadata_path": relative_json_path,
                "metadata_sha256": _sha256_file(json_path),
                "split": payload.get("split"),
                "test_split_used": payload.get("test_split_used"),
                "final_test_status": payload.get("final_test_status"),
                "model_path": payload.get("model_path"),
                "created_at": payload.get("created_at"),
                "confirm_final_test": payload.get("confirm_final_test"),
                "current_campaign": current_campaign,
                "declared_legacy": relative_json_path in known_by_path,
            }
        )

    observed_paths = {finding["metadata_path"] for finding in findings}
    missing_legacy = sorted(set(known_by_path) - observed_paths)
    if missing_legacy:
        raise ProvenanceAuditError(
            "Declared legacy test evidence was not found: " + ", ".join(missing_legacy)
        )
    campaign_findings = [finding for finding in findings if finding["current_campaign"]]
    known_legacy_findings = [
        finding for finding in findings if finding["declared_legacy"]
    ]
    undeclared = sorted(
        finding["metadata_path"]
        for finding in findings
        if not finding["current_campaign"] and not finding["declared_legacy"]
    )
    return {
        "schema_version": 1,
        "campaign_id": _required_text(config, "campaign_id"),
        "audit_policy": _required_text(governance, "phase3b_policy"),
        "campaign_test_free": not campaign_findings,
        "project_test_history_clear": not findings,
        "phase3b_authorized": False,
        "holdout_status": "new_independent_holdout_requires_pm_ml_approval",
        "campaign_test_access": campaign_findings,
        "known_legacy_access": known_legacy_findings,
        "undeclared_test_access": undeclared,
        "unreadable_json": unreadable_json,
    }


def _declares_test_access(payload: dict[str, Any]) -> bool:
    split = str(payload.get("split", "")).strip().lower()
    final_status = str(payload.get("final_test_status", "")).strip().lower()
    return (
        split == "test"
        or payload.get("test_split_used") is True
        or final_status not in {"", "not_run"}
    )


def _provenance_report(
    *,
    freeze_metadata: dict[str, Any],
    model_inventory: list[dict[str, Any]],
    test_access_audit: dict[str, Any],
) -> str:
    lines = [
        "# Phase 3 Research Freeze and Provenance Report",
        "",
        f"Campaign: `{freeze_metadata['campaign_id']}`",
        f"Configuration: `{freeze_metadata['configuration_id']}`",
        (
            "Development-data status: "
            f"**{freeze_metadata['development_period']['label']}**."
        ),
        "",
        "## Frozen Models",
        "",
        "| Seed | Run ID | Checkpoint | Model SHA-256 |",
        "| ---: | --- | --- | --- |",
    ]
    lines.extend(
        f"| {record['seed']} | `{record['run_id']}` | "
        f"`{record['selection_checkpoint']}` | `{record['model_sha256']}` |"
        for record in model_inventory
    )
    legacy = test_access_audit["known_legacy_access"]
    lines.extend(
        [
            "",
            "## Audit Result",
            "",
            "- Five selected checkpoints are present and selection eligible.",
            "- Run snapshots and frozen hashes reconcile.",
            "- All runs use the frozen turnover-v2 training commit and source hash.",
            "- Frozen turnover-v2 campaign test access: **none detected**.",
            (
                "- Legacy project-level test access: "
                f"**{len(legacy)} declared record(s) detected**."
            ),
            "- Phase 3B authorization: **blocked**.",
            "",
            (
                "The legacy test record belongs to a different Phase 2 model. It does "
                "not taint the turnover-v2 campaign directly, but 2025+ cannot be "
                "described as a globally untouched holdout. PM/ML reviewers must "
                "approve a new independent holdout before Phase 3B."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _git_file_hash(root: Path, commit: str, source_path: str) -> str:
    _require_git_commit(root, commit)
    result = subprocess.run(
        ["git", "show", f"{commit}:{source_path}"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    return hashlib.sha256(result.stdout).hexdigest()


def _require_git_commit(root: Path, commit: str) -> None:
    try:
        subprocess.run(
            ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
            cwd=root,
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as exc:
        raise ProvenanceAuditError(
            f"Required Git commit is unavailable: {commit}"
        ) from exc


def _git_execution_state(root: Path) -> dict[str, Any]:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return {"head_commit": head, "worktree_dirty": bool(status.strip())}


def _verified_hash_record(path: Path, expected_hash: str, root: Path) -> dict[str, Any]:
    _require_hash(path, expected_hash, _relative_path(path, root))
    return {
        "path": _relative_path(path, root),
        "expected_sha256": expected_hash,
        "actual_sha256": expected_hash,
        "matches": True,
    }


def _require_selected_configuration(
    selected: dict[str, Any],
    *,
    experiment_name: str,
    configuration_id: str,
) -> None:
    if selected.get("experiment_name") != experiment_name:
        raise ProvenanceAuditError("Selected experiment name does not match")
    if selected.get("configuration_id") != configuration_id:
        raise ProvenanceAuditError("Selected configuration ID does not match")
    if selected.get("validation_only") is not True:
        raise ProvenanceAuditError("Selected configuration is not validation-only")
    if selected.get("test_split_used") is not False:
        raise ProvenanceAuditError("Selected configuration used the test split")
    gates = selected.get("gate_results")
    if (
        not isinstance(gates, dict)
        or not gates
        or not all(value is True for value in gates.values())
    ):
        raise ProvenanceAuditError("Selected configuration has failing gates")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=MODEL_INVENTORY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required JSON artifact is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required YAML artifact is missing: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected YAML mapping: {path}")
    return payload


def _required_mapping(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise TypeError(f"Required mapping is missing: {key}")
    return value


def _required_list(payload: dict[str, Any], key: str) -> list[dict[str, Any]]:
    value = payload.get(key)
    if not isinstance(value, list) or not value:
        raise TypeError(f"Required non-empty list is missing: {key}")
    if not all(isinstance(item, dict) for item in value):
        raise TypeError(f"List entries must be mappings: {key}")
    return value


def _required_text(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Required text value is missing: {key}")
    return value


def _required_value(payload: dict[str, Any], key: str) -> Any:
    if key not in payload:
        raise ValueError(f"Required value is missing: {key}")
    return payload[key]


def _configured_path(root: Path, payload: dict[str, Any], key: str) -> Path:
    return _resolve(root, _required_text(payload, key))


def _source_record(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": _relative_path(path, root),
        "sha256": _sha256_file(path),
    }


def _require_hash(path: Path, expected_hash: str, name: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required {name} is missing: {path}")
    actual_hash = _sha256_file(path)
    if actual_hash != expected_hash:
        raise ProvenanceAuditError(
            f"SHA-256 mismatch for {name}: expected {expected_hash}, "
            f"observed {actual_hash}"
        )


def _relative_path(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _resolve(root: Path, path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def _is_true(value: Any) -> bool:
    return value is True or str(value).strip().lower() == "true"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

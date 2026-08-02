"""Freeze one executable Phase 3 candidate without accessing a final holdout."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from portfolio_rl.policies.ensemble_policy import MeanWeightEnsemblePolicy
from portfolio_rl.policies.overlays import PartialRebalancePolicy
from portfolio_rl.policies.sb3_policy import load_sb3_weight_policy

EXPECTED_SEEDS = (7, 42, 101, 202, 999)
EXPECTED_ASSET_ORDER = (
    "SPY",
    "QQQ",
    "IWM",
    "EFA",
    "EEM",
    "TLT",
    "IEF",
    "SHY",
    "LQD",
    "HYG",
    "GLD",
    "DBC",
    "VNQ",
    "XLU",
)
MODEL_VERSION_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
BLOCKED_GOVERNANCE = {
    "phase3b_status": "blocked",
    "existing_test_designation": "2025+",
    "existing_test_independent": False,
    "block_reason": "Recorded legacy access before final candidate freeze",
    "replacement_holdout_required": True,
    "pm_ml_approval_required": True,
}
GENERATED_FILENAMES = (
    "frozen_candidate.json",
    "acceptance_criteria.yaml",
    "member_models.json",
    "model_hashes.json",
    "data_and_feature_hashes.json",
    "environment_and_training_hashes.json",
    "baseline_definitions.json",
    "candidate_construction.md",
    "evidence_summary.md",
    "known_limitations.md",
    "PM_review_packet.md",
    "commands.md",
    "test_access_audit.json",
    "README.md",
)


class PretestFreezeError(ValueError):
    """Raised when a candidate cannot be frozen or verified safely."""


@dataclass(frozen=True)
class FrozenCandidate:
    """Executable identity of the pre-test candidate."""

    model_version: str
    member_seeds: tuple[int, ...]
    member_model_paths: tuple[Path, ...]
    member_model_hashes: tuple[str, ...]
    action_temperatures: tuple[float, ...]
    partial_rebalance_alpha: float
    initial_portfolio: str
    asset_order: tuple[str, ...]
    feature_version: str
    feature_spec_hash: str
    environment_config_hash: str
    transaction_cost_bps: float
    rebalance_frequency_trading_days: int


@dataclass(frozen=True)
class AcceptanceCriteria:
    """Frozen final-holdout decision rules."""

    primary_hurdle: str
    hard_gates: Mapping[str, Any]
    advisory_diagnostics: tuple[str, ...]
    secondary_comparisons: tuple[str, ...]
    approval_status: str


def freeze_final_candidate(
    *,
    repository_root: Path,
    research_freeze_path: Path,
    ensemble_manifest_path: Path,
    walk_forward_results_path: Path,
    execution_stress_path: Path,
    regime_attribution_path: Path,
    acceptance_config_path: Path,
    model_version: str,
    output_root: Path,
) -> Path:
    """Verify all inputs and atomically write one immutable pre-test freeze."""
    if not MODEL_VERSION_PATTERN.fullmatch(model_version):
        raise ValueError("invalid model_version")
    root = repository_root.resolve()
    acceptance_path = _resolve(root, acceptance_config_path)
    config = _read_yaml(acceptance_path)
    _validate_acceptance_config(config, model_version)
    _require_clean_tracked_worktree(root)

    configured_sources = _verify_evidence_sources(root, config)
    supplied = {
        "research_freeze": _resolve(root, research_freeze_path),
        "ensemble": _resolve(root, ensemble_manifest_path),
        "walk_forward": _resolve(root, walk_forward_results_path),
        "execution_stress": _resolve(root, execution_stress_path),
        "regime_attribution": _resolve(root, regime_attribution_path),
    }
    for name, path in supplied.items():
        if configured_sources[name]["path"] != _relative(path, root):
            raise PretestFreezeError(f"{name} path does not match frozen config")

    research = _read_json(supplied["research_freeze"])
    ensemble = _read_json(supplied["ensemble"])
    walk_forward = _read_json(supplied["walk_forward"])
    execution = _read_json(supplied["execution_stress"])
    attribution = _read_json(supplied["regime_attribution"])
    _validate_evidence_manifests(research, ensemble, walk_forward, execution, attribution)

    members, feature_spec, environment = _verify_members(root, ensemble)
    candidate = _candidate_payload(
        root=root,
        model_version=model_version,
        members=members,
        feature_spec=feature_spec,
        environment=environment,
    )
    acceptance = _acceptance_criteria(config)
    audit = _audit_test_access(root, research, members)
    git_state = _git_state(root)
    candidate["created_at"] = datetime.now(UTC).isoformat()
    candidate["git_commit"] = git_state["head_commit"]
    candidate["git_state"] = git_state

    destination = _resolve(root, output_root) / model_version
    if destination.exists():
        raise FileExistsError(f"pre-test freeze already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{model_version}.", dir=destination.parent)
    )
    try:
        _write_json(temporary / "frozen_candidate.json", candidate)
        shutil.copy2(acceptance_path, temporary / "acceptance_criteria.yaml")
        _write_json(
            temporary / "member_models.json",
            {"schema_version": 1, "member_seed_order": list(EXPECTED_SEEDS), "members": members},
        )
        _write_json(
            temporary / "model_hashes.json",
            {
                "schema_version": 1,
                "hash_algorithm": "sha256",
                "models": [
                    {
                        "seed": member["seed"],
                        "path": member["model_path"],
                        "sha256": member["model_sha256"],
                        "size_bytes": member["model_size_bytes"],
                    }
                    for member in members
                ],
            },
        )
        _write_json(
            temporary / "data_and_feature_hashes.json",
            _data_and_feature_payload(ensemble, feature_spec),
        )
        _write_json(
            temporary / "environment_and_training_hashes.json",
            _environment_and_training_payload(root, members, environment),
        )
        _write_json(temporary / "baseline_definitions.json", _baseline_definitions())
        (temporary / "candidate_construction.md").write_text(
            _candidate_construction(candidate), encoding="utf-8"
        )
        evidence_summary = _evidence_summary(root, configured_sources)
        (temporary / "evidence_summary.md").write_text(
            evidence_summary, encoding="utf-8"
        )
        (temporary / "known_limitations.md").write_text(
            _known_limitations(), encoding="utf-8"
        )
        (temporary / "PM_review_packet.md").write_text(
            build_pm_review_packet(
                evidence_paths={
                    name: root / record["path"]
                    for name, record in configured_sources.items()
                },
                candidate=_candidate_from_payload(candidate),
                acceptance_criteria=acceptance,
            ),
            encoding="utf-8",
        )
        (temporary / "commands.md").write_text(
            _commands(model_version), encoding="utf-8"
        )
        _write_json(temporary / "test_access_audit.json", audit)
        (temporary / "README.md").write_text(
            _package_readme(model_version), encoding="utf-8"
        )

        payload_records = [_file_record(temporary / name) for name in GENERATED_FILENAMES]
        input_records = [
            {"name": name, **record} for name, record in configured_sources.items()
        ]
        input_records.append(
            {
                "name": "acceptance_config",
                "path": _relative(acceptance_path, root),
                "sha256": _sha256(acceptance_path),
                "size_bytes": acceptance_path.stat().st_size,
            }
        )
        manifest = {
            "schema_version": 1,
            "model_version": model_version,
            "package_role": "final_test_candidate",
            "created_at": candidate["created_at"],
            "git_commit": candidate["git_commit"],
            "candidate_campaign_test_accessed": False,
            "project_test_history_clear": False,
            "phase3b_authorized": False,
            "governance": BLOCKED_GOVERNANCE,
            "inputs": sorted(input_records, key=lambda record: record["name"]),
            "files": payload_records,
            "self_hash_contract": "sha256_of_canonical_json_without_manifest_payload_sha256",
        }
        manifest["manifest_payload_sha256"] = _logical_json_sha256(manifest)
        _write_json(temporary / "freeze_manifest.json", manifest)
        temporary.replace(destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


def verify_frozen_candidate(frozen_candidate_path: Path) -> FrozenCandidate:
    """Verify a complete freeze package and return its executable identity."""
    candidate_path = frozen_candidate_path.resolve()
    if candidate_path.name != "frozen_candidate.json":
        raise PretestFreezeError("candidate path must name frozen_candidate.json")
    package_dir = candidate_path.parent
    manifest_path = package_dir / "freeze_manifest.json"
    manifest = _read_json(manifest_path)
    expected_self_hash = _required_text(manifest, "manifest_payload_sha256")
    self_payload = dict(manifest)
    del self_payload["manifest_payload_sha256"]
    if _logical_json_sha256(self_payload) != expected_self_hash:
        raise PretestFreezeError("freeze manifest payload hash mismatch")
    expected_names = set(GENERATED_FILENAMES)
    records = manifest.get("files")
    if not isinstance(records, list):
        raise PretestFreezeError("freeze manifest files must be a list")
    observed_names = {str(record.get("path")) for record in records}
    if observed_names != expected_names:
        raise PretestFreezeError("freeze manifest does not cover every generated file")
    for record in records:
        path = package_dir / str(record["path"])
        _require_hash(path, str(record["sha256"]), "generated freeze artifact")

    payload = _read_json(candidate_path)
    root = _git_root(package_dir)
    if payload.get("model_version") != manifest.get("model_version"):
        raise PretestFreezeError("candidate and manifest model versions differ")
    if payload.get("governance") != BLOCKED_GOVERNANCE:
        raise PretestFreezeError("candidate governance contract mismatch")
    if manifest.get("governance") != BLOCKED_GOVERNANCE:
        raise PretestFreezeError("manifest governance contract mismatch")
    input_records = manifest.get("inputs", [])
    for record in input_records:
        _require_hash(
            _resolve(root, str(record["path"])),
            str(record["sha256"]),
            f"frozen input {record['name']}",
        )
    candidate = _candidate_from_payload(payload)
    member_payload = _read_json(package_dir / "member_models.json")
    member_records = member_payload.get("members")
    if not isinstance(member_records, list) or len(member_records) != 5:
        raise PretestFreezeError("member_models.json must contain five members")
    if tuple(member_payload.get("member_seed_order", [])) != EXPECTED_SEEDS:
        raise PretestFreezeError("member_models.json seed order mismatch")
    for path, expected in zip(
        candidate.member_model_paths,
        candidate.member_model_hashes,
        strict=True,
    ):
        _require_hash(_resolve(root, path), expected, "frozen member model")
    for member in member_records:
        run_manifest_path = _resolve(root, str(member["run_manifest_path"]))
        _require_hash(
            run_manifest_path,
            str(member["run_manifest_sha256"]),
            "member run manifest",
        )
        run_manifest = _read_json(run_manifest_path)
        for manifest_key, filename in (
            ("data_config_hash", "config.yaml"),
            ("data_quality_report_hash", "data_quality_report_v1.json"),
            ("env_config_hash", "env.yaml"),
            ("feature_spec_hash", "feature_spec_v1.json"),
            ("train_config_hash", "train_ppo.yaml"),
        ):
            _require_hash(
                run_manifest_path.parent / filename,
                str(run_manifest[manifest_key]),
                f"member {manifest_key}",
            )
    research_record = next(
        (record for record in input_records if record.get("name") == "research_freeze"),
        None,
    )
    if research_record is None:
        raise PretestFreezeError("freeze manifest is missing the research freeze input")
    current_audit = _audit_test_access(
        root,
        _read_json(_resolve(root, str(research_record["path"]))),
        member_records,
    )
    if current_audit != _read_json(package_dir / "test_access_audit.json"):
        raise PretestFreezeError("current test-access audit differs from frozen audit")
    return candidate


def load_frozen_candidate_policy(
    frozen_candidate_path: Path,
) -> PartialRebalancePolicy:
    """Load the hash-verified ordered ensemble and its frozen overlay."""
    candidate = verify_frozen_candidate(frozen_candidate_path)
    root = _git_root(frozen_candidate_path.resolve().parent)
    member_policies = {
        f"seed_{seed}": load_sb3_weight_policy(
            _resolve(root, model_path), action_temperature=temperature
        )
        for seed, model_path, temperature in zip(
            candidate.member_seeds,
            candidate.member_model_paths,
            candidate.action_temperatures,
            strict=True,
        )
    }
    return PartialRebalancePolicy(
        base_policy=MeanWeightEnsemblePolicy(member_policies=member_policies),
        alpha=candidate.partial_rebalance_alpha,
    )


def build_pm_review_packet(
    *,
    evidence_paths: Mapping[str, Path],
    candidate: FrozenCandidate,
    acceptance_criteria: AcceptanceCriteria,
) -> str:
    """Build the decision packet from already verified evidence paths."""
    del evidence_paths
    seeds = ", ".join(str(seed) for seed in candidate.member_seeds)
    secondary = ", ".join(acceptance_criteria.secondary_comparisons)
    return f"""# PM Review Packet — Final Candidate Pre-Test Freeze

## Decision

Freeze the ordered five-seed (`{seeds}`) mean-target PPO ensemble with a 25%
partial-rebalancing overlay, equal-weight initial endowment, five-trading-day
decisions, and 10 bps cost on half-L1 one-way turnover.

The selected candidate is a five-seed PPO ensemble implemented with 25% partial
rebalancing. Historical evidence suggests that ensembling reduces seed-specific
noise and that gradual execution materially improves turnover, cost resilience,
and repeated pseudo-out-of-sample performance. The policy should be interpreted
as a slow-moving adaptive allocator rather than a proven high-frequency tactical
market-timing system.

## Evidence reviewed

- 2024 is the consumed development/selection period, not independent validation.
- Seed-42 initialization sensitivity did not change the official equal-weight start.
- The executable ensemble improved on the median member in every walk-forward fold.
- Alpha 0.25 produced 3/4 positive active-return folds and the strongest repeated
  return/turnover evidence among the predeclared overlays.
- One-close delay and joint delayed asset-tier stress retained 3/4 positive folds.
- PR 15 replay diagnostics and WF4 attribution show weak dynamic timing evidence
  in 2022–2023; some benefit is attributable to strategic asset mix.

## Baselines and decision rules

Primary hurdle: `{acceptance_criteria.primary_hurdle}`.

Secondary comparisons: {secondary}. These comparisons are diagnostic and are not
requirements to beat every baseline on every metric.

The final result passes only if every hard gate in `acceptance_criteria.yaml`
passes. Advisory diagnostics cannot reverse a failed hard gate.

## Governance

No final-test data were used in this candidate's construction or selection. A
separate legacy Phase 2 model previously accessed the repository's existing
2025+ test designation. That designation is therefore not an untouched Phase 3
holdout. Phase 3B remains blocked pending PM/ML approval of a new independent
holdout. The generated final-test command is a non-executed template only.

## Approval status

Acceptance criteria: `{acceptance_criteria.approval_status}`.
"""


def _validate_acceptance_config(config: dict[str, Any], model_version: str) -> None:
    if config.get("schema_version") != 1 or config.get("model_version") != model_version:
        raise PretestFreezeError("acceptance config identity mismatch")
    candidate = _mapping(config, "candidate")
    required = {
        "policy_type": "five_seed_mean_target_ensemble",
        "seeds": list(EXPECTED_SEEDS),
        "ensemble_rule": "arithmetic_mean_of_member_target_weights",
        "partial_rebalance_alpha": 0.25,
        "initial_portfolio": "equal_weight",
        "rebalance_frequency_trading_days": 5,
        "transaction_cost_bps": 10.0,
        "turnover_definition": "half_l1_one_way",
        "long_only": True,
        "fully_invested": True,
    }
    for key, expected in required.items():
        if candidate.get(key) != expected:
            raise PretestFreezeError(f"candidate.{key} must equal {expected!r}")
    governance = _mapping(config, "governance")
    if governance.get("phase3b_authorized") is not False:
        raise PretestFreezeError("PR 19 must not authorize Phase 3B")
    for key, expected in BLOCKED_GOVERNANCE.items():
        if governance.get(key) != expected:
            raise PretestFreezeError(f"governance.{key} must equal {expected!r}")
    hard = _mapping(config, "hard_gates")
    for key in (
        "positive_active_return_vs_primary",
        "positive_sharpe_difference_vs_primary",
        "require_finite_metrics",
        "require_positive_final_nav",
        "require_no_artifact_hash_mismatch",
        "require_no_test_overwrite",
    ):
        if hard.get(key) is not True:
            raise PretestFreezeError(f"hard gate must remain enabled: {key}")
    if not config.get("advisory_diagnostics") or not config.get("secondary_comparisons"):
        raise PretestFreezeError("advisory and secondary sections must be non-empty")


def _verify_evidence_sources(root: Path, config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    sources = _mapping(config, "evidence_sources")
    required_names = {
        "research_freeze",
        "initialization_sensitivity",
        "ensemble",
        "turnover_overlay",
        "dynamic_value",
        "walk_forward",
        "walk_forward_metrics",
        "execution_stress",
        "regime_attribution",
    }
    if set(sources) != required_names:
        raise PretestFreezeError("evidence source set is incomplete")
    verified = {}
    for name in sources:
        record = _mapping(sources, name)
        path = _resolve(root, _required_text(record, "path"))
        expected = _required_text(record, "sha256")
        _require_hash(path, expected, name)
        verified[name] = {
            "path": _relative(path, root),
            "sha256": expected,
            "size_bytes": path.stat().st_size,
        }
    return verified


def _validate_evidence_manifests(
    research: dict[str, Any],
    ensemble: dict[str, Any],
    walk_forward: dict[str, Any],
    execution: dict[str, Any],
    attribution: dict[str, Any],
) -> None:
    if research.get("campaign_test_free") is not True:
        raise PretestFreezeError("research freeze is not campaign-test-free")
    if research.get("phase3b_authorized") is not False:
        raise PretestFreezeError("research freeze unexpectedly authorizes Phase 3B")
    if ensemble.get("test_accessed") is not False:
        raise PretestFreezeError("ensemble evidence accessed the test split")
    if ensemble.get("development_data_label") != "2024 consumed development/selection data":
        raise PretestFreezeError("ensemble must label 2024 as consumed development data")
    if walk_forward.get("candidate_selected") is not False:
        raise PretestFreezeError("walk-forward campaign must not select the candidate")
    if walk_forward.get("folds") != ["WF1", "WF2", "WF3", "WF4"]:
        raise PretestFreezeError("walk-forward fold set mismatch")
    if execution.get("test_accessed") is not False:
        raise PretestFreezeError("execution stress accessed the test split")
    if execution.get("latest_evaluation_date") > "2023-12-31":
        raise PretestFreezeError("execution stress extends beyond 2023")
    if attribution.get("test_accessed") is not False:
        raise PretestFreezeError("regime attribution accessed the test split")
    if attribution.get("evaluation_period") != "2022-01-03 through 2023-12-29":
        raise PretestFreezeError("regime attribution period mismatch")


def _verify_members(
    root: Path, ensemble: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    if tuple(ensemble.get("member_seed_order", [])) != EXPECTED_SEEDS:
        raise PretestFreezeError("ensemble seed order mismatch")
    raw_members = ensemble.get("members")
    if not isinstance(raw_members, list) or len(raw_members) != 5:
        raise PretestFreezeError("ensemble must contain all five members")
    source_hashes = _mapping(ensemble, "source_hashes")
    feature_record = _mapping(source_hashes, "feature_spec")
    feature_path = _resolve(root, _required_text(feature_record, "path"))
    feature_hash = _required_text(feature_record, "sha256")
    _require_hash(feature_path, feature_hash, "feature specification")
    feature_spec = _read_json(feature_path)
    if tuple(feature_spec.get("asset_order", [])) != EXPECTED_ASSET_ORDER:
        raise PretestFreezeError("canonical asset order mismatch")
    if feature_spec.get("observation_dim") != 316:
        raise PretestFreezeError("observation dimension must remain 316")

    members = []
    environment: dict[str, Any] | None = None
    for order, raw in enumerate(raw_members):
        if raw.get("order") != order or raw.get("seed") != EXPECTED_SEEDS[order]:
            raise PretestFreezeError("member order or seed mismatch")
        if raw.get("selection_checkpoint") != "best_checkpoint":
            raise PretestFreezeError("members must use best validation checkpoints")
        model_path = _resolve(root, _required_text(raw, "model_path"))
        if model_path.is_symlink():
            raise PretestFreezeError("member model paths must not be symlinks")
        expected_hash = _required_text(raw, "model_sha256")
        _require_hash(model_path, expected_hash, "member model")
        run_dir = model_path.parent
        run_manifest = _read_json(run_dir / "manifest.json")
        if run_manifest.get("seed") != EXPECTED_SEEDS[order]:
            raise PretestFreezeError("member run manifest seed mismatch")
        for manifest_key, filename in (
            ("data_config_hash", "config.yaml"),
            ("data_quality_report_hash", "data_quality_report_v1.json"),
            ("env_config_hash", "env.yaml"),
            ("feature_spec_hash", "feature_spec_v1.json"),
            ("train_config_hash", "train_ppo.yaml"),
        ):
            _require_hash(run_dir / filename, str(run_manifest[manifest_key]), manifest_key)
        env_payload = _read_yaml(run_dir / "env.yaml")
        if environment is None:
            environment = env_payload
        elif env_payload != environment:
            raise PretestFreezeError("member environment snapshots differ")
        members.append(
            {
                "order": order,
                "seed": EXPECTED_SEEDS[order],
                "run_id": _required_text(raw, "run_id"),
                "model_path": _relative(model_path, root),
                "model_sha256": expected_hash,
                "model_size_bytes": model_path.stat().st_size,
                "action_temperature": float(raw["action_temperature"]),
                "selection_checkpoint": "best_checkpoint",
                "training_git_commit": run_manifest["git_commit"],
                "train_config_sha256": run_manifest["train_config_hash"],
                "environment_config_sha256": run_manifest["env_config_hash"],
                "run_manifest_path": _relative(run_dir / "manifest.json", root),
                "run_manifest_sha256": _sha256(run_dir / "manifest.json"),
            }
        )
    assert environment is not None
    if {member["action_temperature"] for member in members} != {0.5}:
        raise PretestFreezeError("all frozen member temperatures must equal 0.5")
    return members, {**feature_spec, "sha256": feature_hash}, environment


def _candidate_payload(
    *,
    root: Path,
    model_version: str,
    members: list[dict[str, Any]],
    feature_spec: dict[str, Any],
    environment: dict[str, Any],
) -> dict[str, Any]:
    del root
    return {
        "schema_version": 1,
        "model_version": model_version,
        "candidate_name": "five_seed_ppo_ensemble_alpha_0_25",
        "package_role": "final_test_candidate",
        "policy_type": "five_seed_mean_target_ensemble",
        "member_seed_order": list(EXPECTED_SEEDS),
        "member_model_paths": [member["model_path"] for member in members],
        "member_model_hashes": [member["model_sha256"] for member in members],
        "action_temperatures": [member["action_temperature"] for member in members],
        "action_transformation": "softmax(clip(action,-1,1)*temperature)",
        "ensemble_rule": "arithmetic_mean_of_member_target_weights",
        "partial_rebalance_alpha": 0.25,
        "initial_portfolio": "equal_weight",
        "initial_endowment_nav": 1.0,
        "initial_endowment_establishment_cost": 0.0,
        "asset_order": list(EXPECTED_ASSET_ORDER),
        "feature_version": feature_spec["feature_version"],
        "feature_spec_hash": feature_spec["sha256"],
        "observation_dimension": feature_spec["observation_dim"],
        "environment_config_hash": members[0]["environment_config_sha256"],
        "rebalance_frequency_trading_days": 5,
        "transaction_cost_bps": 10.0,
        "turnover_definition": "0.5 * sum(abs(executed_target - live_drifted_current_weights))",
        "long_only": True,
        "fully_invested": True,
        "final_test_status": "not_run_blocked_pending_new_independent_holdout",
        "test_accessed": False,
        "governance": BLOCKED_GOVERNANCE,
    }


def _candidate_from_payload(payload: dict[str, Any]) -> FrozenCandidate:
    seeds = tuple(int(seed) for seed in payload.get("member_seed_order", []))
    if seeds != EXPECTED_SEEDS:
        raise PretestFreezeError("frozen candidate seed order mismatch")
    alpha = float(payload.get("partial_rebalance_alpha", -1.0))
    if alpha != 0.25:
        raise PretestFreezeError("frozen candidate alpha must equal 0.25")
    if payload.get("initial_portfolio") != "equal_weight":
        raise PretestFreezeError("frozen candidate must start equal weight")
    if payload.get("turnover_definition") != (
        "0.5 * sum(abs(executed_target - live_drifted_current_weights))"
    ):
        raise PretestFreezeError("frozen turnover definition mismatch")
    return FrozenCandidate(
        model_version=_required_text(payload, "model_version"),
        member_seeds=seeds,
        member_model_paths=tuple(Path(path) for path in payload["member_model_paths"]),
        member_model_hashes=tuple(str(value) for value in payload["member_model_hashes"]),
        action_temperatures=tuple(float(value) for value in payload["action_temperatures"]),
        partial_rebalance_alpha=alpha,
        initial_portfolio="equal_weight",
        asset_order=tuple(str(asset) for asset in payload["asset_order"]),
        feature_version=_required_text(payload, "feature_version"),
        feature_spec_hash=_required_text(payload, "feature_spec_hash"),
        environment_config_hash=_required_text(payload, "environment_config_hash"),
        transaction_cost_bps=float(payload["transaction_cost_bps"]),
        rebalance_frequency_trading_days=int(payload["rebalance_frequency_trading_days"]),
    )


def _acceptance_criteria(config: dict[str, Any]) -> AcceptanceCriteria:
    return AcceptanceCriteria(
        primary_hurdle=_required_text(_mapping(config, "primary_hurdle"), "strategy"),
        hard_gates=_mapping(config, "hard_gates"),
        advisory_diagnostics=tuple(str(value) for value in config["advisory_diagnostics"]),
        secondary_comparisons=tuple(str(value) for value in config["secondary_comparisons"]),
        approval_status=_required_text(
            _mapping(config, "governance"), "acceptance_approval_status"
        ),
    )


def _audit_test_access(
    root: Path, research: dict[str, Any], members: list[dict[str, Any]]
) -> dict[str, Any]:
    audit_path = root / "artifacts/research_freeze" / str(research["campaign_id"]) / "test_access_audit.json"
    prior = _read_json(audit_path)
    known = {
        str(record["metadata_path"]): str(record["metadata_sha256"])
        for record in prior.get("known_legacy_access", [])
    }
    member_paths = {member["model_path"] for member in members}
    member_runs = {member["run_id"] for member in members}
    findings = []
    unexpected = []
    for path in sorted((root / "artifacts").rglob("*.json")):
        try:
            payload = _read_json(path)
        except (json.JSONDecodeError, TypeError):
            continue
        if not _declares_test_access(payload):
            continue
        relative = _relative(path, root)
        serialized = json.dumps(payload, sort_keys=True)
        current_candidate = (
            str(payload.get("model_path", "")) in member_paths
            or any(run_id in serialized for run_id in member_runs)
        )
        finding = {
            "metadata_path": relative,
            "metadata_sha256": _sha256(path),
            "declared_legacy": relative in known,
            "current_candidate": current_candidate,
            "split": payload.get("split"),
        }
        findings.append(finding)
        if current_candidate or relative not in known or known.get(relative) != finding["metadata_sha256"]:
            unexpected.append(relative)
    if unexpected:
        raise PretestFreezeError(
            "unexpected test artifacts detected: " + ", ".join(unexpected)
        )
    if set(known) != {finding["metadata_path"] for finding in findings}:
        raise PretestFreezeError("declared legacy test artifacts do not reconcile")
    return {
        "schema_version": 1,
        "audit_policy": "fail_on_unexpected_test_artifact",
        "candidate_campaign_test_accessed": False,
        "project_test_history_clear": False,
        "phase3b_authorized": False,
        "governance": BLOCKED_GOVERNANCE,
        "known_legacy_access": findings,
        "unexpected_test_access": [],
        "test_accessed": False,
    }


def _declares_test_access(payload: dict[str, Any]) -> bool:
    split = str(payload.get("split", "")).lower()
    status = str(payload.get("final_test_status", "")).lower()
    return split == "test" or payload.get("test_split_used") is True or status not in {
        "",
        "not_run",
        "not_run_blocked_pending_new_independent_holdout",
    }


def _data_and_feature_payload(
    ensemble: dict[str, Any], feature_spec: dict[str, Any]
) -> dict[str, Any]:
    sources = _mapping(ensemble, "source_hashes")
    model_matrix = _mapping(sources, "model_matrix")
    return {
        "schema_version": 1,
        "feature_version": feature_spec["feature_version"],
        "feature_spec_sha256": feature_spec["sha256"],
        "asset_order": list(EXPECTED_ASSET_ORDER),
        "observation_dimension": feature_spec["observation_dim"],
        "development_data_snapshot": {
            "path": model_matrix["path"],
            "recorded_sha256": model_matrix["sha256"],
            "verification_mode": "inherited_from_test_free_pr11_pr13_manifest_chain",
            "content_rehashed_in_pr19": False,
            "reason": "avoid accessing rows in the existing legacy test designation",
        },
        "approved_independent_holdout": None,
    }


def _environment_and_training_payload(
    root: Path, members: list[dict[str, Any]], environment: dict[str, Any]
) -> dict[str, Any]:
    source_paths = (
        "src/portfolio_rl/env/action.py",
        "src/portfolio_rl/env/costs.py",
        "src/portfolio_rl/env/drift.py",
        "src/portfolio_rl/policies/ensemble_policy.py",
        "src/portfolio_rl/policies/overlays.py",
        "src/portfolio_rl/policies/sb3_policy.py",
        "src/portfolio_rl/evaluation/backtest.py",
    )
    return {
        "schema_version": 1,
        "environment_config_logical_sha256": _sha256_text(
            yaml.safe_dump(environment, sort_keys=True)
        ),
        "member_training": [
            {
                key: member[key]
                for key in (
                    "seed",
                    "run_id",
                    "training_git_commit",
                    "train_config_sha256",
                    "run_manifest_path",
                    "run_manifest_sha256",
                )
            }
            for member in members
        ],
        "checkpoint_selection": {
            "development_campaign": {
                "split": "2024_consumed_development_selection",
                "metric": "sharpe_ratio",
                "evaluation_frequency_timesteps": 25000,
                "include_final_endpoint": True,
                "tie_break": "earliest_step",
            },
            "walk_forward_campaign": {
                "split": "inner_validation_only",
                "metric": "sharpe_ratio",
                "evaluation_frequency_timesteps": 25000,
                "include_final_endpoint": True,
                "tie_break": "earliest_step",
                "outer_rows_loaded_only_after_selection_freeze": True,
            },
        },
        "implementation_source_hashes": {
            path: _sha256(root / path) for path in source_paths
        },
    }


def _baseline_definitions() -> dict[str, Any]:
    common = {
        "initial_portfolio": "equal_weight_endowment_at_nav_1_no_establishment_cost",
        "rebalance_frequency_trading_days": 5,
        "transaction_cost_bps": 10.0,
        "turnover_definition": "half_l1_one_way",
        "drift_mechanics": "common_close_to_close_asset_returns",
    }
    return {
        "schema_version": 1,
        "primary_hurdle": "equal_weight_weekly",
        "common_contract": common,
        "definitions": {
            "equal_weight_weekly": {"rule": "rebalance_to_1_over_14_each_decision"},
            "buy_and_hold_equal_weight": {"rule": "no_trade_after_equal_weight_endowment"},
            "inverse_volatility": {
                "lookback_trading_days": 63,
                "history": "strictly_past_only_across_evaluation_boundary",
                "volatility_ddof": 0,
                "volatility_floor": 1e-8,
                "rule": "normalized_inverse_realized_volatility",
            },
            "momentum_63d_top3_equal_weight": {
                "lookback_trading_days": 63,
                "score": "sum_of_past_only_log_returns",
                "top_k": 3,
                "tie_break": "canonical_asset_order_stable_sort",
                "rule": "equal_weight_selected_assets",
            },
            "spy_only": {"ticker": "SPY"},
            "shy_only": {"ticker": "SHY"},
        },
    }


def _candidate_construction(candidate: dict[str, Any]) -> str:
    return f"""# Frozen Candidate Construction

At each five-trading-day decision, build one observation using the ensemble
portfolio's live drifted weights. Pass that identical observation to members in
seed order `{candidate['member_seed_order']}`. Each member predicts
deterministically and maps its action with
`softmax(clip(action, -1, 1) * temperature)`.

Average the five target-weight vectors arithmetically, then execute:

```python
executed_target = current_drifted_weights + 0.25 * (
    ensemble_target - current_drifted_weights
)
```

The system manages one live ensemble portfolio, not five separately drifting
member portfolios. Turnover is
`0.5 * sum(abs(executed_target - current_drifted_weights))`. The equal-weight
initial portfolio is an existing NAV-1 endowment with no establishment cost;
normal turnover and cost begin at the first policy decision.
"""


def _evidence_summary(root: Path, sources: dict[str, dict[str, Any]]) -> str:
    metrics_path = root / sources["walk_forward_metrics"]["path"]
    metrics = pd.read_csv(metrics_path)
    alpha = metrics.loc[metrics["strategy"] == "ensemble_alpha_0.25"].copy()
    rows = []
    for row in alpha.sort_values("fold_id").itertuples():
        rows.append(
            f"| {row.fold_id} | {row.active_return_vs_equal_weight:.2%} | "
            f"{row.active_sharpe_vs_equal_weight:+.3f} | "
            f"{row.average_weekly_turnover:.2%} |"
        )
    table = "\n".join(rows)
    return f"""# Evidence Summary

2024 is consumed development/selection data. It is not independent validation.

## Alpha 0.25 nested walk-forward results

| Fold | Active return vs weekly equal weight | Sharpe difference | Weekly turnover |
|---|---:|---:|---:|
{table}

Alpha 0.25 had positive active return in three of four folds. The one-close-delay
and delayed asset-tier joint stresses also retained three positive folds. WF4 and
the PR 15 controls show that the strategy did not demonstrate reliable tactical
timing in 2022–2023; the approved interpretation is a slow-moving PPO-derived
adaptive allocator with ensemble and turnover-control benefits.
"""


def _known_limitations() -> str:
    return """# Known Limitations

- Only four nested historical outer folds were evaluated.
- One outer fold had negative active return.
- Dynamic timing evidence was weak in 2022–2023.
- Some benefit appears attributable to strategic asset mix.
- Execution uses simplified close-to-close assumptions.
- Asset-tier costs are scenarios, not validated execution estimates.
- The universe is a static ETF-only universe.
- Taxes, market impact, and broker execution are not modeled.
- 2024 was consumed during development and selection.
- A legacy Phase 2 model accessed the existing 2025+ designation.
- Phase 3B requires a newly approved independent holdout.
"""


def _commands(model_version: str) -> str:
    return f"""# Commands

## Verify this package

```bash
python scripts/freeze_final_candidate.py --verify \\
  artifacts/pretest_freeze/{model_version}/frozen_candidate.json
```

## Future final-test command — BLOCKED AND NOT EXECUTED

```bash
python scripts/run_final_test.py \\
  --model-version {model_version} \\
  --holdout-manifest <approved-independent-holdout-manifest> \\
  --confirm-final-test
```

`scripts/run_final_test.py` is intentionally outside PR 19. Phase 3B remains
blocked until PM/ML approves a new independent holdout.
"""


def _package_readme(model_version: str) -> str:
    return f"""# Pre-Test Freeze `{model_version}`

This immutable package identifies the five-seed alpha-0.25 final-test candidate,
its evidence, acceptance rules, and source hashes. Member models are referenced
in place and rehashed during verification; they are not copied here.

This package does not authorize or contain a final-test result. The existing
2025+ designation has declared legacy Phase 2 access, so Phase 3B remains blocked
pending a newly approved independent holdout.
"""


def _require_clean_tracked_worktree(root: Path) -> None:
    for args in (["git", "diff", "--quiet"], ["git", "diff", "--cached", "--quiet"]):
        result = subprocess.run(args, cwd=root, check=False)
        if result.returncode != 0:
            raise PretestFreezeError("tracked Git worktree must be clean")


def _git_state(root: Path) -> dict[str, Any]:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()
    untracked = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    return {"head_commit": head, "tracked_worktree_clean": True, "untracked_paths": untracked}


def _git_root(path: Path) -> Path:
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    )
    return Path(result.stdout.strip()).resolve()


def _file_record(path: Path) -> dict[str, Any]:
    return {"path": path.name, "size_bytes": path.stat().st_size, "sha256": _sha256(path)}


def _require_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"required {label} is missing: {path}")
    before = path.stat()
    actual = _sha256(path)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise PretestFreezeError(f"{label} changed while hashing: {path}")
    if actual != expected:
        raise PretestFreezeError(
            f"{label} hash mismatch: expected={expected}, observed={actual}"
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _logical_json_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return _sha256_text(encoded)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected YAML mapping: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _mapping(payload: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise TypeError(f"required mapping is missing: {key}")
    return value


def _required_text(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"required text is missing: {key}")
    return value


def _resolve(root: Path, path: str | Path) -> Path:
    candidate = Path(path)
    return candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())

"""Immutable registration for one independent Phase 3B forward holdout."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from portfolio_rl.phase3b.governance import (
    EXPECTED_CANDIDATE_MANIFEST_SHA256,
    EXPECTED_MODEL_VERSION,
    IDENTIFIER_PATTERN,
    SIGNATURE_NAMESPACE,
    ApprovalRole,
    RegistrationError,
    canonical_json_bytes,
    git_state,
    load_access_policy,
    logical_json_sha256,
    read_json,
    read_yaml,
    relative_path,
    require_commit_exists,
    resolve_path,
    sha256_file,
    validate_certification,
    validate_container_identity,
    validate_schedule,
    verify_candidate,
    verify_ssh_signature,
)
from portfolio_rl.phase3b.incidents import incident_policy_payload

REQUIRED_CONFIG_NAMES = {
    "candidate_acceptance",
    "holdout_registration",
    "access_control",
    "execution",
    "operations",
}
GENERATED_FILES = (
    "holdout_registration.json",
    "approval_record.json",
    "candidate_verification.json",
    "config_hashes.json",
    "container_identity.json",
    "access_policy.json",
    "incident_policy.json",
)


@dataclass(frozen=True)
class RegisteredHoldout:
    """Verified identity of the registered independent holdout."""

    holdout_id: str
    start_decision_date: date
    end_decision_date: date
    final_holding_period_end_date: date
    performance_unseal_not_before: datetime
    candidate_model_version: str
    candidate_manifest_sha256: str
    container_image_digest: str
    git_commit: str


def prepare_registration_challenge(
    *,
    repository_root: Path,
    config_path: Path,
    challenge_output_path: Path,
    now: datetime | None = None,
) -> Path:
    """Validate all unsigned inputs and create one canonical signing challenge."""
    root = repository_root.resolve()
    current_time = _utc_now(now)
    config_resolved = resolve_path(root, config_path)
    output = resolve_path(root, challenge_output_path)
    if output.exists():
        raise FileExistsError(f"registration challenge already exists: {output}")
    config = _load_registration_config(config_resolved)
    if config["status"] != "approved":
        raise RegistrationError("holdout registration config is still draft")
    state = git_state(root, require_clean=True)
    challenge = _build_challenge(
        root=root,
        config_path=config_resolved,
        config=config,
        git_commit=state["head_commit"],
        created_at=current_time,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical_json_bytes(challenge))
    return output


def register_forward_holdout(
    *,
    repository_root: Path,
    challenge_path: Path,
    approvals_dir: Path,
    output_root: Path | None = None,
    now: datetime | None = None,
) -> Path:
    """Verify unanimous approvals and atomically register the signed holdout."""
    root = repository_root.resolve()
    current_time = _utc_now(now)
    challenge_resolved = resolve_path(root, challenge_path)
    challenge = read_json(challenge_resolved)
    _verify_challenge_payload(challenge)
    signed_output_root = resolve_path(root, str(challenge["output_root"]))
    if (
        output_root is not None
        and resolve_path(root, output_root) != signed_output_root
    ):
        raise RegistrationError("output root differs from the signed challenge")
    current_state = git_state(root, require_clean=True)
    if current_state["head_commit"] != challenge["git_commit"]:
        raise RegistrationError("Git HEAD changed after the challenge was prepared")
    _verify_challenge_sources(root, challenge)
    approvals = _verify_approvals(
        root=root,
        approvals_dir=resolve_path(root, approvals_dir),
        challenge=challenge,
    )
    start = _date(challenge["holdout"]["start_decision_date"])
    if current_time.date() >= start:
        raise RegistrationError("holdout registration must occur before its start date")

    signed_output_root.mkdir(parents=True, exist_ok=True)
    destination = signed_output_root / challenge["holdout"]["holdout_id"]
    lock_path = signed_output_root / ".registration.lock"
    lock_fd = _acquire_lock(lock_path)
    try:
        _require_registration_slot(signed_output_root, challenge["holdout"])
        if destination.exists():
            raise FileExistsError(f"holdout registration already exists: {destination}")
        temporary = Path(
            tempfile.mkdtemp(prefix=f".{destination.name}.", dir=signed_output_root)
        )
        try:
            registered_at = current_time.isoformat()
            _write_registration_package(
                root=root,
                temporary=temporary,
                challenge=challenge,
                approvals=approvals,
                registered_at=registered_at,
            )
            temporary.replace(destination)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
    finally:
        os.close(lock_fd)
        lock_path.unlink(missing_ok=True)
    verify_holdout_registration(destination, repository_root=root)
    return destination


def verify_holdout_registration(
    registration_dir: Path,
    *,
    repository_root: Path | None = None,
) -> RegisteredHoldout:
    """Verify a registration package and all referenced sources from scratch."""
    directory = registration_dir.resolve()
    root = (
        repository_root.resolve()
        if repository_root is not None
        else _repository_root(directory)
    )
    manifest = read_json(directory / "registration_manifest.json")
    recorded_payload_hash = manifest.get("manifest_payload_sha256")
    manifest_payload = dict(manifest)
    manifest_payload.pop("manifest_payload_sha256", None)
    if logical_json_sha256(manifest_payload) != recorded_payload_hash:
        raise RegistrationError("registration manifest payload hash mismatch")
    records = manifest.get("files")
    if not isinstance(records, list):
        raise RegistrationError("registration manifest files must be a list")
    if {record.get("path") for record in records} != set(GENERATED_FILES):
        raise RegistrationError("registration manifest coverage mismatch")
    for record in records:
        path = directory / str(record["path"])
        if sha256_file(path) != record.get("sha256"):
            raise RegistrationError(f"registered artifact hash mismatch: {path.name}")

    approval_record = read_json(directory / "approval_record.json")
    challenge = approval_record.get("registration_challenge")
    if not isinstance(challenge, dict):
        raise RegistrationError("approval record lacks its registration challenge")
    _verify_challenge_payload(challenge)
    if approval_record.get("challenge_payload_sha256") != challenge.get(
        "challenge_payload_sha256"
    ):
        raise RegistrationError("approval record challenge hash mismatch")
    _verify_embedded_approvals(challenge, approval_record)
    _verify_challenge_sources(root, challenge)
    require_commit_exists(root, str(challenge["git_commit"]))

    registration = read_json(directory / "holdout_registration.json")
    _require_registration_matches_challenge(registration, challenge)
    if read_json(directory / "candidate_verification.json") != _candidate_record(
        root, challenge
    ):
        raise RegistrationError("candidate verification artifact mismatch")
    if read_json(directory / "config_hashes.json") != {
        "schema_version": 1,
        "configs": challenge["config_records"],
    }:
        raise RegistrationError("config hashes artifact mismatch")
    if read_json(directory / "container_identity.json") != read_json(
        resolve_path(root, challenge["inputs"]["container_identity"]["path"])
    ):
        raise RegistrationError("container identity artifact mismatch")
    if read_json(directory / "access_policy.json") != challenge["access_policy"]:
        raise RegistrationError("access policy artifact mismatch")
    if read_json(directory / "incident_policy.json") != incident_policy_payload():
        raise RegistrationError("incident policy artifact mismatch")
    return RegisteredHoldout(
        holdout_id=registration["holdout_id"],
        start_decision_date=_date(registration["start_decision_date"]),
        end_decision_date=_date(registration["end_decision_date"]),
        final_holding_period_end_date=_date(
            registration["final_holding_period_end_date"]
        ),
        performance_unseal_not_before=_datetime(
            registration["performance_unseal_not_before"]
        ),
        candidate_model_version=registration["candidate_model_version"],
        candidate_manifest_sha256=registration["candidate_manifest_sha256"],
        container_image_digest=registration["container_image_digest"],
        git_commit=registration["git_commit"],
    )


def _load_registration_config(path: Path) -> dict[str, Any]:
    config = read_yaml(path)
    _exact_keys(
        config,
        {
            "schema_version",
            "status",
            "candidate",
            "holdout",
            "inputs",
            "frozen_configs",
            "access_control_path",
            "output_root",
        },
        "holdout registration config",
    )
    if config["schema_version"] != 1:
        raise RegistrationError("unsupported holdout registration schema")
    _exact_keys(
        config["candidate"],
        {"frozen_candidate_path", "candidate_manifest_sha256"},
        "candidate config",
    )
    _exact_keys(
        config["holdout"],
        {
            "holdout_id",
            "start_decision_date",
            "end_decision_date",
            "minimum_valid_rebalance_decisions",
            "expected_rebalance_frequency_trading_days",
            "performance_unseal_not_before",
            "input_schema_version",
            "data_source_contract_version",
            "existing_test_designation",
        },
        "holdout config",
    )
    _exact_keys(
        config["inputs"],
        {"certification_manifest", "trading_session_schedule", "container_identity"},
        "registration input config",
    )
    if set(config["frozen_configs"]) != REQUIRED_CONFIG_NAMES:
        raise RegistrationError(
            "frozen config bundle must include acceptance, registration, access, "
            "execution, and operations"
        )
    return config


def _build_challenge(
    *,
    root: Path,
    config_path: Path,
    config: dict[str, Any],
    git_commit: str,
    created_at: datetime,
) -> dict[str, Any]:
    holdout = config["holdout"]
    holdout_id = holdout["holdout_id"]
    if not isinstance(holdout_id, str) or not IDENTIFIER_PATTERN.fullmatch(holdout_id):
        raise RegistrationError("holdout_id is unresolved or invalid")
    if holdout_id == "2025+" or holdout["existing_test_designation"] != "2025+":
        raise RegistrationError(
            "the contaminated 2025+ designation cannot be registered"
        )
    start = _date(holdout["start_decision_date"])
    end = _date(holdout["end_decision_date"])
    if start <= created_at.date() or end <= start:
        raise RegistrationError("holdout dates must define a future interval")
    if holdout["minimum_valid_rebalance_decisions"] != 50:
        raise RegistrationError("minimum valid rebalance decisions must equal 50")
    if holdout["expected_rebalance_frequency_trading_days"] != 5:
        raise RegistrationError("rebalance frequency must equal five trading days")
    unseal = _datetime(holdout["performance_unseal_not_before"])

    candidate_config = config["candidate"]
    if not isinstance(candidate_config["frozen_candidate_path"], str):
        raise RegistrationError("frozen candidate path is unresolved")
    verified = verify_candidate(
        root,
        Path(candidate_config["frozen_candidate_path"]),
        str(candidate_config["candidate_manifest_sha256"]),
    )
    config_records = _config_records(root, config["frozen_configs"])
    registration_record = next(
        record for record in config_records if record["name"] == "holdout_registration"
    )
    if registration_record["path"] != relative_path(root, config_path):
        raise RegistrationError(
            "registration config did not include its own exact path"
        )
    candidate_acceptance = next(
        record for record in config_records if record["name"] == "candidate_acceptance"
    )
    frozen_acceptance = (
        verified.frozen_candidate_path.parent / "acceptance_criteria.yaml"
    )
    if candidate_acceptance["sha256"] != sha256_file(frozen_acceptance):
        raise RegistrationError(
            "candidate acceptance config differs from the PR 19 freeze"
        )

    if not isinstance(config["access_control_path"], str):
        raise RegistrationError("access-control path is unresolved")
    access_path = resolve_path(root, config["access_control_path"])
    access_policy = load_access_policy(root, access_path)
    access_record = next(
        record for record in config_records if record["name"] == "access_control"
    )
    if access_record["path"] != relative_path(root, access_path):
        raise RegistrationError("access-control path differs from frozen config bundle")

    if any(
        not isinstance(value, str) or not value for value in config["inputs"].values()
    ):
        raise RegistrationError("registration input paths are unresolved")
    input_paths = {
        name: resolve_path(root, value) for name, value in config["inputs"].items()
    }
    input_records = {
        name: _file_record(root, path) for name, path in input_paths.items()
    }
    container = read_json(input_paths["container_identity"])
    validate_container_identity(container, git_commit)
    if container["input_schema_version"] != holdout["input_schema_version"]:
        raise RegistrationError("input schema version differs from container identity")
    if (
        container["data_source_contract_version"]
        != holdout["data_source_contract_version"]
    ):
        raise RegistrationError(
            "data-source contract version differs from container identity"
        )
    schedule_path = input_paths["trading_session_schedule"]
    schedule_payload = read_json(schedule_path)
    schedule = validate_schedule(
        schedule_payload,
        start_decision_date=start,
        end_decision_date=end,
        minimum_decisions=50,
    )
    if unseal < _datetime(schedule["final_holding_period_complete_at_utc"]):
        raise RegistrationError("performance unseal precedes final holding completion")
    cert_payload = read_json(input_paths["certification_manifest"])
    config_hashes = {record["name"]: record["sha256"] for record in config_records}
    validate_certification(
        cert_payload,
        candidate_manifest_sha256=verified.freeze_manifest_sha256,
        container_digest=container["image_digest"],
        git_commit=git_commit,
        schedule_sha256=sha256_file(schedule_path),
        certification_dates=schedule["certification_decision_dates"],
        frozen_config_hashes=config_hashes,
    )
    certification_completed = _datetime(cert_payload["certification_completed_at"])
    if certification_completed > created_at:
        raise RegistrationError("challenge cannot precede certification completion")
    if certification_completed >= _at_start_of_day_utc(start):
        raise RegistrationError("certification must complete before the holdout starts")

    if not isinstance(config["output_root"], str) or not config["output_root"]:
        raise RegistrationError("registration output root is unresolved")
    challenge = {
        "schema_version": 1,
        "challenge_type": "phase3b_forward_holdout_registration",
        "challenge_created_at": created_at.isoformat(),
        "git_commit": git_commit,
        "candidate": {
            "model_version": EXPECTED_MODEL_VERSION,
            "frozen_candidate_path": relative_path(
                root, verified.frozen_candidate_path
            ),
            "freeze_manifest_path": relative_path(root, verified.freeze_manifest_path),
            "candidate_manifest_sha256": verified.freeze_manifest_sha256,
            "member_seed_order": list(verified.candidate.member_seeds),
            "partial_rebalance_alpha": 0.25,
            "initial_portfolio": "equal_weight",
            "rebalance_frequency_trading_days": 5,
            "primary_transaction_cost_bps": 10.0,
            "turnover_definition": "half_l1_one_way",
            "execution_timing": "one_close_delay",
            "asset_tier_cost_role": "stress_advisory",
        },
        "holdout": {
            "holdout_id": holdout_id,
            "start_decision_date": start.isoformat(),
            "end_decision_date": end.isoformat(),
            "final_holding_period_end_date": schedule["final_holding_period_end_date"],
            "minimum_valid_rebalance_decisions": 50,
            "expected_rebalance_frequency_trading_days": 5,
            "performance_unseal_not_before": unseal.isoformat(),
            "input_schema_version": holdout["input_schema_version"],
            "data_source_contract_version": holdout["data_source_contract_version"],
            "early_performance_stopping_prohibited": True,
            "performance_driven_extension_prohibited": True,
        },
        "governance": {
            "legacy_test_designation": "2025+",
            "legacy_test_independent": False,
            "legacy_block_reason": "Recorded legacy access before final candidate freeze",
            "replacement_holdout_required": True,
            "new_holdout_test_accessed": False,
            "performance_data_accessed": False,
            "approval_rule": "unanimous_three_of_three",
        },
        "certification": {
            "certification_id": cert_payload["certification_id"],
            "certification_manifest_sha256": sha256_file(
                input_paths["certification_manifest"]
            ),
            "certification_completed_at": cert_payload["certification_completed_at"],
            "cycle_count": 4,
        },
        "container_image_digest": container["image_digest"],
        "schedule": schedule,
        "inputs": input_records,
        "config_records": config_records,
        "access_policy": access_policy,
        "access_policy_sha256": logical_json_sha256(access_policy),
        "incident_policy_sha256": logical_json_sha256(incident_policy_payload()),
        "signature_namespace": SIGNATURE_NAMESPACE,
        "output_root": relative_path(root, resolve_path(root, config["output_root"])),
    }
    challenge["challenge_payload_sha256"] = logical_json_sha256(challenge)
    return challenge


def _verify_challenge_payload(challenge: dict[str, Any]) -> None:
    recorded = challenge.get("challenge_payload_sha256")
    payload = dict(challenge)
    payload.pop("challenge_payload_sha256", None)
    if logical_json_sha256(payload) != recorded:
        raise RegistrationError("registration challenge payload hash mismatch")
    if challenge.get("schema_version") != 1:
        raise RegistrationError("unsupported registration challenge schema")
    if challenge.get("signature_namespace") != SIGNATURE_NAMESPACE:
        raise RegistrationError("registration challenge namespace mismatch")
    candidate = challenge.get("candidate", {})
    if candidate.get("model_version") != EXPECTED_MODEL_VERSION:
        raise RegistrationError("registration challenge candidate mismatch")
    if candidate.get("candidate_manifest_sha256") != EXPECTED_CANDIDATE_MANIFEST_SHA256:
        raise RegistrationError("registration challenge candidate hash mismatch")
    immutable_candidate_fields = {
        "member_seed_order": [7, 42, 101, 202, 999],
        "partial_rebalance_alpha": 0.25,
        "initial_portfolio": "equal_weight",
        "rebalance_frequency_trading_days": 5,
        "primary_transaction_cost_bps": 10.0,
        "turnover_definition": "half_l1_one_way",
        "execution_timing": "one_close_delay",
        "asset_tier_cost_role": "stress_advisory",
    }
    for key, value in immutable_candidate_fields.items():
        if candidate.get(key) != value:
            raise RegistrationError(f"registration challenge candidate mismatch: {key}")
    if challenge.get("governance", {}).get("new_holdout_test_accessed") is not False:
        raise RegistrationError("registration challenge indicates holdout access")
    if challenge.get("governance", {}).get("performance_data_accessed") is not False:
        raise RegistrationError("registration challenge indicates performance access")


def _verify_challenge_sources(root: Path, challenge: dict[str, Any]) -> None:
    candidate = challenge["candidate"]
    verify_candidate(
        root,
        Path(candidate["frozen_candidate_path"]),
        candidate["candidate_manifest_sha256"],
    )
    config_records = challenge["config_records"]
    if {record.get("name") for record in config_records} != REQUIRED_CONFIG_NAMES:
        raise RegistrationError("signed frozen config bundle is incomplete")
    for record in config_records:
        path = resolve_path(root, record["path"])
        if sha256_file(path) != record["sha256"]:
            raise RegistrationError(f"frozen config hash mismatch: {record['name']}")
        if logical_json_sha256(read_yaml(path)) != record["logical_sha256"]:
            raise RegistrationError(
                f"frozen config logical hash mismatch: {record['name']}"
            )
    access_record = next(
        record for record in config_records if record["name"] == "access_control"
    )
    if (
        load_access_policy(root, Path(access_record["path"]))
        != challenge["access_policy"]
    ):
        raise RegistrationError("signed access policy differs from its frozen config")
    acceptance_record = next(
        record for record in config_records if record["name"] == "candidate_acceptance"
    )
    frozen_acceptance = (
        resolve_path(root, candidate["frozen_candidate_path"]).parent
        / "acceptance_criteria.yaml"
    )
    if acceptance_record["sha256"] != sha256_file(frozen_acceptance):
        raise RegistrationError("signed acceptance config differs from PR 19 freeze")
    for name, record in challenge["inputs"].items():
        path = resolve_path(root, record["path"])
        if sha256_file(path) != record["sha256"]:
            raise RegistrationError(f"registration input hash mismatch: {name}")
    if (
        logical_json_sha256(challenge["access_policy"])
        != challenge["access_policy_sha256"]
    ):
        raise RegistrationError("signed access policy hash mismatch")
    if (
        logical_json_sha256(incident_policy_payload())
        != challenge["incident_policy_sha256"]
    ):
        raise RegistrationError("signed incident policy hash mismatch")
    container = read_json(
        resolve_path(root, challenge["inputs"]["container_identity"]["path"])
    )
    validate_container_identity(container, challenge["git_commit"])
    if (
        container["input_schema_version"]
        != challenge["holdout"]["input_schema_version"]
    ):
        raise RegistrationError("signed input schema version mismatch")
    if (
        container["data_source_contract_version"]
        != challenge["holdout"]["data_source_contract_version"]
    ):
        raise RegistrationError("signed data-source contract version mismatch")
    schedule_path = resolve_path(
        root, challenge["inputs"]["trading_session_schedule"]["path"]
    )
    schedule = validate_schedule(
        read_json(schedule_path),
        start_decision_date=_date(challenge["holdout"]["start_decision_date"]),
        end_decision_date=_date(challenge["holdout"]["end_decision_date"]),
        minimum_decisions=50,
    )
    if schedule != challenge["schedule"]:
        raise RegistrationError("signed schedule summary mismatch")
    if _datetime(challenge["holdout"]["performance_unseal_not_before"]) < _datetime(
        schedule["final_holding_period_complete_at_utc"]
    ):
        raise RegistrationError("signed unseal time precedes final holding completion")
    cert = read_json(
        resolve_path(root, challenge["inputs"]["certification_manifest"]["path"])
    )
    validate_certification(
        cert,
        candidate_manifest_sha256=candidate["candidate_manifest_sha256"],
        container_digest=challenge["container_image_digest"],
        git_commit=challenge["git_commit"],
        schedule_sha256=sha256_file(schedule_path),
        certification_dates=schedule["certification_decision_dates"],
        frozen_config_hashes={
            record["name"]: record["sha256"] for record in challenge["config_records"]
        },
    )
    if cert["certification_id"] != challenge["certification"]["certification_id"]:
        raise RegistrationError("signed certification ID mismatch")
    if (
        cert["certification_completed_at"]
        != challenge["certification"]["certification_completed_at"]
    ):
        raise RegistrationError("signed certification completion mismatch")


def _verify_approvals(
    *, root: Path, approvals_dir: Path, challenge: dict[str, Any]
) -> list[dict[str, str]]:
    if not approvals_dir.is_dir():
        raise RegistrationError("approval signature directory does not exist")
    expected_names = {f"{role.value}.sig" for role in ApprovalRole}
    observed_names = {path.name for path in approvals_dir.iterdir() if path.is_file()}
    if observed_names != expected_names:
        raise RegistrationError("approval signatures are incomplete or unexpected")
    challenge_bytes = canonical_json_bytes(challenge)
    approvals: list[dict[str, str]] = []
    for approver in challenge["access_policy"]["approvers"]:
        signature_path = approvals_dir / f"{approver['role']}.sig"
        signature = signature_path.read_text(encoding="utf-8")
        verify_ssh_signature(
            challenge_bytes=challenge_bytes,
            signature=signature,
            approver=approver,
        )
        approvals.append(
            {
                **approver,
                "decision": "approve",
                "signature": signature,
                "signature_sha256": sha256_file(signature_path),
            }
        )
    return sorted(approvals, key=lambda item: item["role"])


def _verify_embedded_approvals(
    challenge: dict[str, Any], approval_record: dict[str, Any]
) -> None:
    approvals = approval_record.get("approvals")
    if not isinstance(approvals, list) or len(approvals) != 3:
        raise RegistrationError("approval record must contain three approvals")
    expected = {role.value for role in ApprovalRole}
    if {item.get("role") for item in approvals} != expected:
        raise RegistrationError("approval record roles are incomplete")
    approver_map = {
        item["role"]: item for item in challenge["access_policy"]["approvers"]
    }
    for approval in approvals:
        if approval.get("decision") != "approve":
            raise RegistrationError("holdout registration approval is not unanimous")
        approver = approver_map[approval["role"]]
        for key in (
            "name",
            "principal",
            "public_key",
            "public_key_sha256",
            "public_key_fingerprint",
        ):
            if approval.get(key) != approver[key]:
                raise RegistrationError("approval identity differs from signed policy")
        signature = approval.get("signature")
        if not isinstance(signature, str):
            raise RegistrationError("approval signature is missing")
        if (
            approval.get("signature_sha256")
            != hashlib.sha256(signature.encode("utf-8")).hexdigest()
        ):
            raise RegistrationError("approval signature hash mismatch")
        if approval.get("verified_at") != approval_record.get("verified_at"):
            raise RegistrationError("approval verification timestamp mismatch")
        verify_ssh_signature(
            challenge_bytes=canonical_json_bytes(challenge),
            signature=signature,
            approver=approver,
        )


def _write_registration_package(
    *,
    root: Path,
    temporary: Path,
    challenge: dict[str, Any],
    approvals: list[dict[str, str]],
    registered_at: str,
) -> None:
    holdout = challenge["holdout"]
    registration = {
        "schema_version": 1,
        "holdout_id": holdout["holdout_id"],
        "status": "registered",
        "start_decision_date": holdout["start_decision_date"],
        "end_decision_date": holdout["end_decision_date"],
        "final_holding_period_end_date": holdout["final_holding_period_end_date"],
        "registration_timestamp": registered_at,
        "certification_id": challenge["certification"]["certification_id"],
        "certification_completed_at": challenge["certification"][
            "certification_completed_at"
        ],
        "minimum_valid_rebalance_decisions": 50,
        "expected_rebalance_frequency_trading_days": 5,
        "performance_unseal_not_before": holdout["performance_unseal_not_before"],
        "candidate_model_version": EXPECTED_MODEL_VERSION,
        "candidate_manifest_sha256": EXPECTED_CANDIDATE_MANIFEST_SHA256,
        "container_image_digest": challenge["container_image_digest"],
        "git_commit": challenge["git_commit"],
        "input_schema_version": holdout["input_schema_version"],
        "data_source_contract_version": holdout["data_source_contract_version"],
        "primary_execution_timing": "one_close_delay",
        "primary_transaction_cost_bps": 10.0,
        "asset_tier_cost_role": "stress_advisory",
        "legacy_test_designation": "2025+",
        "legacy_test_independent": False,
        "new_holdout_test_accessed": False,
        "performance_data_accessed": False,
        "early_performance_stopping_prohibited": True,
        "performance_driven_extension_prohibited": True,
    }
    approval_record = {
        "schema_version": 1,
        "signature_namespace": SIGNATURE_NAMESPACE,
        "challenge_payload_sha256": challenge["challenge_payload_sha256"],
        "registration_challenge": challenge,
        "approvals": [
            {**approval, "verified_at": registered_at} for approval in approvals
        ],
        "verified_at": registered_at,
        "approval_rule": "unanimous_three_of_three",
    }
    payloads = {
        "holdout_registration.json": registration,
        "approval_record.json": approval_record,
        "candidate_verification.json": _candidate_record(root, challenge),
        "config_hashes.json": {
            "schema_version": 1,
            "configs": challenge["config_records"],
        },
        "container_identity.json": read_json(
            resolve_path(root, challenge["inputs"]["container_identity"]["path"])
        ),
        "access_policy.json": challenge["access_policy"],
        "incident_policy.json": incident_policy_payload(),
    }
    for filename, payload in payloads.items():
        _write_json(temporary / filename, payload)
    manifest = {
        "schema_version": 1,
        "holdout_id": holdout["holdout_id"],
        "registered_at": registered_at,
        "git_commit": challenge["git_commit"],
        "candidate_manifest_sha256": EXPECTED_CANDIDATE_MANIFEST_SHA256,
        "challenge_payload_sha256": challenge["challenge_payload_sha256"],
        "test_accessed": False,
        "performance_data_accessed": False,
        "files": [
            _file_record(temporary, temporary / name) for name in GENERATED_FILES
        ],
        "self_hash_contract": (
            "sha256_of_canonical_json_without_manifest_payload_sha256"
        ),
    }
    manifest["manifest_payload_sha256"] = logical_json_sha256(manifest)
    _write_json(temporary / "registration_manifest.json", manifest)


def _candidate_record(root: Path, challenge: dict[str, Any]) -> dict[str, Any]:
    candidate = challenge["candidate"]
    verified = verify_candidate(
        root,
        Path(candidate["frozen_candidate_path"]),
        candidate["candidate_manifest_sha256"],
    )
    return {
        "schema_version": 1,
        "verified": True,
        "model_version": verified.candidate.model_version,
        "frozen_candidate_path": candidate["frozen_candidate_path"],
        "freeze_manifest_path": candidate["freeze_manifest_path"],
        "candidate_manifest_sha256": verified.freeze_manifest_sha256,
        "member_seed_order": list(verified.candidate.member_seeds),
        "member_model_hashes": list(verified.candidate.member_model_hashes),
        "partial_rebalance_alpha": verified.candidate.partial_rebalance_alpha,
        "initial_portfolio": verified.candidate.initial_portfolio,
        "rebalance_frequency_trading_days": (
            verified.candidate.rebalance_frequency_trading_days
        ),
        "transaction_cost_bps": verified.candidate.transaction_cost_bps,
        "turnover_definition": "half_l1_one_way",
        "test_accessed": False,
    }


def _require_registration_matches_challenge(
    registration: dict[str, Any], challenge: dict[str, Any]
) -> None:
    holdout = challenge["holdout"]
    expected = {
        "holdout_id": holdout["holdout_id"],
        "start_decision_date": holdout["start_decision_date"],
        "end_decision_date": holdout["end_decision_date"],
        "final_holding_period_end_date": holdout["final_holding_period_end_date"],
        "performance_unseal_not_before": holdout["performance_unseal_not_before"],
        "candidate_model_version": EXPECTED_MODEL_VERSION,
        "candidate_manifest_sha256": EXPECTED_CANDIDATE_MANIFEST_SHA256,
        "container_image_digest": challenge["container_image_digest"],
        "git_commit": challenge["git_commit"],
    }
    for key, value in expected.items():
        if registration.get(key) != value:
            raise RegistrationError(f"registered holdout differs from challenge: {key}")
    if registration.get("status") != "registered":
        raise RegistrationError("holdout status is not registered")
    if registration.get("new_holdout_test_accessed") is not False:
        raise RegistrationError("registration indicates holdout access")


def _require_registration_slot(output_root: Path, holdout: dict[str, Any]) -> None:
    requested_start = _date(holdout["start_decision_date"])
    requested_end = _date(holdout["final_holding_period_end_date"])
    for child in output_root.iterdir():
        if not child.is_dir() or child.name.startswith("."):
            continue
        registration_path = child / "holdout_registration.json"
        if not registration_path.exists():
            raise RegistrationError(
                "unexpected incomplete registration artifact exists"
            )
        existing = read_json(registration_path)
        if existing.get("holdout_id") == holdout["holdout_id"]:
            raise RegistrationError("duplicate holdout registration")
        existing_start = _date(existing["start_decision_date"])
        existing_end = _date(existing["final_holding_period_end_date"])
        if requested_start <= existing_end and existing_start <= requested_end:
            raise RegistrationError(
                "holdout registration overlaps an existing interval"
            )
        raise RegistrationError(
            "only one independent forward holdout may be registered"
        )


def _config_records(root: Path, mapping: Any) -> list[dict[str, Any]]:
    if not isinstance(mapping, dict):
        raise RegistrationError("frozen_configs must be a mapping")
    records = []
    for name, value in mapping.items():
        if not isinstance(value, str) or not value:
            raise RegistrationError(f"frozen config path is unresolved: {name}")
        path = resolve_path(root, value)
        payload = read_yaml(path)
        records.append(
            {
                "name": name,
                "path": relative_path(root, path),
                "sha256": sha256_file(path),
                "logical_sha256": logical_json_sha256(payload),
                "size_bytes": path.stat().st_size,
            }
        )
    return sorted(records, key=lambda record: record["name"])


def _file_record(root: Path, path: Path) -> dict[str, Any]:
    return {
        "path": relative_path(root, path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _acquire_lock(path: Path) -> int:
    try:
        return os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise RegistrationError(
            "registration lock already exists; fail closed"
        ) from exc


def _repository_root(path: Path) -> Path:
    import subprocess

    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=path,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RegistrationError("registration package is not inside a Git repository")
    return Path(result.stdout.strip()).resolve()


def _exact_keys(payload: Any, expected: set[str], label: str) -> None:
    if not isinstance(payload, dict):
        raise RegistrationError(f"{label} must be a mapping")
    if set(payload) != expected:
        raise RegistrationError(
            f"{label} keys mismatch; missing={sorted(expected - set(payload))}, "
            f"unexpected={sorted(set(payload) - expected)}"
        )


def _date(value: Any) -> date:
    if not isinstance(value, str):
        raise RegistrationError("expected ISO date")
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise RegistrationError("expected ISO date") from exc


def _datetime(value: Any) -> datetime:
    if not isinstance(value, str):
        raise RegistrationError("expected ISO UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise RegistrationError("expected ISO UTC timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != UTC.utcoffset(parsed):
        raise RegistrationError("timestamp must use UTC")
    return parsed


def _at_start_of_day_utc(value: date) -> datetime:
    return datetime(value.year, value.month, value.day, tzinfo=UTC)


def _utc_now(value: datetime | None) -> datetime:
    current = value or datetime.now(UTC)
    if current.tzinfo is None or current.utcoffset() != UTC.utcoffset(current):
        raise RegistrationError("registration time must use UTC")
    return current

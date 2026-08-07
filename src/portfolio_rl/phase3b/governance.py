"""Fail-closed validation for Phase 3B holdout registration inputs."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import UTC, date, datetime
from enum import StrEnum
from itertools import pairwise
from pathlib import Path
from typing import Any

import yaml

from portfolio_rl.training.pretest_freeze import (
    EXPECTED_SEEDS,
    FrozenCandidate,
    verify_frozen_candidate,
)

EXPECTED_MODEL_VERSION = "ppo_v1_ensemble5_alpha025_pretest_v1"
EXPECTED_CANDIDATE_MANIFEST_SHA256 = (
    "1480c8de2323fa8555e5fa4e8f9f5adfd39b465a742ac0bc43bff066dcc39edd"
)
SIGNATURE_NAMESPACE = "portfolio-rl-phase3b-registration-v1"
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
GIT_COMMIT_PATTERN = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
OCI_DIGEST_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
SSH_PRINCIPAL_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9@._+-]*$")


class RegistrationError(ValueError):
    """Base error for Phase 3B registration and verification failures."""


class GovernanceError(RegistrationError):
    """Raised when a Phase 3B governance precondition is not satisfied."""


class ApprovalRole(StrEnum):
    """The three independent roles required to register the holdout."""

    PORTFOLIO_MANAGER = "portfolio_manager"
    INDEPENDENT_REVIEWER = "independent_reviewer"
    DATA_OPERATIONS_CUSTODIAN = "data_operations_custodian"


@dataclass(frozen=True)
class VerifiedCandidate:
    """Verified Phase 3A identity recorded in a registration challenge."""

    candidate: FrozenCandidate
    frozen_candidate_path: Path
    freeze_manifest_path: Path
    freeze_manifest_sha256: str


def canonical_json_bytes(payload: Any) -> bytes:
    """Serialize a JSON-compatible payload deterministically."""
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def logical_json_sha256(payload: Any) -> str:
    """Hash canonical JSON content."""
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash a file without loading large model artifacts into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ssh_public_key_fingerprint(path: Path) -> str:
    """Return the OpenSSH SHA-256 fingerprint for one public key file."""
    result = subprocess.run(
        ["ssh-keygen", "-lf", str(path), "-E", "sha256"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise GovernanceError("cannot calculate approver SSH key fingerprint")
    fields = result.stdout.split()
    if len(fields) < 2 or not fields[1].startswith("SHA256:"):
        raise GovernanceError("unexpected approver SSH key fingerprint format")
    return fields[1]


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GovernanceError(f"cannot read JSON object: {path}") from exc
    if not isinstance(payload, dict):
        raise GovernanceError(f"expected JSON object: {path}")
    return payload


def read_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML mapping."""
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise GovernanceError(f"cannot read YAML mapping: {path}") from exc
    if not isinstance(payload, dict):
        raise GovernanceError(f"expected YAML mapping: {path}")
    return payload


def resolve_path(root: Path, value: str | Path) -> Path:
    """Resolve a repository-relative path without permitting path escape."""
    path = Path(value)
    resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise GovernanceError(f"path escapes repository root: {value}") from exc
    return resolved


def relative_path(root: Path, path: Path) -> str:
    """Return a stable repository-relative path."""
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError as exc:
        raise GovernanceError(f"path is outside repository root: {path}") from exc


def verify_candidate(
    root: Path, candidate_path: Path, expected_sha256: str
) -> VerifiedCandidate:
    """Verify the exact PR 19 candidate and every transitive model/config hash."""
    if expected_sha256 != EXPECTED_CANDIDATE_MANIFEST_SHA256:
        raise GovernanceError("candidate manifest hash is not the approved PR 19 hash")
    resolved = resolve_path(root, candidate_path)
    manifest_path = resolved.parent / "freeze_manifest.json"
    if sha256_file(manifest_path) != expected_sha256:
        raise GovernanceError("candidate freeze manifest hash mismatch")
    try:
        candidate = verify_frozen_candidate(resolved)
    except (OSError, ValueError) as exc:
        raise GovernanceError("Phase 3A candidate verification failed") from exc
    if candidate.model_version != EXPECTED_MODEL_VERSION:
        raise GovernanceError("candidate model version mismatch")
    if candidate.member_seeds != EXPECTED_SEEDS:
        raise GovernanceError("candidate seed order mismatch")
    if candidate.partial_rebalance_alpha != 0.25:
        raise GovernanceError("candidate alpha must equal 0.25")
    if candidate.initial_portfolio != "equal_weight":
        raise GovernanceError("candidate initial portfolio must be equal weight")
    if candidate.rebalance_frequency_trading_days != 5:
        raise GovernanceError("candidate rebalance frequency must equal five days")
    if candidate.transaction_cost_bps != 10.0:
        raise GovernanceError("candidate primary transaction cost must equal 10 bps")
    frozen_payload = read_json(resolved)
    expected_turnover = "0.5 * sum(abs(executed_target - live_drifted_current_weights))"
    if frozen_payload.get("turnover_definition") != expected_turnover:
        raise GovernanceError("candidate turnover definition must be half-L1 one-way")
    return VerifiedCandidate(
        candidate=candidate,
        frozen_candidate_path=resolved,
        freeze_manifest_path=manifest_path,
        freeze_manifest_sha256=expected_sha256,
    )


def git_state(root: Path, *, require_clean: bool) -> dict[str, Any]:
    """Return the repository commit and optionally require a completely clean tree."""
    head = _git(root, "rev-parse", "HEAD").strip()
    if not GIT_COMMIT_PATTERN.fullmatch(head):
        raise GovernanceError("Git HEAD is not a full SHA-1/256 commit identifier")
    status = _git(root, "status", "--porcelain", "--untracked-files=all")
    if require_clean and status.strip():
        raise GovernanceError("Git working tree must be clean before registration")
    return {"head_commit": head, "working_tree_clean": not status.strip()}


def require_commit_exists(root: Path, commit: str) -> None:
    """Require a recorded registration commit to remain addressable."""
    result = subprocess.run(
        ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise GovernanceError("recorded registration Git commit is unavailable")


def validate_container_identity(payload: dict[str, Any], expected_commit: str) -> None:
    """Validate the immutable runner-container attestation."""
    _require_keys(
        payload,
        {
            "schema_version",
            "image_reference",
            "image_digest",
            "git_commit",
            "input_schema_version",
            "data_source_contract_version",
            "built_at",
        },
        "container identity",
    )
    if payload["schema_version"] != 1:
        raise GovernanceError("unsupported container identity schema")
    if (
        not isinstance(payload["image_reference"], str)
        or not payload["image_reference"]
    ):
        raise GovernanceError("container image reference is unresolved")
    if not OCI_DIGEST_PATTERN.fullmatch(str(payload["image_digest"])):
        raise GovernanceError("container image digest must be an immutable OCI digest")
    if payload["git_commit"] != expected_commit:
        raise GovernanceError("container image Git commit does not match current HEAD")
    _parse_datetime(payload["built_at"], "container built_at")
    for key in ("input_schema_version", "data_source_contract_version"):
        if not isinstance(payload[key], str) or not payload[key].strip():
            raise GovernanceError(f"container {key} is unresolved")


def validate_schedule(
    payload: dict[str, Any],
    *,
    start_decision_date: date,
    end_decision_date: date,
    minimum_decisions: int,
) -> dict[str, Any]:
    """Validate the custodian-supplied certification and holdout session schedule."""
    _require_keys(
        payload,
        {
            "schema_version",
            "schedule_id",
            "timezone",
            "trading_sessions",
            "certification_decision_dates",
            "holdout_decision_dates",
            "final_holding_period_end_date",
            "final_holding_period_complete_at_utc",
        },
        "trading-session schedule",
    )
    if payload["schema_version"] != 1:
        raise GovernanceError("unsupported trading-session schedule schema")
    if not isinstance(payload["schedule_id"], str) or not IDENTIFIER_PATTERN.fullmatch(
        payload["schedule_id"]
    ):
        raise GovernanceError("trading-session schedule ID is unresolved or invalid")
    if payload["timezone"] != "America/New_York":
        raise GovernanceError(
            "trading-session schedule timezone must be America/New_York"
        )
    sessions = _date_list(payload["trading_sessions"], "trading_sessions")
    if sessions != sorted(set(sessions)):
        raise GovernanceError("trading sessions must be unique and ordered")
    positions = {session: index for index, session in enumerate(sessions)}
    certification = _date_list(
        payload["certification_decision_dates"], "certification_decision_dates"
    )
    decisions = _date_list(payload["holdout_decision_dates"], "holdout_decision_dates")
    if len(certification) != 4:
        raise GovernanceError(
            "certification schedule must contain exactly four decisions"
        )
    if len(decisions) < minimum_decisions:
        raise GovernanceError("holdout schedule has fewer than 50 decisions")
    _require_five_session_spacing(certification, positions, "certification")
    _require_five_session_spacing(decisions, positions, "holdout")
    if decisions[0] != start_decision_date or decisions[-1] != end_decision_date:
        raise GovernanceError("registered dates do not match the holdout schedule")
    if certification[-1] >= decisions[0]:
        raise GovernanceError("certification must end before the holdout starts")
    anniversary = _calendar_anniversary(start_decision_date)
    if end_decision_date > anniversary:
        raise GovernanceError("holdout end decision is after the 12-month anniversary")
    next_position = positions[end_decision_date] + 5
    if next_position >= len(sessions):
        raise GovernanceError("schedule lacks the next decision after the anniversary")
    if sessions[next_position] <= anniversary:
        raise GovernanceError(
            "holdout end is not the last weekly decision by anniversary"
        )
    final_holding_end = _parse_date(
        payload["final_holding_period_end_date"], "final_holding_period_end_date"
    )
    if final_holding_end != sessions[next_position]:
        raise GovernanceError(
            "final holding period is not exactly five trading sessions"
        )
    completion = _parse_datetime(
        payload["final_holding_period_complete_at_utc"],
        "final_holding_period_complete_at_utc",
    )
    if completion.date() < final_holding_end:
        raise GovernanceError("final holding completion precedes its final session")
    return {
        "schedule_id": payload["schedule_id"],
        "certification_decision_dates": [item.isoformat() for item in certification],
        "holdout_decision_dates": [item.isoformat() for item in decisions],
        "final_holding_period_end_date": final_holding_end.isoformat(),
        "final_holding_period_complete_at_utc": completion.isoformat(),
    }


def validate_certification(
    payload: dict[str, Any],
    *,
    candidate_manifest_sha256: str,
    container_digest: str,
    git_commit: str,
    schedule_sha256: str,
    certification_dates: list[str],
    frozen_config_hashes: dict[str, str],
) -> None:
    """Require a hash-bound, four-cycle, performance-free certification receipt."""
    required = {
        "schema_version",
        "certification_id",
        "status",
        "cycle_count",
        "completed_decision_dates",
        "candidate_manifest_sha256",
        "container_image_digest",
        "git_commit",
        "schedule_sha256",
        "frozen_config_hashes",
        "performance_metrics_computed",
        "official",
        "all_required_checks_passed",
        "missed_scheduled_decisions",
        "cycle_manifest_hashes",
        "approved_identities",
        "canonical_holdout_registered",
        "certification_artifacts_excluded_from_holdout",
        "certification_completed_at",
        "manifest_payload_sha256",
    }
    _require_keys(payload, required, "certification manifest")
    expected_payload = dict(payload)
    recorded_hash = expected_payload.pop("manifest_payload_sha256")
    if logical_json_sha256(expected_payload) != recorded_hash:
        raise GovernanceError("certification manifest payload hash mismatch")
    if payload["schema_version"] != 1 or payload["status"] != "passed":
        raise GovernanceError("operational certification has not passed")
    if payload["cycle_count"] != 4:
        raise GovernanceError("operational certification requires four cycles")
    if payload["official"] is not True:
        raise GovernanceError("development cycles cannot satisfy certification")
    if payload["all_required_checks_passed"] is not True:
        raise GovernanceError("certification has incomplete operational checks")
    if payload["missed_scheduled_decisions"] != 0:
        raise GovernanceError("certification cycles must be consecutive and complete")
    if payload["canonical_holdout_registered"] is not False:
        raise GovernanceError("certification cannot register a canonical holdout")
    if payload["certification_artifacts_excluded_from_holdout"] is not True:
        raise GovernanceError("certification artifacts must be excluded from holdout")
    cycle_hashes = payload["cycle_manifest_hashes"]
    if (
        not isinstance(cycle_hashes, list)
        or len(cycle_hashes) != 4
        or any(not SHA256_PATTERN.fullmatch(str(value)) for value in cycle_hashes)
    ):
        raise GovernanceError("certification must bind four cycle manifests")
    if payload["completed_decision_dates"] != certification_dates:
        raise GovernanceError("certification dates do not match the approved schedule")
    expected = {
        "candidate_manifest_sha256": candidate_manifest_sha256,
        "container_image_digest": container_digest,
        "git_commit": git_commit,
        "schedule_sha256": schedule_sha256,
    }
    for key, value in expected.items():
        if payload[key] != value:
            raise GovernanceError(f"certification {key} mismatch")
    certified_hashes = payload["frozen_config_hashes"]
    if not isinstance(certified_hashes, dict):
        raise GovernanceError("certification config hashes must be a mapping")
    required_certified_configs = {"candidate_acceptance", "execution", "operations"}
    if set(certified_hashes) != required_certified_configs:
        raise GovernanceError(
            "certification must bind acceptance, execution, and operations configs"
        )
    for name, value in certified_hashes.items():
        if frozen_config_hashes.get(name) != value:
            raise GovernanceError(f"certification config hash mismatch: {name}")
    identities = payload["approved_identities"]
    identity_keys = {
        "scaler_sha256",
        "service_signing_fingerprint",
        "container_image_digest",
        "runtime_git_commit",
        "feature_snapshot_schema_version",
        "execution_config_sha256",
        "asset_tier_cost_map_sha256",
        "operations_config_sha256",
    }
    _require_keys(identities, identity_keys, "certification approved identities")
    for key in (
        "scaler_sha256",
        "execution_config_sha256",
        "asset_tier_cost_map_sha256",
        "operations_config_sha256",
    ):
        if not SHA256_PATTERN.fullmatch(str(identities[key])):
            raise GovernanceError(f"certification approved {key} is invalid")
    if not str(identities["service_signing_fingerprint"]).startswith("SHA256:"):
        raise GovernanceError("certification service-signing fingerprint is invalid")
    if identities["container_image_digest"] != container_digest:
        raise GovernanceError("certification approved container identity mismatch")
    if identities["runtime_git_commit"] != git_commit:
        raise GovernanceError("certification approved Git identity mismatch")
    if identities["execution_config_sha256"] != certified_hashes["execution"]:
        raise GovernanceError("certification approved execution config mismatch")
    if identities["operations_config_sha256"] != certified_hashes["operations"]:
        raise GovernanceError("certification approved operations config mismatch")
    if payload["performance_metrics_computed"] is not False:
        raise GovernanceError("certification must not compute holdout performance")
    completed_at = _parse_datetime(
        payload["certification_completed_at"], "certification_completed_at"
    )
    last_decision = _parse_date(
        certification_dates[-1], "last certification decision date"
    )
    if completed_at.date() < last_decision:
        raise GovernanceError("certification completed before its final decision")


def load_access_policy(root: Path, path: Path) -> dict[str, Any]:
    """Load three approved SSH identities and bind their key material."""
    config = read_yaml(resolve_path(root, path))
    _require_keys(
        config,
        {"schema_version", "status", "signature_namespace", "approvers"},
        "access-control config",
    )
    if config["schema_version"] != 1 or config["status"] != "approved":
        raise GovernanceError("access-control config is not approved")
    if config["signature_namespace"] != SIGNATURE_NAMESPACE:
        raise GovernanceError("unexpected SSH signature namespace")
    approvers = config["approvers"]
    if not isinstance(approvers, list) or len(approvers) != 3:
        raise GovernanceError("access policy requires exactly three approvers")
    required_roles = {role.value for role in ApprovalRole}
    observed_roles: set[str] = set()
    records: list[dict[str, str]] = []
    for approver in approvers:
        if not isinstance(approver, dict):
            raise GovernanceError("approver records must be mappings")
        _require_keys(
            approver,
            {
                "role",
                "name",
                "principal",
                "public_key_path",
                "public_key_sha256",
                "public_key_fingerprint",
            },
            "approver record",
        )
        role = str(approver["role"])
        if role not in required_roles or role in observed_roles:
            raise GovernanceError("access policy roles are incomplete or duplicated")
        if not isinstance(approver["name"], str) or not approver["name"].strip():
            raise GovernanceError("approver name is unresolved")
        if not isinstance(
            approver["principal"], str
        ) or not SSH_PRINCIPAL_PATTERN.fullmatch(approver["principal"]):
            raise GovernanceError("approver principal is unresolved or invalid")
        key_path = resolve_path(root, str(approver["public_key_path"]))
        key_sha = sha256_file(key_path)
        if key_sha != approver["public_key_sha256"]:
            raise GovernanceError("approver public key hash mismatch")
        fingerprint = ssh_public_key_fingerprint(key_path)
        if fingerprint != approver["public_key_fingerprint"]:
            raise GovernanceError("approver public key fingerprint mismatch")
        public_key = key_path.read_text(encoding="utf-8").strip()
        if len(public_key.splitlines()) != 1:
            raise GovernanceError(
                "approver public key file must contain exactly one key"
            )
        if not public_key.startswith(("ssh-ed25519 ", "ssh-rsa ", "ecdsa-sha2-")):
            raise GovernanceError("unsupported SSH public key format")
        observed_roles.add(role)
        records.append(
            {
                "role": role,
                "name": approver["name"].strip(),
                "principal": approver["principal"].strip(),
                "public_key": public_key,
                "public_key_sha256": key_sha,
                "public_key_fingerprint": fingerprint,
            }
        )
    if observed_roles != required_roles:
        raise GovernanceError("access policy must contain all three approval roles")
    return {
        "schema_version": 1,
        "signature_namespace": SIGNATURE_NAMESPACE,
        "approvers": sorted(records, key=lambda record: record["role"]),
    }


def verify_ssh_signature(
    *, challenge_bytes: bytes, signature: str, approver: dict[str, str]
) -> None:
    """Verify one role-bound detached SSH signature over the canonical challenge."""
    with tempfile.TemporaryDirectory(prefix="phase3b-signature-") as temp_name:
        temp = Path(temp_name)
        allowed = temp / "allowed_signers"
        signature_path = temp / "approval.sig"
        allowed.write_text(
            f"{approver['principal']} {approver['public_key']}\n", encoding="utf-8"
        )
        signature_path.write_text(signature, encoding="utf-8")
        result = subprocess.run(
            [
                "ssh-keygen",
                "-Y",
                "verify",
                "-f",
                str(allowed),
                "-I",
                approver["principal"],
                "-n",
                SIGNATURE_NAMESPACE,
                "-s",
                str(signature_path),
            ],
            input=challenge_bytes,
            check=False,
            capture_output=True,
        )
    if result.returncode != 0:
        raise GovernanceError(
            f"invalid SSH approval signature for role {approver['role']}"
        )


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=root, check=False, capture_output=True, text=True
    )
    if result.returncode != 0:
        raise GovernanceError(f"Git command failed: git {' '.join(args)}")
    return result.stdout


def _require_keys(payload: dict[str, Any], expected: set[str], label: str) -> None:
    observed = set(payload)
    if observed != expected:
        missing = sorted(expected - observed)
        unexpected = sorted(observed - expected)
        raise GovernanceError(
            f"{label} keys mismatch; missing={missing}, unexpected={unexpected}"
        )


def _parse_date(value: Any, label: str) -> date:
    if not isinstance(value, str):
        raise GovernanceError(f"{label} must be an ISO date")
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise GovernanceError(f"{label} must be an ISO date") from exc


def _parse_datetime(value: Any, label: str) -> datetime:
    if not isinstance(value, str):
        raise GovernanceError(f"{label} must be an ISO UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise GovernanceError(f"{label} must be an ISO UTC timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != UTC.utcoffset(parsed):
        raise GovernanceError(f"{label} must use UTC")
    return parsed


def _date_list(value: Any, label: str) -> list[date]:
    if not isinstance(value, list) or not value:
        raise GovernanceError(f"{label} must be a non-empty list")
    dates = [_parse_date(item, label) for item in value]
    if dates != sorted(set(dates)):
        raise GovernanceError(f"{label} must be unique and ordered")
    return dates


def _require_five_session_spacing(
    decisions: list[date], positions: dict[date, int], label: str
) -> None:
    try:
        indices = [positions[item] for item in decisions]
    except KeyError as exc:
        raise GovernanceError(f"{label} decision is not a trading session") from exc
    if any(right - left != 5 for left, right in pairwise(indices)):
        raise GovernanceError(f"{label} decisions are not five trading sessions apart")


def _calendar_anniversary(value: date) -> date:
    try:
        return value.replace(year=value.year + 1)
    except ValueError:
        return value.replace(year=value.year + 1, day=28)

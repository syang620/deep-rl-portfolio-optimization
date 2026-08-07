"""Separate unanimous authorization to begin Phase 3B certification."""

from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from portfolio_rl.phase3b.certification import (
    CERTIFICATION_APPROVAL_NAMESPACE,
    CertificationIdentity,
    build_certification_authorization_payload,
    verify_certification_authorization,
)
from portfolio_rl.phase3b.execution import load_execution_config
from portfolio_rl.phase3b.governance import (
    IDENTIFIER_PATTERN,
    ApprovalRole,
    GovernanceError,
    logical_json_sha256,
    read_json,
    read_yaml,
    relative_path,
    resolve_path,
    sha256_file,
)
from portfolio_rl.phase3b.identity_approval import verify_identity_approval
from portfolio_rl.phase3b.operational_metrics import load_operations_config
from portfolio_rl.phase3b.signatures import (
    create_signature_record,
    verify_signature_record,
)

ROLES = tuple(role.value for role in ApprovalRole)


@dataclass(frozen=True)
class ApprovedCertificationAuthorization:
    certification_id: str
    identity: CertificationIdentity
    identity_package_path: Path
    package_path: Path
    approved_start_date: date
    approved_end_date: date


def prepare_certification_authorization(
    *,
    repository_root: Path,
    identity_package_path: Path,
    certification_id: str,
    approved_start_date: date,
    approved_end_date: date,
    output_root: Path,
    created_at: datetime,
) -> Path:
    """Create an unsigned authorization bound to one finalized identity package."""
    root = repository_root.resolve()
    if not IDENTIFIER_PATTERN.fullmatch(certification_id):
        raise GovernanceError("certification ID is invalid")
    if approved_end_date < approved_start_date:
        raise GovernanceError("certification start window is invalid")
    _require_aware(created_at, "certification authorization creation")
    approved = verify_identity_approval(
        repository_root=root,
        package_path=identity_package_path,
        require_current_evidence=True,
    )
    output = (resolve_path(root, output_root) / certification_id).resolve()
    if output.exists():
        raise FileExistsError(f"certification authorization already exists: {output}")
    payload = build_certification_authorization_payload(
        certification_id=certification_id,
        identity=approved.identity,
        identity_package_manifest_sha256=sha256_file(
            approved.package_path / "identity_approval_manifest.json"
        ),
        identity_package_finalized_sha256=sha256_file(
            approved.package_path / "finalized_identity.json"
        ),
        approved_start_date=approved_start_date,
        approved_end_date=approved_end_date,
    )
    payload.update(
        {
            "created_at": created_at.astimezone(UTC).isoformat(),
            "signature_namespace": CERTIFICATION_APPROVAL_NAMESPACE,
            "identity_package_path": relative_path(root, approved.package_path),
        }
    )
    payload["authorization_payload_sha256"] = logical_json_sha256(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{certification_id}.", dir=output.parent)
    )
    try:
        _write_json(temporary / "authorization_challenge.json", payload)
        _write_json(temporary / "challenge.json", payload)
        (temporary / "challenge.sha256").write_text(
            sha256_file(temporary / "challenge.json") + "  challenge.json\n",
            encoding="utf-8",
        )
        (temporary / "SIGNING_INSTRUCTIONS.md").write_text(
            "# Certification authorization\n\nAll three roles independently sign the exact "
            "canonical challenge using external private keys. This authorization permits "
            "cycle 1 only after the separate identity package is finalized and readiness passes.\n",
            encoding="utf-8",
        )
        temporary.replace(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output


def sign_certification_authorization(
    *,
    repository_root: Path,
    package_path: Path,
    role: str,
    private_key_path: Path,
    signed_at: datetime,
) -> Path:
    """Sign one authorization role with an external key; never generate keys."""
    root = repository_root.resolve()
    package = resolve_path(root, package_path)
    challenge = _verify_challenge(package)
    if role not in ROLES:
        raise GovernanceError("certification authorization role is invalid")
    identity_package = resolve_path(root, challenge["identity_package_path"])
    approved = verify_identity_approval(
        repository_root=root,
        package_path=identity_package,
        require_current_evidence=True,
    )
    destination = package / "approvals" / f"{role}.json"
    if destination.exists():
        raise FileExistsError(
            f"certification authorization signature exists: {destination}"
        )
    access = read_yaml(approved.access_control_path)
    claim = next(row for row in access["approvers"] if row["role"] == role)
    record = create_signature_record(
        payload=challenge,
        payload_path="challenge.json",
        artifact_type="phase3b_certification_authorization",
        role=role,
        principal=claim["principal"],
        namespace=CERTIFICATION_APPROVAL_NAMESPACE,
        private_key_path=private_key_path.resolve(),
        public_key_path=identity_package / f"candidate/public_keys/{role}.pub",
        signed_at=signed_at,
    )
    _write_json(destination, record)
    return destination


def finalize_certification_authorization(
    *,
    repository_root: Path,
    package_path: Path,
    finalized_at: datetime,
) -> Path:
    """Finalize once only after all three exact role signatures verify."""
    root = repository_root.resolve()
    package = resolve_path(root, package_path)
    if (package / "finalized_authorization.json").exists():
        raise FileExistsError("certification authorization is already finalized")
    _require_aware(finalized_at, "certification authorization finalization")
    challenge = _verify_challenge(package)
    identity_package = resolve_path(root, challenge["identity_package_path"])
    approved = verify_identity_approval(
        repository_root=root,
        package_path=identity_package,
        require_current_evidence=True,
    )
    approvals = _verify_approvals(package, challenge, approved.package_path)
    finalized = {
        "schema_version": 1,
        "status": "finalized",
        "certification_id": challenge["certification_id"],
        "finalized_at": finalized_at.astimezone(UTC).isoformat(),
        "authorization_challenge_sha256": sha256_file(package / "challenge.json"),
        "authorization_logical_sha256": logical_json_sha256(challenge),
        "identity_sha256": approved.identity.identity_sha256,
        "approval_rule": "unanimous_three_of_three",
        "approvals": approvals,
    }
    finalized["finalized_payload_sha256"] = logical_json_sha256(finalized)
    _write_json(package / "finalized_authorization.json", finalized)
    _write_json(package / "authorization_manifest.json", _manifest(package))
    return package


def verify_finalized_certification_authorization(
    *,
    repository_root: Path,
    package_path: Path,
    require_current_evidence: bool = True,
) -> ApprovedCertificationAuthorization:
    """Verify authorization and its transitive finalized identity from a fresh process."""
    root = repository_root.resolve()
    package = resolve_path(root, package_path)
    challenge = _verify_challenge(package)
    finalized = read_json(package / "finalized_authorization.json")
    _payload_hash(finalized, "finalized_payload_sha256")
    if finalized.get("status") != "finalized":
        raise GovernanceError("certification authorization is not finalized")
    identity_package = resolve_path(root, challenge["identity_package_path"])
    approved = verify_identity_approval(
        repository_root=root,
        package_path=identity_package,
        require_current_evidence=require_current_evidence,
    )
    if challenge["identity_package_manifest_sha256"] != sha256_file(
        identity_package / "identity_approval_manifest.json"
    ) or challenge["identity_package_finalized_sha256"] != sha256_file(
        identity_package / "finalized_identity.json"
    ):
        raise GovernanceError("certification authorization identity package changed")
    records = _verify_approvals(package, challenge, identity_package)
    if finalized.get("approvals") != records:
        raise GovernanceError("finalized certification approval records differ")
    execution = load_execution_config(
        approved.execution_config_path, repository_root=root, require_approved=True
    )
    operations = load_operations_config(
        approved.operations_config_path, repository_root=root, require_approved=True
    )
    verify_certification_authorization(
        payload=challenge,
        approval_records={
            role: read_json(package / f"approvals/{role}.json") for role in ROLES
        },
        approver_public_keys={
            role: identity_package / f"candidate/public_keys/{role}.pub"
            for role in ROLES
        },
        execution_config=execution,
        operations_config=operations,
    )
    _verify_manifest(package)
    window = challenge["approved_start_window"]
    return ApprovedCertificationAuthorization(
        certification_id=challenge["certification_id"],
        identity=approved.identity,
        identity_package_path=identity_package,
        package_path=package,
        approved_start_date=date.fromisoformat(window["start_date"]),
        approved_end_date=date.fromisoformat(window["end_date"]),
    )


def _verify_challenge(package: Path) -> dict[str, Any]:
    challenge = read_json(package / "authorization_challenge.json")
    if read_json(package / "challenge.json") != challenge:
        raise GovernanceError("certification authorization challenge copies differ")
    _payload_hash(challenge, "authorization_payload_sha256")
    checksum = sha256_file(package / "challenge.json") + "  challenge.json\n"
    if (package / "challenge.sha256").read_text(encoding="utf-8") != checksum:
        raise GovernanceError("certification authorization checksum mismatch")
    if challenge.get("signature_namespace") != CERTIFICATION_APPROVAL_NAMESPACE:
        raise GovernanceError("certification authorization namespace mismatch")
    if challenge.get("authorization_type") != "phase3b_official_certification":
        raise GovernanceError("certification authorization type mismatch")
    for key in (
        "identity_package_manifest_sha256",
        "identity_package_finalized_sha256",
    ):
        value = challenge.get(key)
        if not isinstance(value, str) or len(value) != 64:
            raise GovernanceError(f"certification authorization {key} is invalid")
    try:
        window = challenge["approved_start_window"]
        start = date.fromisoformat(window["start_date"])
        end = date.fromisoformat(window["end_date"])
    except (KeyError, TypeError, ValueError) as exc:
        raise GovernanceError("certification authorization start window is invalid") from exc
    if end < start:
        raise GovernanceError("certification authorization start window is invalid")
    return challenge


def _verify_approvals(
    package: Path, challenge: dict[str, Any], identity: Path
) -> list[dict[str, Any]]:
    approval_dir = package / "approvals"
    observed = {
        path.name for path in approval_dir.iterdir() if path.is_file()
    } if approval_dir.exists() else set()
    expected = {f"{role}.json" for role in ROLES}
    if observed != expected:
        raise GovernanceError("certification authorization signature inventory mismatch")
    access = read_yaml(identity / "candidate/access_control.approved.yaml")
    claims = {row["role"]: row for row in access["approvers"]}
    results = []
    for role in ROLES:
        record = read_json(package / f"approvals/{role}.json")
        verify_signature_record(
            payload=challenge,
            record=record,
            public_key_path=identity / f"candidate/public_keys/{role}.pub",
            expected_role=role,
            expected_namespace=CERTIFICATION_APPROVAL_NAMESPACE,
        )
        if record["principal"] != claims[role]["principal"]:
            raise GovernanceError("certification authorization principal mismatch")
        results.append(
            {
                key: record[key]
                for key in (
                    "role",
                    "principal",
                    "key_fingerprint",
                    "canonical_payload_sha256",
                    "signature_timestamp",
                    "signature_algorithm",
                    "verification_status",
                )
            }
        )
    return results


def _manifest(package: Path) -> dict[str, Any]:
    files = [
        path
        for path in sorted(package.rglob("*"))
        if path.is_file() and path.name != "authorization_manifest.json"
    ]
    payload = {
        "schema_version": 1,
        "self_hash_contract": "manifest excludes itself",
        "files": [
            {
                "path": path.relative_to(package).as_posix(),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for path in files
        ],
    }
    payload["manifest_payload_sha256"] = logical_json_sha256(payload)
    return payload


def _verify_manifest(package: Path) -> None:
    manifest = read_json(package / "authorization_manifest.json")
    _payload_hash(manifest, "manifest_payload_sha256")
    expected = {
        path.relative_to(package).as_posix()
        for path in package.rglob("*")
        if path.is_file() and path.name != "authorization_manifest.json"
    }
    records = manifest.get("files")
    if (
        not isinstance(records, list)
        or {row.get("path") for row in records} != expected
    ):
        raise GovernanceError("certification authorization manifest inventory mismatch")
    for record in records:
        if sha256_file(package / record["path"]) != record["sha256"]:
            raise GovernanceError(
                f"certification authorization file changed: {record['path']}"
            )


def _payload_hash(payload: dict[str, Any], field: str) -> None:
    observed = payload.get(field)
    stripped = dict(payload)
    stripped.pop(field, None)
    if observed != logical_json_sha256(stripped):
        raise GovernanceError("certification authorization payload hash mismatch")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(
            f"refusing to overwrite certification authorization: {path}"
        )
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _require_aware(value: datetime, label: str) -> None:
    if value.tzinfo is None:
        raise GovernanceError(f"{label} timestamp must be timezone-aware")

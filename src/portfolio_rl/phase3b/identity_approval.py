"""Evidence-backed approval of exact Phase 3B certification runtime identities."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from portfolio_rl.phase3b.certification import CertificationIdentity
from portfolio_rl.phase3b.governance import (
    EXPECTED_CANDIDATE_MANIFEST_SHA256,
    IDENTIFIER_PATTERN,
    ApprovalRole,
    GovernanceError,
    canonical_json_bytes,
    logical_json_sha256,
    read_json,
    read_yaml,
    relative_path,
    resolve_path,
    sha256_file,
    ssh_public_key_fingerprint,
    validate_container_identity,
    verify_candidate,
)
from portfolio_rl.phase3b.operational_metrics import (
    decode_sealing_public_key,
    sealing_key_fingerprint,
)
from portfolio_rl.phase3b.scaler_reconciliation import reconcile_frozen_scaler
from portfolio_rl.phase3b.signatures import (
    create_signature_record,
    verify_signature_record,
)

EXPECTED_PR22_MERGE_SHA = "f53f64afeff1638302977b1f7b30979f488fcd43"
IDENTITY_APPROVAL_NAMESPACE = "portfolio-rl-phase3b-identity-approval-v1"
APPROVER_ROLES = tuple(role.value for role in ApprovalRole)
PREPARED_FILES = (
    "candidate/access_control.approved.yaml",
    "candidate/execution.approved.yaml",
    "candidate/operations.approved.yaml",
    "candidate/container_identity.json",
    "candidate/public_keys/service_signing.pub",
    "candidate/public_keys/performance_sealing.pub",
    "candidate/public_keys/portfolio_manager.pub",
    "candidate/public_keys/independent_reviewer.pub",
    "candidate/public_keys/data_operations_custodian.pub",
    "evidence/scaler_reconciliation.json",
    "evidence/test_access_audit.json",
    "evidence/certification_registry_state.json",
    "evidence/holdout_registry_state.json",
)


@dataclass(frozen=True)
class ApprovedRuntimeIdentity:
    """Exact finalized identities authorized for later certification approval."""

    approval_id: str
    pr22_merge_sha: str
    identity_tooling_merge_sha: str
    container_digest: str
    identity: CertificationIdentity
    execution_config_path: Path
    operations_config_path: Path
    access_control_path: Path
    package_path: Path


def prepare_identity_approval(
    *,
    repository_root: Path,
    config_path: Path,
    created_at: datetime,
) -> Path:
    """Create an immutable unsigned identity challenge from verified evidence."""
    root = repository_root.resolve()
    config = _load_input_config(resolve_path(root, config_path))
    approval_id = _text(config["approval_id"], "approval_id")
    if not IDENTIFIER_PATTERN.fullmatch(approval_id):
        raise GovernanceError("identity approval ID is invalid")
    output_root = resolve_path(root, _text(config["output_root"], "output root"))
    output = (output_root / approval_id).resolve()
    if output.exists():
        raise FileExistsError(f"identity approval package already exists: {output}")
    tooling_sha = _verify_runtime_git(
        root, pr22_merge_sha=str(config["pr22_merge_sha"])
    )
    pretest_package = resolve_path(
        root, _text(config["pretest_package"], "pretest package path")
    )
    verified_candidate = verify_candidate(
        root,
        pretest_package / "frozen_candidate.json",
        EXPECTED_CANDIDATE_MANIFEST_SHA256,
    )
    container_path = resolve_path(
        root, _text(config["container_identity_path"], "container identity path")
    )
    container = read_json(container_path)
    validate_container_identity(container, tooling_sha)

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{approval_id}.", dir=output.parent))
    try:
        _prepare_public_keys(root, temporary, config)
        reconciliation = _scaler_reconciliation(root)
        _write_json(temporary / "evidence/scaler_reconciliation.json", reconciliation)
        evidence = _governance_evidence(root)
        for name, payload in evidence.items():
            _write_json(temporary / f"evidence/{name}.json", payload)
        _write_json(temporary / "candidate/container_identity.json", container)
        configs = _approved_configs(
            root, storage_package=temporary, final_package=output, config=config
        )
        for name, payload in configs.items():
            _write_yaml(temporary / f"candidate/{name}.approved.yaml", payload)
        identity = _certification_identity(
            temporary=temporary,
            tooling_sha=tooling_sha,
            container=container,
            execution=configs["execution"],
            operations=configs["operations"],
        )
        challenge = _challenge(
            root=root,
            package=temporary,
            final_package=output,
            approval_id=approval_id,
            pr22_merge_sha=str(config["pr22_merge_sha"]),
            tooling_sha=tooling_sha,
            container=container,
            identity=identity,
            candidate_path=verified_candidate.frozen_candidate_path,
            candidate_manifest_path=verified_candidate.freeze_manifest_path,
            created_at=created_at,
        )
        _write_json(temporary / "identity_challenge.json", challenge)
        temporary.replace(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output


def sign_identity_approval(
    *,
    package_path: Path,
    role: str,
    private_key_path: Path,
    signed_at: datetime,
) -> Path:
    """Sign the canonical identity challenge with one external role key."""
    package = package_path.resolve()
    challenge = _verify_prepared_package(package)
    if package.name != challenge.get("approval_id"):
        raise GovernanceError("identity approval package directory mismatch")
    if role not in APPROVER_ROLES:
        raise GovernanceError("identity approval role is invalid")
    destination = package / "approvals" / f"{role}.json"
    if destination.exists():
        raise FileExistsError(f"identity approval already exists: {destination}")
    public_key = package / f"candidate/public_keys/{role}.pub"
    record = create_signature_record(
        payload=challenge,
        payload_path="identity_challenge.json",
        artifact_type="phase3b_runtime_identity_approval",
        role=role,
        principal=_approver_principal(challenge, role),
        namespace=IDENTITY_APPROVAL_NAMESPACE,
        private_key_path=private_key_path.resolve(),
        public_key_path=public_key,
        signed_at=signed_at,
    )
    _write_json(destination, record)
    return destination


def finalize_identity_approval(
    *, repository_root: Path, package_path: Path, finalized_at: datetime
) -> Path:
    """Verify unanimous signatures and unchanged evidence, then finalize once."""
    root = repository_root.resolve()
    package = resolve_path(root, package_path)
    if (package / "finalized_identity.json").exists():
        raise FileExistsError("identity approval package is already finalized")
    challenge = _verify_prepared_package(package)
    if relative_path(root, package) != challenge.get("package_path"):
        raise GovernanceError("identity approval package path mismatch")
    _verify_runtime_git(root, pr22_merge_sha=challenge["provenance"]["pr22_merge_sha"])
    _verify_current_evidence(root, package, challenge)
    approvals = _verify_approvals(package, challenge)
    finalized = {
        "schema_version": 1,
        "approval_id": challenge["approval_id"],
        "status": "finalized",
        "finalized_at": _utc(finalized_at).isoformat(),
        "challenge_sha256": sha256_file(package / "identity_challenge.json"),
        "challenge_logical_sha256": logical_json_sha256(challenge),
        "approval_rule": "unanimous_three_of_three",
        "approval_record_hashes": {
            role: sha256_file(package / f"approvals/{role}.json")
            for role in APPROVER_ROLES
        },
        "approvals": approvals,
        "provenance": challenge["provenance"],
        "identity": challenge["identity"],
        "governance_evidence": challenge["governance_evidence"],
        "test_accessed": False,
        "official_certification_started": False,
        "canonical_holdout_registered": False,
    }
    finalized["finalized_payload_sha256"] = logical_json_sha256(finalized)
    _write_json(package / "finalized_identity.json", finalized)
    manifest = _final_manifest(package)
    _write_json(package / "identity_approval_manifest.json", manifest)
    return package


def verify_identity_approval(
    *,
    repository_root: Path,
    package_path: Path,
    require_current_evidence: bool = True,
) -> ApprovedRuntimeIdentity:
    """Verify every finalized input, signature, manifest, and governance proof."""
    root = repository_root.resolve()
    package = resolve_path(root, package_path)
    challenge = _verify_prepared_package(package)
    if relative_path(root, package) != challenge.get("package_path"):
        raise GovernanceError("identity approval package path mismatch")
    finalized = read_json(package / "finalized_identity.json")
    _verify_payload_hash(finalized, "finalized_payload_sha256", "finalized identity")
    if finalized.get("status") != "finalized":
        raise GovernanceError("identity approval is not finalized")
    if finalized.get("challenge_logical_sha256") != logical_json_sha256(challenge):
        raise GovernanceError("finalized identity challenge mismatch")
    _verify_approvals(package, challenge)
    _verify_final_manifest(package)
    tooling_sha = _verify_runtime_git(
        root, pr22_merge_sha=challenge["provenance"]["pr22_merge_sha"]
    )
    if tooling_sha != challenge["provenance"]["identity_tooling_merge_sha"]:
        raise GovernanceError("current runtime Git identity differs from approval")
    candidate = challenge.get("candidate", {})
    verify_candidate(
        root,
        Path(candidate.get("frozen_candidate_path", "")),
        str(candidate.get("candidate_manifest_sha256", "")),
    )
    if require_current_evidence:
        _verify_current_evidence(root, package, challenge)
    identity = CertificationIdentity(**challenge["identity"])
    return ApprovedRuntimeIdentity(
        approval_id=challenge["approval_id"],
        pr22_merge_sha=challenge["provenance"]["pr22_merge_sha"],
        identity_tooling_merge_sha=tooling_sha,
        container_digest=challenge["provenance"]["container_digest"],
        identity=identity,
        execution_config_path=package / "candidate/execution.approved.yaml",
        operations_config_path=package / "candidate/operations.approved.yaml",
        access_control_path=package / "candidate/access_control.approved.yaml",
        package_path=package,
    )


def _load_input_config(path: Path) -> dict[str, Any]:
    payload = read_yaml(path)
    expected = {
        "schema_version",
        "status",
        "approval_id",
        "pr22_merge_sha",
        "container_identity_path",
        "service_signing",
        "performance_sealing",
        "approvers",
        "draft_configs",
        "pretest_package",
        "output_root",
    }
    if set(payload) != expected or payload["schema_version"] != 1:
        raise GovernanceError("identity approval input schema mismatch")
    if payload["status"] != "ready_for_preparation":
        raise GovernanceError("identity approval input is still draft")
    if payload["pr22_merge_sha"] != EXPECTED_PR22_MERGE_SHA:
        raise GovernanceError("PR 22 merge provenance mismatch")
    return payload


def _verify_runtime_git(root: Path, *, pr22_merge_sha: str) -> str:
    if _git(root, "diff", "--quiet", check=False).returncode != 0 or _git(
        root, "diff", "--cached", "--quiet", check=False
    ).returncode != 0:
        raise GovernanceError("tracked working tree must be clean")
    head = _git_text(root, "rev-parse", "HEAD")
    origin_main = _git_text(root, "rev-parse", "origin/main")
    remote_main_result = _git(root, "ls-remote", "origin", "refs/heads/main")
    remote_fields = remote_main_result.stdout.split()
    if len(remote_fields) != 2 or remote_fields[1] != "refs/heads/main":
        raise GovernanceError("cannot resolve authoritative remote main")
    remote_main = remote_fields[0]
    if head != origin_main or head != remote_main:
        raise GovernanceError("runtime Git HEAD must equal origin/main")
    if _git(root, "merge-base", "--is-ancestor", pr22_merge_sha, head, check=False).returncode != 0:
        raise GovernanceError("PR 22 merge must be an ancestor of runtime Git HEAD")
    return head


def _prepare_public_keys(root: Path, package: Path, config: dict[str, Any]) -> None:
    service = config["service_signing"]
    sealing = config["performance_sealing"]
    if set(service) != {"principal", "public_key_path"} or set(sealing) != {
        "public_key_path"
    }:
        raise GovernanceError("identity service or sealing key schema mismatch")
    _text(service["principal"], "service signing principal")
    records = config["approvers"]
    if not isinstance(records, list) or {row.get("role") for row in records} != set(
        APPROVER_ROLES
    ):
        raise GovernanceError("identity approval requires all three approvers")
    key_dir = package / "candidate/public_keys"
    key_dir.mkdir(parents=True)
    service_source = resolve_path(
        root, _text(service["public_key_path"], "service public-key path")
    )
    shutil.copy2(service_source, key_dir / "service_signing.pub")
    fingerprints = {ssh_public_key_fingerprint(service_source)}
    sealing_source = resolve_path(
        root, _text(sealing["public_key_path"], "sealing public-key path")
    )
    decode_sealing_public_key(sealing_source)
    shutil.copy2(sealing_source, key_dir / "performance_sealing.pub")
    for record in records:
        if set(record) != {"role", "name", "principal", "public_key_path"}:
            raise GovernanceError("identity approver schema mismatch")
        _text(record["name"], "approver name")
        _text(record["principal"], "approver principal")
        source = resolve_path(
            root, _text(record["public_key_path"], "approver public-key path")
        )
        fingerprint = ssh_public_key_fingerprint(source)
        if fingerprint in fingerprints:
            raise GovernanceError("service and approver SSH keys must be distinct")
        fingerprints.add(fingerprint)
        shutil.copy2(source, key_dir / f"{record['role']}.pub")


def _scaler_reconciliation(root: Path) -> dict[str, object]:
    return reconcile_frozen_scaler(
        scaler_path=root / "artifacts/scalers/feature_scaler_v1.pkl",
        feature_spec_path=root / "artifacts/feature_specs/feature_spec_v1.json",
        raw_asset_features_path=root / "data/processed/features_daily.parquet",
        normalized_asset_features_path=root
        / "data/processed/features_normalized_daily.parquet",
        raw_global_features_path=root / "data/processed/global_features_daily.parquet",
        normalized_global_features_path=root
        / "data/processed/global_features_normalized_daily.parquet",
        model_matrix_path=root / "data/processed/model_matrix_daily.parquet",
    ).to_payload()


def _governance_evidence(root: Path) -> dict[str, dict[str, Any]]:
    return {
        "test_access_audit": _test_access_audit(root),
        "certification_registry_state": _registry_state(
            root, root / "artifacts/phase3b/certification", "certification"
        ),
        "holdout_registry_state": _registry_state(
            root, root / "artifacts/phase3b/registration", "holdout"
        ),
    }


def _test_access_audit(root: Path) -> dict[str, Any]:
    pretest_path = root / (
        "artifacts/pretest_freeze/ppo_v1_ensemble5_alpha025_pretest_v1/"
        "test_access_audit.json"
    )
    prior = read_json(pretest_path)
    known = {
        str(record["metadata_path"]): str(record["metadata_sha256"])
        for record in prior.get("known_legacy_access", [])
    }
    findings = []
    unexpected = []
    for path in sorted((root / "artifacts").rglob("*.json")):
        try:
            payload = read_json(path)
        except GovernanceError:
            continue
        if not _declares_test_access(payload):
            continue
        relative = relative_path(root, path)
        finding = {
            "path": relative,
            "sha256": sha256_file(path),
            "declared_legacy": relative in known,
        }
        findings.append(finding)
        if relative not in known or known[relative] != finding["sha256"]:
            unexpected.append(relative)
    if unexpected or set(known) != {row["path"] for row in findings}:
        raise GovernanceError("test-access audit does not reconcile to legacy allowlist")
    return {
        "schema_version": 1,
        "policy": "fail_on_new_or_mutated_test_access",
        "pretest_audit_path": relative_path(root, pretest_path),
        "pretest_audit_sha256": sha256_file(pretest_path),
        "known_legacy_access": findings,
        "unexpected_test_access": [],
        "test_accessed": False,
    }


def _registry_state(root: Path, registry: Path, kind: str) -> dict[str, Any]:
    files = (
        [path for path in sorted(registry.rglob("*")) if path.is_file()]
        if registry.exists()
        else []
    )
    inventory = [
        {
            "path": relative_path(root, path),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in files
    ]
    official_cycles = 0
    registrations = 0
    for path in files:
        if path.suffix != ".json":
            continue
        try:
            payload = read_json(path)
        except GovernanceError as exc:
            raise GovernanceError(
                f"cannot verify {kind} registry JSON: {path.name}"
            ) from exc
        official_cycles += int(payload.get("official") is True)
        registrations += int(path.name == "holdout_registration.json")
    if kind == "certification" and official_cycles:
        raise GovernanceError("official certification has already started")
    if kind == "holdout" and registrations:
        raise GovernanceError("canonical holdout is already registered")
    return {
        "schema_version": 1,
        "registry_kind": kind,
        "registry_root": relative_path(root, registry),
        "registry_exists": registry.exists(),
        "inventory": inventory,
        "inventory_logical_sha256": logical_json_sha256(inventory),
        "official_cycle_count": official_cycles,
        "canonical_registration_count": registrations,
    }


def _approved_configs(
    root: Path,
    *,
    storage_package: Path,
    final_package: Path,
    config: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    draft = config["draft_configs"]
    if set(draft) != {"execution", "operations", "access_control"}:
        raise GovernanceError("identity draft config inventory mismatch")
    execution = read_yaml(
        resolve_path(root, _text(draft["execution"], "draft execution config"))
    )
    operations = read_yaml(
        resolve_path(root, _text(draft["operations"], "draft operations config"))
    )
    access = read_yaml(
        resolve_path(root, _text(draft["access_control"], "draft access config"))
    )
    if any(
        payload.get("status") != "draft"
        for payload in (execution, operations, access)
    ):
        raise GovernanceError("tracked identity configuration templates must be draft")
    storage_keys = storage_package / "candidate/public_keys"
    final_keys = final_package / "candidate/public_keys"
    approvers = {row["role"]: row for row in config["approvers"]}
    execution["status"] = "approved"
    scaler = execution["snapshot_contract"]["scaler"]
    scaler["status"] = "approved_for_phase3b"
    scaler["approved_hash"] = scaler["configured_hash"]
    scaler["approved_by"] = list(APPROVER_ROLES)
    service_path = storage_keys / "service_signing.pub"
    execution["recommendation_signing"] = {
        "namespace": execution["recommendation_signing"]["namespace"],
        "principal": config["service_signing"]["principal"],
        "public_key_path": relative_path(root, final_keys / "service_signing.pub"),
        "public_key_sha256": sha256_file(service_path),
        "public_key_fingerprint": ssh_public_key_fingerprint(service_path),
        "approval_status": "approved",
    }
    operations["status"] = "approved"
    sealing_path = storage_keys / "performance_sealing.pub"
    sealing_raw = decode_sealing_public_key(sealing_path)
    operations["sealed_ledger"].update(
        {
            "recipient_public_key_path": relative_path(
                root, final_keys / "performance_sealing.pub"
            ),
            "recipient_public_key_sha256": sha256_file(sealing_path),
            "recipient_key_fingerprint": sealing_key_fingerprint(sealing_raw),
            "approval_status": "approved_for_phase3b",
        }
    )
    access["status"] = "approved"
    access["approvers"] = [
        {
            "role": role,
            "name": approvers[role]["name"],
            "principal": approvers[role]["principal"],
            "public_key_path": relative_path(
                root, final_keys / f"{role}.pub"
            ),
            "public_key_sha256": sha256_file(storage_keys / f"{role}.pub"),
            "public_key_fingerprint": ssh_public_key_fingerprint(
                storage_keys / f"{role}.pub"
            ),
        }
        for role in APPROVER_ROLES
    ]
    return {"execution": execution, "operations": operations, "access_control": access}


def _certification_identity(
    *,
    temporary: Path,
    tooling_sha: str,
    container: dict[str, Any],
    execution: dict[str, Any],
    operations: dict[str, Any],
) -> CertificationIdentity:
    costs = execution["costs"]["asset_cost_bps"]
    return CertificationIdentity(
        scaler_sha256=execution["snapshot_contract"]["scaler"]["approved_hash"],
        service_signing_fingerprint=execution["recommendation_signing"][
            "public_key_fingerprint"
        ],
        container_image_digest=container["image_digest"],
        runtime_git_commit=tooling_sha,
        feature_snapshot_schema_version=execution["snapshot_contract"][
            "feature_payload_schema_version"
        ],
        execution_config_sha256=sha256_file(
            temporary / "candidate/execution.approved.yaml"
        ),
        asset_tier_cost_map_sha256=logical_json_sha256(costs),
        operations_config_sha256=sha256_file(
            temporary / "candidate/operations.approved.yaml"
        ),
        access_control_config_sha256=sha256_file(
            temporary / "candidate/access_control.approved.yaml"
        ),
        candidate_manifest_sha256=execution["candidate"][
            "candidate_manifest_sha256"
        ],
        feature_spec_sha256=execution["snapshot_contract"]["scaler"][
            "feature_spec_hash"
        ],
        performance_sealing_fingerprint=operations["sealed_ledger"][
            "recipient_key_fingerprint"
        ],
    )


def _challenge(
    *,
    root: Path,
    package: Path,
    final_package: Path,
    approval_id: str,
    pr22_merge_sha: str,
    tooling_sha: str,
    container: dict[str, Any],
    identity: CertificationIdentity,
    candidate_path: Path,
    candidate_manifest_path: Path,
    created_at: datetime,
) -> dict[str, Any]:
    evidence = {}
    for name in (
        "test_access_audit",
        "certification_registry_state",
        "holdout_registry_state",
    ):
        path = package / f"evidence/{name}.json"
        payload = read_json(path)
        evidence[name] = {
            "path": f"evidence/{name}.json",
            "sha256": sha256_file(path),
            "logical_sha256": logical_json_sha256(payload),
        }
    challenge = {
        "schema_version": 1,
        "approval_id": approval_id,
        "package_path": relative_path(root, final_package),
        "created_at": _utc(created_at).isoformat(),
        "signature_namespace": IDENTITY_APPROVAL_NAMESPACE,
        "provenance": {
            "pr22_merge_sha": pr22_merge_sha,
            "identity_tooling_merge_sha": tooling_sha,
            "container_digest": container["image_digest"],
        },
        "identity": identity.__dict__,
        "candidate": {
            "frozen_candidate_path": relative_path(root, candidate_path),
            "frozen_candidate_sha256": sha256_file(candidate_path),
            "candidate_manifest_path": relative_path(root, candidate_manifest_path),
            "candidate_manifest_sha256": sha256_file(candidate_manifest_path),
        },
        "approvers": _approver_claims(package),
        "prepared_files": [_file_record(package, name) for name in PREPARED_FILES],
        "governance_evidence": evidence,
        "test_accessed": False,
        "official_certification_started": False,
        "canonical_holdout_registered": False,
        "private_keys_recorded": False,
        "repository_root_binding": root.name,
    }
    challenge["challenge_payload_sha256"] = logical_json_sha256(challenge)
    return challenge


def _verify_prepared_package(package: Path) -> dict[str, Any]:
    challenge = read_json(package / "identity_challenge.json")
    _verify_payload_hash(challenge, "challenge_payload_sha256", "identity challenge")
    if challenge.get("signature_namespace") != IDENTITY_APPROVAL_NAMESPACE:
        raise GovernanceError("identity challenge namespace mismatch")
    records = challenge.get("prepared_files")
    if not isinstance(records, list) or {row.get("path") for row in records} != set(
        PREPARED_FILES
    ):
        raise GovernanceError("identity prepared file inventory mismatch")
    for record in records:
        path = package / record["path"]
        if sha256_file(path) != record["sha256"]:
            raise GovernanceError(f"identity prepared file hash mismatch: {record['path']}")
    return challenge


def _verify_current_evidence(
    root: Path, package: Path, challenge: dict[str, Any]
) -> None:
    current = _governance_evidence(root)
    for name, payload in current.items():
        recorded = challenge["governance_evidence"][name]
        if logical_json_sha256(payload) != recorded["logical_sha256"]:
            raise GovernanceError(f"governance evidence changed after challenge: {name}")
        if logical_json_sha256(read_json(package / recorded["path"])) != recorded[
            "logical_sha256"
        ]:
            raise GovernanceError(f"packaged governance evidence mismatch: {name}")


def _verify_approvals(
    package: Path, challenge: dict[str, Any]
) -> list[dict[str, Any]]:
    claims = {row["role"]: row for row in challenge["approvers"]}
    results = []
    for role in APPROVER_ROLES:
        path = package / f"approvals/{role}.json"
        record = read_json(path)
        verify_signature_record(
            payload=challenge,
            record=record,
            public_key_path=package / f"candidate/public_keys/{role}.pub",
            expected_role=role,
            expected_namespace=IDENTITY_APPROVAL_NAMESPACE,
        )
        if record["principal"] != claims[role]["principal"]:
            raise GovernanceError("identity approval principal mismatch")
        results.append(
            {
                "role": role,
                "principal": record["principal"],
                "key_fingerprint": record["key_fingerprint"],
                "canonical_payload_sha256": record["canonical_payload_sha256"],
                "signature_timestamp": record["signature_timestamp"],
                "signature_algorithm": record["signature_algorithm"],
                "verification_status": "verified",
            }
        )
    return results


def _final_manifest(package: Path) -> dict[str, Any]:
    files = [
        path
        for path in sorted(package.rglob("*"))
        if path.is_file() and path.name != "identity_approval_manifest.json"
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


def _verify_final_manifest(package: Path) -> None:
    manifest = read_json(package / "identity_approval_manifest.json")
    _verify_payload_hash(manifest, "manifest_payload_sha256", "identity manifest")
    expected = {
        path.relative_to(package).as_posix()
        for path in package.rglob("*")
        if path.is_file() and path.name != "identity_approval_manifest.json"
    }
    records = manifest.get("files")
    if not isinstance(records, list) or {row.get("path") for row in records} != expected:
        raise GovernanceError("identity final manifest inventory mismatch")
    for record in records:
        if sha256_file(package / record["path"]) != record["sha256"]:
            raise GovernanceError(f"identity final artifact mutation: {record['path']}")


def _approver_claims(package: Path) -> list[dict[str, str]]:
    access = read_yaml(package / "candidate/access_control.approved.yaml")
    return [
        {
            "role": row["role"],
            "name": row["name"],
            "principal": row["principal"],
            "public_key_sha256": row["public_key_sha256"],
            "public_key_fingerprint": row["public_key_fingerprint"],
        }
        for row in access["approvers"]
    ]


def _approver_principal(challenge: dict[str, Any], role: str) -> str:
    matches = [row["principal"] for row in challenge["approvers"] if row["role"] == role]
    if len(matches) != 1:
        raise GovernanceError("identity approver claim is missing or ambiguous")
    return matches[0]


def _file_record(package: Path, name: str) -> dict[str, Any]:
    path = package / name
    return {"path": name, "sha256": sha256_file(path), "size_bytes": path.stat().st_size}


def _declares_test_access(payload: dict[str, Any]) -> bool:
    split = str(payload.get("split", "")).lower()
    status = str(payload.get("final_test_status", "")).lower()
    return split == "test" or payload.get("test_split_used") is True or status not in {
        "",
        "not_run",
        "not_run_blocked_pending_new_independent_holdout",
    }


def _verify_payload_hash(payload: dict[str, Any], field: str, label: str) -> None:
    content = dict(payload)
    recorded = content.pop(field, None)
    if logical_json_sha256(content) != recorded:
        raise GovernanceError(f"{label} payload hash mismatch")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(payload))


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise GovernanceError(f"{label} is unresolved")
    return value.strip()


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise GovernanceError("identity approval timestamp must be timezone-aware")
    return value.astimezone(UTC)


def _git(root: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["git", *args], cwd=root, check=False, capture_output=True, text=True
    )
    if check and result.returncode != 0:
        raise GovernanceError(f"Git command failed: git {' '.join(args)}")
    return result


def _git_text(root: Path, *args: str) -> str:
    return _git(root, *args).stdout.strip()

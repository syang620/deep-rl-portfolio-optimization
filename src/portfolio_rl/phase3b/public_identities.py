"""Validation of externally generated Phase 3B public identities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from portfolio_rl.phase3b.governance import (
    ApprovalRole,
    GovernanceError,
    sha256_file,
    ssh_public_key_fingerprint,
)
from portfolio_rl.phase3b.operational_metrics import (
    decode_sealing_public_key,
    sealing_key_fingerprint,
)


def inspect_public_identities(
    *,
    service_signing_key: Path,
    performance_sealing_key: Path,
    approver_keys: dict[str, Path],
) -> dict[str, Any]:
    """Validate public-only, role-separated identities without copying key data."""
    roles = tuple(role.value for role in ApprovalRole)
    if set(approver_keys) != set(roles):
        raise GovernanceError(
            "public identity inventory requires all three approver roles"
        )
    ssh_paths = {"service_signing": service_signing_key, **approver_keys}
    records: dict[str, dict[str, str]] = {}
    fingerprints: dict[str, str] = {}
    for name, path in ssh_paths.items():
        _require_ssh_public_key(path)
        fingerprint = ssh_public_key_fingerprint(path)
        fingerprints[name] = fingerprint
        records[name] = {
            "key_type": "openssh_public_key",
            "sha256": sha256_file(path),
            "fingerprint": fingerprint,
        }
    if len(set(fingerprints.values())) != len(fingerprints):
        raise GovernanceError("service and approver SSH keys must be distinct")
    _reject_private_key_material(performance_sealing_key)
    sealing_raw = decode_sealing_public_key(performance_sealing_key)
    sealing_sha = sha256_file(performance_sealing_key)
    sealing_fingerprint = sealing_key_fingerprint(sealing_raw)
    if sealing_sha in {record["sha256"] for record in records.values()}:
        raise GovernanceError("performance sealing key must be a separate identity")
    records["performance_sealing"] = {
        "key_type": "curve25519_public_key",
        "sha256": sealing_sha,
        "fingerprint": sealing_fingerprint,
    }
    return {
        "schema_version": 1,
        "public_keys_only": True,
        "service_key_is_not_approver": True,
        "service_key_is_not_unseal_authority": True,
        "sealing_identity_is_distinct": True,
        "identities": records,
    }


def _require_ssh_public_key(path: Path) -> None:
    _reject_private_key_material(path)
    try:
        fields = path.read_text(encoding="utf-8").strip().split()
    except OSError as exc:
        raise GovernanceError(f"cannot read public key: {path}") from exc
    if len(fields) < 2 or not fields[0].startswith(("ssh-", "ecdsa-")):
        raise GovernanceError("SSH identity input must be an OpenSSH public key")


def _reject_private_key_material(path: Path) -> None:
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise GovernanceError(f"cannot read public identity: {path}") from exc
    upper = data.upper()
    if b"PRIVATE KEY" in upper or b"BEGIN OPENSSH" in upper or b"BEGIN RSA" in upper:
        raise GovernanceError("private key input is forbidden")

"""Role-bound signature records for Phase 3B operational artifacts."""

from __future__ import annotations

import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from portfolio_rl.phase3b.governance import (
    GovernanceError,
    canonical_json_bytes,
    logical_json_sha256,
    ssh_public_key_fingerprint,
)

SIGNATURE_ALGORITHM = "openssh_sshsig_sha256"


def create_signature_record(
    *,
    payload: dict[str, Any],
    payload_path: str,
    artifact_type: str,
    role: str,
    principal: str,
    namespace: str,
    private_key_path: Path,
    public_key_path: Path,
    signed_at: datetime,
) -> dict[str, Any]:
    """Sign one canonical artifact hash and return its auditable record."""
    if signed_at.tzinfo is None:
        raise GovernanceError("signature timestamp must be timezone-aware")
    public_key = public_key_path.read_text(encoding="utf-8").strip()
    public_key_algorithm = public_key.split(maxsplit=1)[0]
    fingerprint = ssh_public_key_fingerprint(public_key_path)
    _verify_private_key(private_key_path, fingerprint)
    claim = {
        "schema_version": 1,
        "artifact_type": _identifier(artifact_type, "artifact_type"),
        "payload_path": _identifier(payload_path, "payload_path"),
        "canonical_payload_sha256": logical_json_sha256(payload),
        "key_fingerprint": fingerprint,
        "role": _identifier(role, "role"),
        "principal": _identifier(principal, "principal"),
        "signature_timestamp": signed_at.astimezone(UTC).isoformat(),
        "signature_algorithm": SIGNATURE_ALGORITHM,
        "public_key_algorithm": public_key_algorithm,
        "namespace": _identifier(namespace, "namespace"),
    }
    signature = _sign_bytes(
        canonical_json_bytes(claim), private_key_path=private_key_path, namespace=namespace
    )
    return {**claim, "signature": signature, "verification_status": "verified"}


def verify_signature_record(
    *,
    payload: dict[str, Any],
    record: dict[str, Any],
    public_key_path: Path,
    expected_role: str,
    expected_namespace: str,
) -> None:
    """Recompute the record and verify its detached SSH signature."""
    expected_keys = {
        "schema_version",
        "artifact_type",
        "payload_path",
        "canonical_payload_sha256",
        "key_fingerprint",
        "role",
        "principal",
        "signature_timestamp",
        "signature_algorithm",
        "public_key_algorithm",
        "namespace",
        "signature",
        "verification_status",
    }
    if set(record) != expected_keys or record["schema_version"] != 1:
        raise GovernanceError("signature record schema mismatch")
    if record["canonical_payload_sha256"] != logical_json_sha256(payload):
        raise GovernanceError("signature record payload hash mismatch")
    if record["role"] != expected_role or record["namespace"] != expected_namespace:
        raise GovernanceError("signature record role or namespace mismatch")
    if record["signature_algorithm"] != SIGNATURE_ALGORITHM:
        raise GovernanceError("signature algorithm mismatch")
    if record["verification_status"] != "verified":
        raise GovernanceError("signature record is not marked verified")
    fingerprint = ssh_public_key_fingerprint(public_key_path)
    if record["key_fingerprint"] != fingerprint:
        raise GovernanceError("signature record key fingerprint mismatch")
    public_key = public_key_path.read_text(encoding="utf-8").strip()
    if record["public_key_algorithm"] != public_key.split(maxsplit=1)[0]:
        raise GovernanceError("signature record public-key algorithm mismatch")
    claim = {
        key: value
        for key, value in record.items()
        if key not in {"signature", "verification_status"}
    }
    _verify_bytes(
        canonical_json_bytes(claim),
        signature=str(record["signature"]),
        public_key=public_key,
        principal=str(record["principal"]),
        namespace=expected_namespace,
    )


def _sign_bytes(data: bytes, *, private_key_path: Path, namespace: str) -> str:
    with tempfile.TemporaryDirectory(prefix="phase3b-sign-") as directory_name:
        message = Path(directory_name) / "payload.json"
        message.write_bytes(data)
        result = subprocess.run(
            [
                "ssh-keygen",
                "-Y",
                "sign",
                "-f",
                str(private_key_path),
                "-n",
                namespace,
                str(message),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise GovernanceError("artifact signing failed")
        return Path(f"{message}.sig").read_text(encoding="utf-8")


def _verify_bytes(
    data: bytes,
    *,
    signature: str,
    public_key: str,
    principal: str,
    namespace: str,
) -> None:
    with tempfile.TemporaryDirectory(prefix="phase3b-verify-") as directory_name:
        directory = Path(directory_name)
        allowed = directory / "allowed_signers"
        signature_path = directory / "payload.sig"
        allowed.write_text(f"{principal} {public_key}\n", encoding="utf-8")
        signature_path.write_text(signature, encoding="utf-8")
        result = subprocess.run(
            [
                "ssh-keygen",
                "-Y",
                "verify",
                "-f",
                str(allowed),
                "-I",
                principal,
                "-n",
                namespace,
                "-s",
                str(signature_path),
            ],
            input=data,
            check=False,
            capture_output=True,
        )
    if result.returncode != 0:
        raise GovernanceError("artifact signature verification failed")


def _verify_private_key(path: Path, expected_fingerprint: str) -> None:
    result = subprocess.run(
        ["ssh-keygen", "-y", "-f", str(path)],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise GovernanceError("cannot read artifact signing private key")
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8") as public:
        public.write(result.stdout)
        public.flush()
        if ssh_public_key_fingerprint(Path(public.name)) != expected_fingerprint:
            raise GovernanceError("private signing key does not match public identity")


def _identifier(value: str, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise GovernanceError(f"{label} is unresolved")
    return value.strip()

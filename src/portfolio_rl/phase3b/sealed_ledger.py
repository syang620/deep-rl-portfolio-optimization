"""Encrypted, append-only Phase 3B performance ledger."""

from __future__ import annotations

import base64
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from nacl.public import PrivateKey, PublicKey, SealedBox

from portfolio_rl.phase3b.governance import (
    GovernanceError,
    canonical_json_bytes,
    logical_json_sha256,
    read_json,
    sha256_file,
)
from portfolio_rl.phase3b.operational_metrics import (
    SEALING_ALGORITHM,
    OperationsConfig,
    decode_sealing_public_key,
)
from portfolio_rl.phase3b.signatures import (
    create_signature_record,
    verify_signature_record,
)

GENESIS_LEDGER_HASH = "0" * 64
LEDGER_SIGNATURE_NAMESPACE = "portfolio-rl-phase3b-ledger-v1"
CUSTODIAN_CHECKPOINT_NAMESPACE = "portfolio-rl-phase3b-custodian-checkpoint-v1"


def append_sealed_entry(
    *,
    ledger_root: Path,
    entry_id: str,
    context_type: str,
    context_id: str,
    close_date: str,
    performance_payload: dict[str, Any],
    bindings: dict[str, str],
    operations_config: OperationsConfig,
    service_private_key_path: Path,
    service_public_key_path: Path,
    service_principal: str,
    signed_at: datetime,
) -> Path:
    """Encrypt and append one immutable entry, or verify an identical retry."""
    root = ledger_root.resolve()
    entries = root / "entries"
    entries.mkdir(parents=True, exist_ok=True)
    destination = entries / entry_id
    if destination.exists():
        _verify_retry(destination, performance_payload, bindings)
        return destination
    lock = root / ".append.lock"
    descriptor = _acquire_lock(lock)
    try:
        if destination.exists():
            _verify_retry(destination, performance_payload, bindings)
            return destination
        previous_hash, sequence = _ledger_tip(entries)
        plaintext_hash = logical_json_sha256(performance_payload)
        ciphertext = SealedBox(
            PublicKey(decode_sealing_public_key(operations_config.sealing_public_key_path))
        ).encrypt(canonical_json_bytes(performance_payload))
        destination.mkdir()
        ciphertext_path = destination / "entry.sealed"
        ciphertext_path.write_bytes(ciphertext)
        content = {
            "schema_version": 1,
            "entry_id": entry_id,
            "sequence": sequence,
            "context_type": context_type,
            "context_id": context_id,
            "close_date": close_date,
            "previous_entry_hash": previous_hash,
            "ciphertext_sha256": sha256_file(ciphertext_path),
            "plaintext_sha256": plaintext_hash,
            "sealing_algorithm": SEALING_ALGORITHM,
            "sealing_key_fingerprint": operations_config.sealing_key_fingerprint,
            "bindings": dict(sorted(bindings.items())),
        }
        content["entry_hash"] = logical_json_sha256(content)
        _write_json(destination / "entry_manifest.json", content)
        signature = create_signature_record(
            payload=content,
            payload_path="entry_manifest.json",
            artifact_type="sealed_ledger_entry",
            role="service_signing",
            principal=service_principal,
            namespace=LEDGER_SIGNATURE_NAMESPACE,
            private_key_path=service_private_key_path,
            public_key_path=service_public_key_path,
            signed_at=signed_at,
        )
        _write_json(destination / "entry_signature.json", signature)
    except Exception:
        if destination.exists():
            for child in destination.iterdir():
                child.unlink()
            destination.rmdir()
        raise
    finally:
        os.close(descriptor)
        lock.unlink(missing_ok=True)
    return destination


def write_custodian_checkpoint(
    *,
    ledger_root: Path,
    checkpoint_date: str,
    custodian_private_key_path: Path,
    custodian_public_key_path: Path,
    custodian_principal: str,
    signed_at: datetime,
) -> Path:
    """Sign the current entry count and chain tip to expose tail deletion."""
    root = ledger_root.resolve()
    previous_hash, next_sequence = _ledger_tip(root / "entries")
    payload = {
        "schema_version": 1,
        "checkpoint_date": checkpoint_date,
        "entry_count": next_sequence - 1,
        "ledger_tip_hash": previous_hash,
    }
    checkpoint = root / "checkpoints" / f"{checkpoint_date}.json"
    if checkpoint.exists():
        if read_json(checkpoint)["checkpoint"] != payload:
            raise GovernanceError("custodian checkpoint overwrite is forbidden")
        return checkpoint
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    signature = create_signature_record(
        payload=payload,
        payload_path=checkpoint.name,
        artifact_type="sealed_ledger_checkpoint",
        role="data_operations_custodian",
        principal=custodian_principal,
        namespace=CUSTODIAN_CHECKPOINT_NAMESPACE,
        private_key_path=custodian_private_key_path,
        public_key_path=custodian_public_key_path,
        signed_at=signed_at,
    )
    _write_json(checkpoint, {"checkpoint": payload, "signature_record": signature})
    return checkpoint


def verify_sealed_ledger(
    *,
    ledger_root: Path,
    service_public_key_path: Path,
    custodian_public_key_path: Path | None = None,
) -> dict[str, Any]:
    """Verify contiguous entries, hashes, signatures, and optional checkpoints."""
    root = ledger_root.resolve()
    entries = _entry_directories(root / "entries")
    previous = GENESIS_LEDGER_HASH
    for sequence, directory in enumerate(entries, start=1):
        manifest = read_json(directory / "entry_manifest.json")
        if manifest.get("sequence") != sequence:
            raise GovernanceError("sealed-ledger sequence is not contiguous")
        if manifest.get("entry_id") != directory.name:
            raise GovernanceError("sealed-ledger entry ID mismatch")
        if manifest.get("previous_entry_hash") != previous:
            raise GovernanceError("sealed-ledger chain is broken")
        if sha256_file(directory / "entry.sealed") != manifest.get(
            "ciphertext_sha256"
        ):
            raise GovernanceError("sealed-ledger ciphertext hash mismatch")
        unhashed = dict(manifest)
        recorded_hash = unhashed.pop("entry_hash", None)
        if logical_json_sha256(unhashed) != recorded_hash:
            raise GovernanceError("sealed-ledger entry hash mismatch")
        verify_signature_record(
            payload=manifest,
            record=read_json(directory / "entry_signature.json"),
            public_key_path=service_public_key_path,
            expected_role="service_signing",
            expected_namespace=LEDGER_SIGNATURE_NAMESPACE,
        )
        previous = str(recorded_hash)
    checkpoints = sorted((root / "checkpoints").glob("*.json")) if (root / "checkpoints").exists() else []
    if checkpoints:
        if custodian_public_key_path is None:
            raise GovernanceError("custodian key is required to verify checkpoints")
        latest = read_json(checkpoints[-1])
        checkpoint = latest.get("checkpoint")
        if not isinstance(checkpoint, dict):
            raise GovernanceError("custodian checkpoint is malformed")
        if checkpoint.get("entry_count") != len(entries) or checkpoint.get(
            "ledger_tip_hash"
        ) != previous:
            raise GovernanceError("sealed-ledger tip differs from custodian checkpoint")
        verify_signature_record(
            payload=checkpoint,
            record=latest.get("signature_record", {}),
            public_key_path=custodian_public_key_path,
            expected_role="data_operations_custodian",
            expected_namespace=CUSTODIAN_CHECKPOINT_NAMESPACE,
        )
    return {"entry_count": len(entries), "ledger_tip_hash": previous, "verified": True}


def decrypt_entry_for_verification(
    *, ciphertext_path: Path, recipient_private_key_path: Path
) -> dict[str, Any]:
    """Decrypt an entry for authorized tests or post-holdout verification only."""
    raw = base64.b64decode(
        recipient_private_key_path.read_text(encoding="utf-8").strip(), validate=True
    )
    if len(raw) != 32:
        raise GovernanceError("performance-sealing private key must contain 32 bytes")
    plaintext = SealedBox(PrivateKey(raw)).decrypt(ciphertext_path.read_bytes())
    import json

    payload = json.loads(plaintext)
    if not isinstance(payload, dict):
        raise GovernanceError("sealed performance payload is not an object")
    return payload


def _verify_retry(
    destination: Path, payload: dict[str, Any], bindings: dict[str, str]
) -> None:
    manifest = read_json(destination / "entry_manifest.json")
    if manifest.get("plaintext_sha256") != logical_json_sha256(payload):
        raise GovernanceError("same-entry retry contains different performance data")
    if manifest.get("bindings") != dict(sorted(bindings.items())):
        raise GovernanceError("same-entry retry contains different bindings")


def _ledger_tip(entries_root: Path) -> tuple[str, int]:
    directories = _entry_directories(entries_root)
    if not directories:
        return GENESIS_LEDGER_HASH, 1
    manifest = read_json(directories[-1] / "entry_manifest.json")
    if manifest.get("sequence") != len(directories):
        raise GovernanceError("sealed-ledger sequence is not contiguous")
    value = manifest.get("entry_hash")
    if not isinstance(value, str) or len(value) != 64:
        raise GovernanceError("sealed-ledger tip hash is invalid")
    return value, len(directories) + 1


def _entry_directories(root: Path) -> list[Path]:
    return (
        sorted(path for path in root.iterdir() if path.is_dir())
        if root.exists()
        else []
    )


def _acquire_lock(path: Path) -> int:
    try:
        return os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise GovernanceError("sealed-ledger append is already in progress") from exc


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_bytes(canonical_json_bytes(payload))

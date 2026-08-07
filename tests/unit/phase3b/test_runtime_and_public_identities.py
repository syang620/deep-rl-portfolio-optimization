from __future__ import annotations

import base64
import json
import subprocess
from pathlib import Path

import pytest
from nacl.public import PrivateKey

import portfolio_rl.phase3b.runtime_identity as runtime_module
from portfolio_rl.phase3b.governance import GovernanceError, validate_container_identity
from portfolio_rl.phase3b.public_identities import inspect_public_identities
from portfolio_rl.phase3b.runtime_identity import verify_runtime_identity


def test_runtime_sha_matches_container_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sha = "a" * 40
    monkeypatch.setattr(runtime_module, "_git_text", lambda *args: sha)
    embedded = tmp_path / "embedded.json"
    embedded.write_text(json.dumps({"runtime_git_sha": sha}), encoding="utf-8")
    container = tmp_path / "container.json"
    container.write_text(json.dumps(_container(sha)), encoding="utf-8")

    assert verify_runtime_identity(
        repository_root=tmp_path,
        embedded_identity_path=embedded,
        container_identity_path=container,
    ) == sha


def test_runtime_sha_mismatch_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runtime_module, "_git_text", lambda *args: "a" * 40)
    embedded = tmp_path / "embedded.json"
    embedded.write_text(json.dumps({"runtime_git_sha": "b" * 40}), encoding="utf-8")
    container = tmp_path / "container.json"
    container.write_text(json.dumps(_container("a" * 40)), encoding="utf-8")

    with pytest.raises(GovernanceError, match="runtime SHAs differ"):
        verify_runtime_identity(
            repository_root=tmp_path,
            embedded_identity_path=embedded,
            container_identity_path=container,
        )


def test_mutable_tag_not_used_as_authoritative_identity() -> None:
    payload = _container("a" * 40)
    payload["image_reference"] = "registry.example/phase3b:latest"
    with pytest.raises(GovernanceError, match="mutable image tag"):
        validate_container_identity(payload, "a" * 40)


def test_rejects_private_key_input(tmp_path: Path) -> None:
    keys = _keys(tmp_path)
    with pytest.raises(GovernanceError, match="private key input"):
        inspect_public_identities(
            service_signing_key=keys["service_private"],
            performance_sealing_key=keys["sealing_public"],
            approver_keys=keys["approvers"],
        )


def test_approver_keys_must_be_distinct(tmp_path: Path) -> None:
    keys = _keys(tmp_path)
    approvers = dict(keys["approvers"])
    approvers["independent_reviewer"] = approvers["portfolio_manager"]
    with pytest.raises(GovernanceError, match="must be distinct"):
        inspect_public_identities(
            service_signing_key=keys["service_public"],
            performance_sealing_key=keys["sealing_public"],
            approver_keys=approvers,
        )


def test_service_key_cannot_be_approver(tmp_path: Path) -> None:
    keys = _keys(tmp_path)
    approvers = dict(keys["approvers"])
    approvers["portfolio_manager"] = keys["service_public"]
    with pytest.raises(GovernanceError, match="must be distinct"):
        inspect_public_identities(
            service_signing_key=keys["service_public"],
            performance_sealing_key=keys["sealing_public"],
            approver_keys=approvers,
        )


def test_sealing_key_is_separate_identity(tmp_path: Path) -> None:
    keys = _keys(tmp_path)
    result = inspect_public_identities(
        service_signing_key=keys["service_public"],
        performance_sealing_key=keys["sealing_public"],
        approver_keys=keys["approvers"],
    )
    assert result["service_key_is_not_unseal_authority"] is True
    assert result["sealing_identity_is_distinct"] is True
    assert result["public_keys_only"] is True


def _container(sha: str) -> dict[str, object]:
    digest = "sha256:" + "c" * 64
    return {
        "schema_version": 1,
        "image_reference": "registry.example/phase3b@" + digest,
        "image_digest": digest,
        "git_commit": sha,
        "input_schema_version": "phase3b_normalized_features_v1",
        "data_source_contract_version": "point-in-time-v1",
        "built_at": "2030-01-01T00:00:00+00:00",
    }


def _keys(root: Path) -> dict[str, object]:
    public: dict[str, Path] = {}
    private: dict[str, Path] = {}
    for role in (
        "service",
        "portfolio_manager",
        "independent_reviewer",
        "data_operations_custodian",
    ):
        path = root / role
        subprocess.run(
            ["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(path)],
            check=True,
        )
        private[role] = path
        public[role] = Path(f"{path}.pub")
    sealing = PrivateKey.generate().public_key
    sealing_path = root / "sealing.pub"
    sealing_path.write_text(base64.b64encode(bytes(sealing)).decode(), encoding="utf-8")
    return {
        "service_private": private["service"],
        "service_public": public["service"],
        "sealing_public": sealing_path,
        "approvers": {role: public[role] for role in (
            "portfolio_manager", "independent_reviewer", "data_operations_custodian"
        )},
    }

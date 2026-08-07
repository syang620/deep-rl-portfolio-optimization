from __future__ import annotations

import base64
import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from nacl.public import PrivateKey

import portfolio_rl.phase3b.identity_approval as identity_module
from portfolio_rl.phase3b.execution import load_execution_config
from portfolio_rl.phase3b.governance import GovernanceError, load_access_policy
from portfolio_rl.phase3b.identity_approval import (
    APPROVER_ROLES,
    finalize_identity_approval,
    prepare_identity_approval,
    sign_identity_approval,
    verify_identity_approval,
)
from portfolio_rl.phase3b.operational_metrics import load_operations_config
from scripts.run_phase3b_certification import main as certification_main


class _Reconciliation:
    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "scaler_sha256": "0" * 64,
            "feature_spec_sha256": "1" * 64,
            "compared_model_rows": 3522,
            "refit_performed": False,
            "reconciled": True,
        }


def test_identity_package_requires_three_signatures_and_detects_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path)
    evidence = _evidence()
    monkeypatch.setattr(
        identity_module, "_verify_runtime_git", lambda *args, **kwargs: "a" * 40
    )
    monkeypatch.setattr(
        identity_module,
        "verify_candidate",
        lambda *args, **kwargs: SimpleNamespace(
            frozen_candidate_path=fixture["frozen_candidate"],
            freeze_manifest_path=fixture["freeze_manifest"],
        ),
    )
    monkeypatch.setattr(
        identity_module,
        "reconcile_frozen_scaler",
        lambda **kwargs: _Reconciliation(),
    )
    monkeypatch.setattr(identity_module, "_governance_evidence", lambda root: evidence)
    package = prepare_identity_approval(
        repository_root=tmp_path,
        config_path=fixture["input_config"],
        created_at=datetime(2030, 1, 1, tzinfo=UTC),
    )
    challenge = identity_module.read_json(package / "identity_challenge.json")
    assert challenge["provenance"] == {
        "pr22_merge_sha": identity_module.EXPECTED_PR22_MERGE_SHA,
        "identity_tooling_merge_sha": "a" * 40,
        "container_digest": "sha256:" + "c" * 64,
    }
    assert challenge["governance_evidence"].keys() == evidence.keys()
    approved_execution = yaml.safe_load(
        (package / "candidate/execution.approved.yaml").read_text(encoding="utf-8")
    )
    assert approved_execution["status"] == "approved"
    assert (
        approved_execution["snapshot_contract"]["scaler"]["status"]
        == "approved_for_phase3b"
    )
    execution = load_execution_config(
        package / "candidate/execution.approved.yaml",
        repository_root=tmp_path,
        require_approved=True,
    )
    assert execution.scaler_status == "approved_for_phase3b"
    operations = load_operations_config(
        package / "candidate/operations.approved.yaml",
        repository_root=tmp_path,
        require_approved=True,
    )
    assert operations.sealing_approval_status == "approved_for_phase3b"
    access = load_access_policy(
        tmp_path, package / "candidate/access_control.approved.yaml"
    )
    assert {row["role"] for row in access["approvers"]} == set(APPROVER_ROLES)
    with pytest.raises(GovernanceError, match="cannot read JSON"):
        finalize_identity_approval(
            repository_root=tmp_path,
            package_path=package,
            finalized_at=datetime(2030, 1, 2, tzinfo=UTC),
        )
    for role in APPROVER_ROLES:
        sign_identity_approval(
            package_path=package,
            role=role,
            private_key_path=fixture[f"{role}_private"],
            signed_at=datetime(2030, 1, 2, tzinfo=UTC),
        )
    changed_evidence = _evidence()
    changed_evidence["certification_registry_state"]["official_cycle_count"] = 1
    monkeypatch.setattr(
        identity_module, "_governance_evidence", lambda root: changed_evidence
    )
    with pytest.raises(GovernanceError, match="evidence changed after challenge"):
        finalize_identity_approval(
            repository_root=tmp_path,
            package_path=package,
            finalized_at=datetime(2030, 1, 2, tzinfo=UTC),
        )
    monkeypatch.setattr(identity_module, "_governance_evidence", lambda root: evidence)
    finalize_identity_approval(
        repository_root=tmp_path,
        package_path=package,
        finalized_at=datetime(2030, 1, 2, tzinfo=UTC),
    )
    verified = verify_identity_approval(
        repository_root=tmp_path, package_path=package
    )
    assert verified.identity_tooling_merge_sha == "a" * 40
    assert verified.identity.runtime_git_commit == "a" * 40
    with pytest.raises(FileExistsError, match="already finalized"):
        finalize_identity_approval(
            repository_root=tmp_path,
            package_path=package,
            finalized_at=datetime(2030, 1, 3, tzinfo=UTC),
        )
    execution_path = package / "candidate/execution.approved.yaml"
    execution_path.write_text(
        execution_path.read_text(encoding="utf-8") + "\n# mutation\n",
        encoding="utf-8",
    )
    with pytest.raises(GovernanceError, match="prepared file hash mismatch"):
        verify_identity_approval(repository_root=tmp_path, package_path=package)


def test_identity_package_rejects_service_key_as_approver(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path)
    config = yaml.safe_load(fixture["input_config"].read_text(encoding="utf-8"))
    config["approvers"][0]["public_key_path"] = config["service_signing"][
        "public_key_path"
    ]
    fixture["input_config"].write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    monkeypatch.setattr(
        identity_module, "_verify_runtime_git", lambda *args, **kwargs: "a" * 40
    )
    monkeypatch.setattr(
        identity_module,
        "verify_candidate",
        lambda *args, **kwargs: SimpleNamespace(
            frozen_candidate_path=fixture["frozen_candidate"],
            freeze_manifest_path=fixture["freeze_manifest"],
        ),
    )
    with pytest.raises(GovernanceError, match="must be distinct"):
        prepare_identity_approval(
            repository_root=tmp_path,
            config_path=fixture["input_config"],
            created_at=datetime(2030, 1, 1, tzinfo=UTC),
        )


def test_registry_evidence_rejects_started_certification_and_holdout(
    tmp_path: Path,
) -> None:
    certification = tmp_path / "artifacts/phase3b/certification/cert/cycle.json"
    certification.parent.mkdir(parents=True)
    certification.write_text('{"official":true}', encoding="utf-8")
    with pytest.raises(GovernanceError, match="already started"):
        identity_module._registry_state(
            tmp_path,
            tmp_path / "artifacts/phase3b/certification",
            "certification",
        )
    registration = (
        tmp_path
        / "artifacts/phase3b/registration/holdout/holdout_registration.json"
    )
    registration.parent.mkdir(parents=True)
    registration.write_text("{}", encoding="utf-8")
    with pytest.raises(GovernanceError, match="already registered"):
        identity_module._registry_state(
            tmp_path, tmp_path / "artifacts/phase3b/registration", "holdout"
        )


def test_runtime_git_identity_requires_origin_main_and_pr22_ancestry(
    tmp_path: Path,
) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "fixture@example.com"],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Fixture"], cwd=tmp_path, check=True
    )
    tracked = tmp_path / "runtime.txt"
    tracked.write_text("pr22", encoding="utf-8")
    subprocess.run(["git", "add", "runtime.txt"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "pr22"], cwd=tmp_path, check=True)
    subprocess.run(["git", "branch", "-M", "main"], cwd=tmp_path, check=True)
    pr22_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    tracked.write_text("tooling", encoding="utf-8")
    subprocess.run(["git", "add", "runtime.txt"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "tooling"], cwd=tmp_path, check=True
    )
    subprocess.run(
        ["git", "update-ref", "refs/remotes/origin/main", "HEAD"],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(
        ["git", "remote", "add", "origin", str(tmp_path)],
        cwd=tmp_path,
        check=True,
    )
    tooling_sha = identity_module._verify_runtime_git(
        tmp_path, pr22_merge_sha=pr22_sha
    )
    assert tooling_sha != pr22_sha
    tracked.write_text("dirty", encoding="utf-8")
    with pytest.raises(GovernanceError, match="tracked working tree"):
        identity_module._verify_runtime_git(tmp_path, pr22_merge_sha=pr22_sha)


def test_official_certification_requires_finalized_identity_package(
    tmp_path: Path,
) -> None:
    evidence = tmp_path / "cycle_evidence.json"
    evidence.write_text('{"identity_sha256":"' + "a" * 64 + '"}', encoding="utf-8")
    with pytest.raises(GovernanceError, match="approved configs"):
        certification_main(
            [
                "--root",
                str(tmp_path),
                "--certification-id",
                "cert-1",
                "--cycle-number",
                "1",
                "--decision-date",
                "2030-01-01",
                "--execution-date",
                "2030-01-02",
                "--evidence",
                str(evidence),
                "--output",
                str(tmp_path / "cycle.json"),
                "--official",
                "--authorization",
                "authorization.json",
                "--execution-config",
                "execution.yaml",
                "--operations-config",
                "operations.yaml",
            ]
        )


def test_fresh_test_access_audit_rejects_new_test_artifact(tmp_path: Path) -> None:
    pretest = (
        tmp_path
        / "artifacts/pretest_freeze/ppo_v1_ensemble5_alpha025_pretest_v1"
    )
    pretest.mkdir(parents=True)
    (pretest / "test_access_audit.json").write_text(
        '{"known_legacy_access":[]}', encoding="utf-8"
    )
    unexpected = tmp_path / "artifacts/new_test_result.json"
    unexpected.write_text('{"split":"test"}', encoding="utf-8")
    with pytest.raises(GovernanceError, match="legacy allowlist"):
        identity_module._test_access_audit(tmp_path)


def _fixture(root: Path) -> dict[str, Path]:
    (root / "configs/phase3b").mkdir(parents=True)
    source_root = Path(__file__).resolve().parents[3]
    for name in ("execution.yaml", "operations.yaml", "access_control.yaml"):
        shutil.copy2(
            source_root / f"configs/phase3b/{name}",
            root / f"configs/phase3b/{name}",
        )
    shutil.copy2(source_root / "configs/universe.yaml", root / "configs/universe.yaml")
    scaler = root / "artifacts/scalers/feature_scaler_v1.pkl"
    scaler.parent.mkdir(parents=True)
    shutil.copy2(source_root / "artifacts/scalers/feature_scaler_v1.pkl", scaler)
    key_root = root / "keys"
    key_root.mkdir()
    result: dict[str, Path] = {}
    for role in ("service", *APPROVER_ROLES):
        private = key_root / role
        subprocess.run(
            ["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(private)],
            check=True,
        )
        result[f"{role}_private"] = private
        result[f"{role}_public"] = Path(f"{private}.pub")
    sealing = PrivateKey.generate()
    sealing_public = key_root / "sealing.pub"
    sealing_public.write_text(
        base64.b64encode(bytes(sealing.public_key)).decode("ascii"), encoding="utf-8"
    )
    container = root / "container.json"
    container.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "image_reference": "registry.example/portfolio-rl@sha256:" + "c" * 64,
                "image_digest": "sha256:" + "c" * 64,
                "git_commit": "a" * 40,
                "input_schema_version": "phase3b-input-v1",
                "data_source_contract_version": "point-in-time-v1",
                "built_at": "2030-01-01T00:00:00+00:00",
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    config = {
        "schema_version": 1,
        "status": "ready_for_preparation",
        "approval_id": "identity-fixture",
        "pr22_merge_sha": identity_module.EXPECTED_PR22_MERGE_SHA,
        "container_identity_path": "container.json",
        "service_signing": {
            "principal": "phase3b-service",
            "public_key_path": "keys/service.pub",
        },
        "performance_sealing": {"public_key_path": "keys/sealing.pub"},
        "approvers": [
            {
                "role": role,
                "name": f"Fixture {role}",
                "principal": f"fixture-{role}",
                "public_key_path": f"keys/{role}.pub",
            }
            for role in APPROVER_ROLES
        ],
        "draft_configs": {
            "execution": "configs/phase3b/execution.yaml",
            "operations": "configs/phase3b/operations.yaml",
            "access_control": "configs/phase3b/access_control.yaml",
        },
        "pretest_package": "artifacts/pretest_freeze/fixture",
        "output_root": "artifacts/phase3b/identity_approval",
    }
    input_config = root / "identity.yaml"
    input_config.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    result["input_config"] = input_config
    pretest = root / "artifacts/pretest_freeze/fixture"
    pretest.mkdir(parents=True)
    result["frozen_candidate"] = pretest / "frozen_candidate.json"
    result["freeze_manifest"] = pretest / "freeze_manifest.json"
    result["frozen_candidate"].write_text("{}", encoding="utf-8")
    result["freeze_manifest"].write_text("{}", encoding="utf-8")
    return result


def _evidence() -> dict[str, dict[str, object]]:
    return {
        "test_access_audit": {
            "schema_version": 1,
            "unexpected_test_access": [],
            "test_accessed": False,
        },
        "certification_registry_state": {
            "schema_version": 1,
            "official_cycle_count": 0,
        },
        "holdout_registry_state": {
            "schema_version": 1,
            "canonical_registration_count": 0,
        },
    }

from __future__ import annotations

import base64
import json
import shutil
import subprocess
from datetime import UTC, date, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from nacl.public import PrivateKey

import portfolio_rl.phase3b.certification_readiness as readiness_module
import portfolio_rl.phase3b.identity_approval as identity_module
from portfolio_rl.phase3b.certification_authorization import (
    finalize_certification_authorization,
    prepare_certification_authorization,
    sign_certification_authorization,
    verify_finalized_certification_authorization,
)
from portfolio_rl.phase3b.certification_readiness import check_certification_readiness
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
            "maximum_asset_normalization_error": 0.0,
            "maximum_global_normalization_error": 0.0,
            "maximum_model_matrix_error": 0.0,
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
    with pytest.raises(FileExistsError, match="already exists"):
        prepare_identity_approval(
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
    assert set(challenge["frozen_contracts"]) == {
        "scaler_sha256",
        "feature_spec_sha256",
        "scaler_reconciliation_sha256",
        "cost_map_logical_sha256",
        "execution_config_sha256",
        "operations_config_sha256",
        "access_control_config_sha256",
        "universe_config_path",
        "universe_config_sha256",
    }
    assert all(
        record["sha256"] and record["logical_sha256"]
        for record in challenge["governance_evidence"].values()
    )
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
    with pytest.raises(GovernanceError, match="signature inventory"):
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
    with pytest.raises(GovernanceError, match="finalized identity"):
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


def test_certification_authorization_is_separate_and_requires_three_signatures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture, identity_package = _finalized_identity(tmp_path, monkeypatch)
    authorization = prepare_certification_authorization(
        repository_root=tmp_path,
        identity_package_path=identity_package,
        certification_id="cert-fixture",
        approved_start_date=date(2030, 2, 1),
        approved_end_date=date(2030, 2, 28),
        output_root=Path("artifacts/phase3b/certification_authorization"),
        created_at=datetime(2030, 1, 3, tzinfo=UTC),
    )
    assert authorization != identity_package
    assert not (authorization / "finalized_authorization.json").exists()
    with pytest.raises(GovernanceError, match="signature inventory"):
        finalize_certification_authorization(
            repository_root=tmp_path,
            package_path=authorization,
            finalized_at=datetime(2030, 1, 4, tzinfo=UTC),
        )
    with pytest.raises(GovernanceError, match="role is invalid"):
        sign_certification_authorization(
            repository_root=tmp_path,
            package_path=authorization,
            role="service",
            private_key_path=fixture["service_private"],
            signed_at=datetime(2030, 1, 4, tzinfo=UTC),
        )
    with pytest.raises(GovernanceError, match="does not match public identity"):
        sign_certification_authorization(
            repository_root=tmp_path,
            package_path=authorization,
            role="portfolio_manager",
            private_key_path=fixture["independent_reviewer_private"],
            signed_at=datetime(2030, 1, 4, tzinfo=UTC),
        )
    for role in APPROVER_ROLES:
        sign_certification_authorization(
            repository_root=tmp_path,
            package_path=authorization,
            role=role,
            private_key_path=fixture[f"{role}_private"],
            signed_at=datetime(2030, 1, 4, tzinfo=UTC),
        )
    with pytest.raises(FileExistsError, match="signature exists"):
        sign_certification_authorization(
            repository_root=tmp_path,
            package_path=authorization,
            role="portfolio_manager",
            private_key_path=fixture["portfolio_manager_private"],
            signed_at=datetime(2030, 1, 4, tzinfo=UTC),
        )
    finalize_certification_authorization(
        repository_root=tmp_path,
        package_path=authorization,
        finalized_at=datetime(2030, 1, 5, tzinfo=UTC),
    )
    verified = verify_finalized_certification_authorization(
        repository_root=tmp_path, package_path=authorization
    )
    assert verified.certification_id == "cert-fixture"
    assert verified.identity_package_path == identity_package
    assert not (tmp_path / "artifacts/phase3b/certification").exists()


def test_authorization_payload_or_identity_mutation_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture, identity_package = _finalized_identity(tmp_path, monkeypatch)
    authorization = _finalized_authorization(
        tmp_path, fixture, identity_package
    )
    challenge = authorization / "challenge.json"
    challenge.write_text(challenge.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(GovernanceError, match="checksum mismatch"):
        verify_finalized_certification_authorization(
            repository_root=tmp_path, package_path=authorization
        )


def test_container_config_and_key_mutation_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, identity_package = _finalized_identity(tmp_path, monkeypatch)
    for relative in (
        "candidate/container_identity.json",
        "candidate/execution.approved.yaml",
        "candidate/public_keys/portfolio_manager.pub",
    ):
        path = identity_package / relative
        original = path.read_bytes()
        path.write_bytes(original + b"\n")
        with pytest.raises(GovernanceError, match="prepared file hash mismatch"):
            verify_identity_approval(
                repository_root=tmp_path, package_path=identity_package
            )
        path.write_bytes(original)


def test_identity_change_invalidates_certification_authorization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture, identity_package = _finalized_identity(tmp_path, monkeypatch)
    authorization = _finalized_authorization(tmp_path, fixture, identity_package)
    finalized = identity_package / "finalized_identity.json"
    finalized.write_text(
        finalized.read_text(encoding="utf-8") + "\n", encoding="utf-8"
    )
    with pytest.raises(GovernanceError, match="final artifact mutation"):
        verify_finalized_certification_authorization(
            repository_root=tmp_path, package_path=authorization
        )


def test_readiness_blocks_holdout_and_unexpected_test_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture, identity_package = _finalized_identity(tmp_path, monkeypatch)
    authorization = _finalized_authorization(tmp_path, fixture, identity_package)
    monkeypatch.setattr(readiness_module, "verify_runtime_identity", lambda **kwargs: "a" * 40)
    monkeypatch.setattr(readiness_module, "_test_access_audit", lambda root: {})
    container = identity_package / "candidate/container_identity.json"
    embedded = tmp_path / "embedded.json"
    embedded.write_text('{"runtime_git_sha":"' + "a" * 40 + '"}', encoding="utf-8")
    registration = tmp_path / "artifacts/phase3b/registration/x/holdout_registration.json"
    registration.parent.mkdir(parents=True)
    registration.write_text("{}", encoding="utf-8")
    missing_authorization = check_certification_readiness(
        repository_root=tmp_path,
        identity_package_path=identity_package,
        authorization_package_path=tmp_path / "missing-authorization",
        container_identity_path=container,
        embedded_identity_path=embedded,
        certification_id="cert-fixture",
    )
    assert any("certification authorization" in reason for reason in missing_authorization)
    blockers = check_certification_readiness(
        repository_root=tmp_path,
        identity_package_path=identity_package,
        authorization_package_path=authorization,
        container_identity_path=container,
        embedded_identity_path=embedded,
        certification_id="cert-fixture",
        decision_date=date(2030, 2, 1),
        cycle_number=1,
    )
    assert any("holdout registry" in reason for reason in blockers)
    registration.unlink()
    monkeypatch.setattr(
        readiness_module,
        "verify_runtime_identity",
        lambda **kwargs: (_ for _ in ()).throw(GovernanceError("runtime mismatch")),
    )
    blockers = check_certification_readiness(
        repository_root=tmp_path,
        identity_package_path=identity_package,
        authorization_package_path=authorization,
        container_identity_path=container,
        embedded_identity_path=embedded,
        certification_id="cert-fixture",
        decision_date=date(2030, 2, 1),
        cycle_number=1,
    )
    assert any("runtime identity" in reason for reason in blockers)
    monkeypatch.setattr(readiness_module, "verify_runtime_identity", lambda **kwargs: "a" * 40)
    monkeypatch.setattr(
        readiness_module,
        "_test_access_audit",
        lambda root: (_ for _ in ()).throw(GovernanceError("unexpected access")),
    )
    blockers = check_certification_readiness(
        repository_root=tmp_path,
        identity_package_path=identity_package,
        authorization_package_path=authorization,
        container_identity_path=container,
        embedded_identity_path=embedded,
        certification_id="cert-fixture",
        decision_date=date(2030, 2, 1),
        cycle_number=1,
    )
    assert any("test-access audit" in reason for reason in blockers)


def _finalized_identity(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[dict[str, Path], Path]:
    fixture = _fixture(root)
    evidence = _evidence()
    monkeypatch.setattr(identity_module, "_verify_runtime_git", lambda *args, **kwargs: "a" * 40)
    monkeypatch.setattr(
        identity_module,
        "verify_candidate",
        lambda *args, **kwargs: SimpleNamespace(
            frozen_candidate_path=fixture["frozen_candidate"],
            freeze_manifest_path=fixture["freeze_manifest"],
        ),
    )
    monkeypatch.setattr(identity_module, "reconcile_frozen_scaler", lambda **kwargs: _Reconciliation())
    monkeypatch.setattr(identity_module, "_governance_evidence", lambda root: evidence)
    package = prepare_identity_approval(
        repository_root=root,
        config_path=fixture["input_config"],
        created_at=datetime(2030, 1, 1, tzinfo=UTC),
    )
    for role in APPROVER_ROLES:
        sign_identity_approval(
            package_path=package,
            role=role,
            private_key_path=fixture[f"{role}_private"],
            signed_at=datetime(2030, 1, 2, tzinfo=UTC),
        )
    finalize_identity_approval(
        repository_root=root,
        package_path=package,
        finalized_at=datetime(2030, 1, 2, tzinfo=UTC),
    )
    return fixture, package


def _finalized_authorization(
    root: Path, fixture: dict[str, Path], identity_package: Path
) -> Path:
    package = prepare_certification_authorization(
        repository_root=root,
        identity_package_path=identity_package,
        certification_id="cert-fixture",
        approved_start_date=date(2030, 2, 1),
        approved_end_date=date(2030, 2, 28),
        output_root=Path("artifacts/phase3b/certification_authorization"),
        created_at=datetime(2030, 1, 3, tzinfo=UTC),
    )
    for role in APPROVER_ROLES:
        sign_certification_authorization(
            repository_root=root,
            package_path=package,
            role=role,
            private_key_path=fixture[f"{role}_private"],
            signed_at=datetime(2030, 1, 4, tzinfo=UTC),
        )
    finalize_certification_authorization(
        repository_root=root,
        package_path=package,
        finalized_at=datetime(2030, 1, 5, tzinfo=UTC),
    )
    return package


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

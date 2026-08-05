from __future__ import annotations

import json
import shutil
import subprocess
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest
import yaml

from portfolio_rl.phase3b import holdout_registry
from portfolio_rl.phase3b.governance import (
    EXPECTED_CANDIDATE_MANIFEST_SHA256,
    EXPECTED_MODEL_VERSION,
    SIGNATURE_NAMESPACE,
    VerifiedCandidate,
    logical_json_sha256,
    sha256_file,
    ssh_public_key_fingerprint,
)
from portfolio_rl.phase3b.holdout_registry import (
    RegistrationError,
    prepare_registration_challenge,
    register_forward_holdout,
    verify_holdout_registration,
)
from portfolio_rl.training.pretest_freeze import (
    EXPECTED_ASSET_ORDER,
    EXPECTED_SEEDS,
    FrozenCandidate,
)


def test_signed_registration_builds_and_verifies_immutable_package(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _registration_fixture(tmp_path, monkeypatch)
    challenge = prepare_registration_challenge(
        repository_root=tmp_path,
        config_path=fixture["config"],
        challenge_output_path=fixture["challenge"],
        now=fixture["prepare_time"],
    )
    approvals = _sign_challenge(challenge, fixture["private_keys"])

    output = register_forward_holdout(
        repository_root=tmp_path,
        challenge_path=challenge,
        approvals_dir=approvals,
        now=fixture["register_time"],
    )
    registered = verify_holdout_registration(output, repository_root=tmp_path)

    assert registered.holdout_id == "independent-forward-fixture"
    assert registered.candidate_model_version == EXPECTED_MODEL_VERSION
    assert registered.candidate_manifest_sha256 == EXPECTED_CANDIDATE_MANIFEST_SHA256
    registration = _read_json(output / "holdout_registration.json")
    assert registration["primary_execution_timing"] == "one_close_delay"
    assert registration["primary_transaction_cost_bps"] == 10.0
    assert registration["asset_tier_cost_role"] == "stress_advisory"
    assert registration["legacy_test_designation"] == "2025+"
    assert registration["legacy_test_independent"] is False
    manifest = _read_json(output / "registration_manifest.json")
    assert manifest["test_accessed"] is False
    assert manifest["performance_data_accessed"] is False
    assert {record["path"] for record in manifest["files"]} == {
        path.name
        for path in output.iterdir()
        if path.name != "registration_manifest.json"
    }


def test_registration_requires_all_three_signatures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _registration_fixture(tmp_path, monkeypatch)
    challenge = prepare_registration_challenge(
        repository_root=tmp_path,
        config_path=fixture["config"],
        challenge_output_path=fixture["challenge"],
        now=fixture["prepare_time"],
    )
    approvals = _sign_challenge(challenge, fixture["private_keys"])
    (approvals / "portfolio_manager.sig").unlink()

    with pytest.raises(RegistrationError, match="incomplete or unexpected"):
        register_forward_holdout(
            repository_root=tmp_path,
            challenge_path=challenge,
            approvals_dir=approvals,
            now=fixture["register_time"],
        )


def test_registration_rejects_wrong_role_signature(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _registration_fixture(tmp_path, monkeypatch)
    challenge = prepare_registration_challenge(
        repository_root=tmp_path,
        config_path=fixture["config"],
        challenge_output_path=fixture["challenge"],
        now=fixture["prepare_time"],
    )
    approvals = _sign_challenge(challenge, fixture["private_keys"])
    shutil.copy2(
        approvals / "data_ops_custodian.sig",
        approvals / "portfolio_manager.sig",
    )

    with pytest.raises(RegistrationError, match="invalid SSH approval signature"):
        register_forward_holdout(
            repository_root=tmp_path,
            challenge_path=challenge,
            approvals_dir=approvals,
            now=fixture["register_time"],
        )


def test_registration_refuses_duplicate_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _registration_fixture(tmp_path, monkeypatch)
    challenge = prepare_registration_challenge(
        repository_root=tmp_path,
        config_path=fixture["config"],
        challenge_output_path=fixture["challenge"],
        now=fixture["prepare_time"],
    )
    approvals = _sign_challenge(challenge, fixture["private_keys"])
    register_forward_holdout(
        repository_root=tmp_path,
        challenge_path=challenge,
        approvals_dir=approvals,
        now=fixture["register_time"],
    )

    with pytest.raises(RegistrationError, match="duplicate"):
        register_forward_holdout(
            repository_root=tmp_path,
            challenge_path=challenge,
            approvals_dir=approvals,
            now=fixture["register_time"],
        )


def test_verifier_detects_post_registration_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _registration_fixture(tmp_path, monkeypatch)
    challenge = prepare_registration_challenge(
        repository_root=tmp_path,
        config_path=fixture["config"],
        challenge_output_path=fixture["challenge"],
        now=fixture["prepare_time"],
    )
    approvals = _sign_challenge(challenge, fixture["private_keys"])
    output = register_forward_holdout(
        repository_root=tmp_path,
        challenge_path=challenge,
        approvals_dir=approvals,
        now=fixture["register_time"],
    )
    access_path = output / "access_policy.json"
    access = _read_json(access_path)
    access["signature_namespace"] = "mutated"
    _write_json(access_path, access)

    with pytest.raises(RegistrationError, match="artifact hash mismatch"):
        verify_holdout_registration(output, repository_root=tmp_path)


def test_signed_challenge_mutation_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _registration_fixture(tmp_path, monkeypatch)
    challenge_path = prepare_registration_challenge(
        repository_root=tmp_path,
        config_path=fixture["config"],
        challenge_output_path=fixture["challenge"],
        now=fixture["prepare_time"],
    )
    approvals = _sign_challenge(challenge_path, fixture["private_keys"])
    challenge = _read_json(challenge_path)
    challenge["candidate"]["partial_rebalance_alpha"] = 0.5
    _write_json(challenge_path, challenge)

    with pytest.raises(RegistrationError, match="challenge payload hash mismatch"):
        register_forward_holdout(
            repository_root=tmp_path,
            challenge_path=challenge_path,
            approvals_dir=approvals,
            now=fixture["register_time"],
        )


def test_draft_registration_cannot_prepare(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _registration_fixture(tmp_path, monkeypatch)
    config = yaml.safe_load(fixture["config"].read_text(encoding="utf-8"))
    config["status"] = "draft"
    fixture["config"].write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(RegistrationError, match="still draft"):
        prepare_registration_challenge(
            repository_root=tmp_path,
            config_path=fixture["config"],
            challenge_output_path=fixture["challenge"],
            now=fixture["prepare_time"],
        )


def _registration_fixture(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> dict[str, object]:
    _run(["git", "init", "-q"], root)
    _run(["git", "config", "user.email", "phase3b@example.com"], root)
    _run(["git", "config", "user.name", "Phase3B Fixture"], root)
    (root / ".gitignore").write_text("artifacts/\n", encoding="utf-8")
    config_dir = root / "configs/phase3b"
    config_dir.mkdir(parents=True)
    keys_dir = root / "keys"
    keys_dir.mkdir()
    artifacts = root / "artifacts/inputs"
    artifacts.mkdir(parents=True)
    private_dir = root / "artifacts/private_keys"
    private_dir.mkdir(parents=True)

    roles = [role.value for role in holdout_registry.ApprovalRole]
    private_keys: dict[str, Path] = {}
    approvers = []
    for role in roles:
        private_path = private_dir / role
        _run(
            [
                "ssh-keygen",
                "-q",
                "-t",
                "ed25519",
                "-N",
                "",
                "-f",
                str(private_path),
            ],
            root,
        )
        public_path = keys_dir / f"{role}.pub"
        shutil.copy2(Path(f"{private_path}.pub"), public_path)
        private_keys[role] = private_path
        approvers.append(
            {
                "role": role,
                "name": role.replace("_", " ").title(),
                "principal": f"{role}@fixture",
                "public_key_path": public_path.relative_to(root).as_posix(),
                "public_key_sha256": sha256_file(public_path),
                "public_key_fingerprint": ssh_public_key_fingerprint(public_path),
            }
        )

    access_path = config_dir / "access_control.yaml"
    _write_yaml(
        access_path,
        {
            "schema_version": 1,
            "status": "approved",
            "signature_namespace": SIGNATURE_NAMESPACE,
            "approvers": approvers,
        },
    )
    acceptance_path = root / "configs/final_candidate_acceptance.yaml"
    _write_yaml(acceptance_path, {"schema_version": 1, "frozen": "pr19"})
    execution_path = config_dir / "execution.yaml"
    _write_yaml(
        execution_path,
        {
            "schema_version": 1,
            "execution_timing": "one_close_delay",
            "primary_transaction_cost_bps": 10.0,
            "asset_tier_cost_role": "stress_advisory",
        },
    )
    operations_path = config_dir / "operations.yaml"
    _write_yaml(
        operations_path,
        {"schema_version": 1, "minimum_valid_rebalance_decisions": 50},
    )

    schedule = _schedule_payload()
    start = schedule["holdout_decision_dates"][0]
    end = schedule["holdout_decision_dates"][-1]
    config_path = config_dir / "holdout_registration.yaml"
    _write_yaml(
        config_path,
        {
            "schema_version": 1,
            "status": "approved",
            "candidate": {
                "frozen_candidate_path": (
                    "artifacts/pretest_freeze/"
                    f"{EXPECTED_MODEL_VERSION}/frozen_candidate.json"
                ),
                "candidate_manifest_sha256": EXPECTED_CANDIDATE_MANIFEST_SHA256,
            },
            "holdout": {
                "holdout_id": "independent-forward-fixture",
                "start_decision_date": start,
                "end_decision_date": end,
                "minimum_valid_rebalance_decisions": 50,
                "expected_rebalance_frequency_trading_days": 5,
                "performance_unseal_not_before": schedule[
                    "final_holding_period_complete_at_utc"
                ],
                "input_schema_version": "phase3b-input-v1",
                "data_source_contract_version": "point-in-time-v1",
                "existing_test_designation": "2025+",
            },
            "inputs": {
                "certification_manifest": "artifacts/inputs/certification.json",
                "trading_session_schedule": "artifacts/inputs/schedule.json",
                "container_identity": "artifacts/inputs/container.json",
            },
            "frozen_configs": {
                "candidate_acceptance": "configs/final_candidate_acceptance.yaml",
                "holdout_registration": "configs/phase3b/holdout_registration.yaml",
                "access_control": "configs/phase3b/access_control.yaml",
                "execution": "configs/phase3b/execution.yaml",
                "operations": "configs/phase3b/operations.yaml",
            },
            "access_control_path": "configs/phase3b/access_control.yaml",
            "output_root": "artifacts/phase3b/registration",
        },
    )
    _run(["git", "add", "."], root)
    _run(["git", "commit", "-q", "-m", "fixture"], root)
    commit = _run(["git", "rev-parse", "HEAD"], root).strip()

    _write_json(artifacts / "schedule.json", schedule)
    container = {
        "schema_version": 1,
        "image_reference": "registry.example/portfolio-rl@sha256:" + "c" * 64,
        "image_digest": "sha256:" + "c" * 64,
        "git_commit": commit,
        "input_schema_version": "phase3b-input-v1",
        "data_source_contract_version": "point-in-time-v1",
        "built_at": "2029-12-28T12:00:00+00:00",
    }
    _write_json(artifacts / "container.json", container)
    config_hashes = {
        "candidate_acceptance": sha256_file(acceptance_path),
        "execution": sha256_file(execution_path),
        "operations": sha256_file(operations_path),
    }
    certification = {
        "schema_version": 1,
        "certification_id": "certification-fixture",
        "status": "passed",
        "cycle_count": 4,
        "completed_decision_dates": schedule["certification_decision_dates"],
        "candidate_manifest_sha256": EXPECTED_CANDIDATE_MANIFEST_SHA256,
        "container_image_digest": container["image_digest"],
        "git_commit": commit,
        "schedule_sha256": sha256_file(artifacts / "schedule.json"),
        "frozen_config_hashes": config_hashes,
        "performance_metrics_computed": False,
        "certification_completed_at": "2029-12-28T18:00:00+00:00",
    }
    certification["manifest_payload_sha256"] = logical_json_sha256(certification)
    _write_json(artifacts / "certification.json", certification)

    candidate_dir = root / "artifacts/pretest_freeze" / EXPECTED_MODEL_VERSION
    candidate_dir.mkdir(parents=True)
    candidate_path = candidate_dir / "frozen_candidate.json"
    _write_json(candidate_path, {"model_version": EXPECTED_MODEL_VERSION})
    (candidate_dir / "freeze_manifest.json").write_text("fixture", encoding="utf-8")
    shutil.copy2(acceptance_path, candidate_dir / "acceptance_criteria.yaml")
    fake_candidate = FrozenCandidate(
        model_version=EXPECTED_MODEL_VERSION,
        member_seeds=EXPECTED_SEEDS,
        member_model_paths=tuple(Path(f"models/{seed}.zip") for seed in EXPECTED_SEEDS),
        member_model_hashes=tuple(str(seed).zfill(64) for seed in EXPECTED_SEEDS),
        action_temperatures=(1.0, 1.0, 1.0, 1.0, 1.0),
        partial_rebalance_alpha=0.25,
        initial_portfolio="equal_weight",
        asset_order=EXPECTED_ASSET_ORDER,
        feature_version="feature_v1",
        feature_spec_hash="f" * 64,
        environment_config_hash="e" * 64,
        transaction_cost_bps=10.0,
        rebalance_frequency_trading_days=5,
    )
    verified = VerifiedCandidate(
        candidate=fake_candidate,
        frozen_candidate_path=candidate_path,
        freeze_manifest_path=candidate_dir / "freeze_manifest.json",
        freeze_manifest_sha256=EXPECTED_CANDIDATE_MANIFEST_SHA256,
    )
    monkeypatch.setattr(holdout_registry, "verify_candidate", lambda *args: verified)
    return {
        "config": config_path,
        "challenge": root / "artifacts/registration_challenge.json",
        "private_keys": private_keys,
        "prepare_time": datetime(2029, 12, 29, 12, tzinfo=UTC),
        "register_time": datetime(2029, 12, 29, 18, tzinfo=UTC),
    }


def _schedule_payload() -> dict[str, object]:
    sessions = []
    cursor = date(2029, 12, 3)
    while len(sessions) < 330:
        if cursor.weekday() < 5:
            sessions.append(cursor)
        cursor += timedelta(days=1)
    certification = sessions[:20:5]
    start_position = 20
    start = sessions[start_position]
    anniversary = start.replace(year=start.year + 1)
    positions = []
    position = start_position
    while sessions[position] <= anniversary:
        positions.append(position)
        position += 5
    final_end = sessions[positions[-1] + 5]
    return {
        "schema_version": 1,
        "schedule_id": "forward-schedule-fixture",
        "timezone": "America/New_York",
        "trading_sessions": [item.isoformat() for item in sessions],
        "certification_decision_dates": [item.isoformat() for item in certification],
        "holdout_decision_dates": [sessions[index].isoformat() for index in positions],
        "final_holding_period_end_date": final_end.isoformat(),
        "final_holding_period_complete_at_utc": datetime(
            final_end.year, final_end.month, final_end.day, 22, tzinfo=UTC
        ).isoformat(),
    }


def _sign_challenge(challenge: Path, private_keys: dict[str, Path]) -> Path:
    approvals = challenge.parent / "approvals"
    approvals.mkdir()
    generated = Path(f"{challenge}.sig")
    for role, private_key in private_keys.items():
        _run(
            [
                "ssh-keygen",
                "-Y",
                "sign",
                "-f",
                str(private_key),
                "-n",
                SIGNATURE_NAMESPACE,
                str(challenge),
            ],
            challenge.parent,
        )
        shutil.move(generated, approvals / f"{role}.sig")
    return approvals


def _run(command: list[str], cwd: Path) -> str:
    result = subprocess.run(
        command, cwd=cwd, check=False, capture_output=True, text=True
    )
    if result.returncode != 0:
        raise AssertionError(result.stderr)
    return result.stdout


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_yaml(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

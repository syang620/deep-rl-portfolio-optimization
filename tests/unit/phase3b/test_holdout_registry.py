from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from portfolio_rl.phase3b.holdout_registry import (
    RegistrationError,
    _load_registration_config,
)


def test_registration_config_requires_complete_frozen_bundle(tmp_path: Path) -> None:
    path = tmp_path / "registration.yaml"
    payload = _config()
    del payload["frozen_configs"]["operations"]
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(RegistrationError, match="frozen config bundle"):
        _load_registration_config(path)


def test_registration_config_rejects_unknown_fields(tmp_path: Path) -> None:
    path = tmp_path / "registration.yaml"
    payload = _config()
    payload["allow_overwrite"] = True
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(RegistrationError, match="keys mismatch"):
        _load_registration_config(path)


def _config() -> dict[str, object]:
    return {
        "schema_version": 1,
        "status": "draft",
        "candidate": {
            "frozen_candidate_path": "candidate/frozen_candidate.json",
            "candidate_manifest_sha256": "a" * 64,
        },
        "holdout": {
            "holdout_id": None,
            "start_decision_date": None,
            "end_decision_date": None,
            "minimum_valid_rebalance_decisions": 50,
            "expected_rebalance_frequency_trading_days": 5,
            "performance_unseal_not_before": None,
            "input_schema_version": None,
            "data_source_contract_version": None,
            "existing_test_designation": "2025+",
        },
        "inputs": {
            "certification_manifest": None,
            "trading_session_schedule": None,
            "container_identity": None,
        },
        "frozen_configs": {
            "candidate_acceptance": "acceptance.yaml",
            "holdout_registration": "registration.yaml",
            "access_control": "access.yaml",
            "execution": "execution.yaml",
            "operations": "operations.yaml",
        },
        "access_control_path": "access.yaml",
        "output_root": "artifacts/registration",
    }

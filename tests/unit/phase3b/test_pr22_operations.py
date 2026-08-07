from __future__ import annotations

import base64
import json
import subprocess
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pytest
from nacl.public import PrivateKey

from portfolio_rl.phase3b.certification import (
    REQUIRED_CYCLE_CHECKS,
    reconstruct_certification_status,
)
from portfolio_rl.phase3b.close_processor import (
    StrategyCloseState,
    process_market_close,
)
from portfolio_rl.phase3b.governance import GovernanceError, sha256_file
from portfolio_rl.phase3b.incidents import reject_and_log_unseal_attempt
from portfolio_rl.phase3b.operational_metrics import (
    OperationsConfig,
    assert_operationally_safe,
    sealing_key_fingerprint,
)
from portfolio_rl.phase3b.operational_state import (
    load_restricted_state,
    write_restricted_state,
)
from portfolio_rl.phase3b.sealed_ledger import (
    append_sealed_entry,
    decrypt_entry_for_verification,
    verify_sealed_ledger,
    write_custodian_checkpoint,
)
from portfolio_rl.phase3b.snapshot_chain import STRATEGIES

ASSETS = (
    "SPY",
    "QQQ",
    "IWM",
    "EFA",
    "EEM",
    "TLT",
    "IEF",
    "SHY",
    "LQD",
    "HYG",
    "GLD",
    "DBC",
    "VNQ",
    "XLU",
)


def test_one_close_execution_uses_drifted_weights_and_signed_target() -> None:
    equal = tuple(np.full(len(ASSETS), 1.0 / len(ASSETS)))
    states = {
        strategy: StrategyCloseState(weights=equal, nav=1.0, peak_nav=1.0)
        for strategy in STRATEGIES
    }
    target = np.zeros(len(ASSETS))
    target[0] = 1.0
    targets = {strategy: tuple(target) for strategy in STRATEGIES}
    costs = {ticker: 10.0 for ticker in ASSETS}
    returns = (np.log(1.10), *([0.0] * (len(ASSETS) - 1)))

    result = process_market_close(
        previous_close_date=date(2030, 1, 2),
        close_date=date(2030, 1, 3),
        asset_order=ASSETS,
        asset_log_returns=returns,
        states=states,
        signed_targets=targets,
        recommendation_hash="a" * 64,
        expected_execution_date=date(2030, 1, 3),
        flat_transaction_cost_bps=10.0,
        asset_cost_bps=costs,
    )
    replay = process_market_close(
        previous_close_date=date(2030, 1, 2),
        close_date=date(2030, 1, 3),
        asset_order=ASSETS,
        asset_log_returns=returns,
        states=states,
        signed_targets=targets,
        recommendation_hash="a" * 64,
        expected_execution_date=date(2030, 1, 3),
        flat_transaction_cost_bps=10.0,
        asset_cost_bps=costs,
    )
    assert replay == result

    candidate = next(
        row
        for row in result.sealed_performance["strategies"]
        if row["strategy"] == "candidate"
    )
    drifted = np.asarray(candidate["pre_trade_weights"])
    assert drifted[0] > equal[0]
    assert candidate["turnover"] == pytest.approx(
        0.5 * np.abs(target - drifted).sum()
    )
    assert candidate["execution_weights"] == list(targets["candidate"])
    buy_hold = result.next_states["buy_and_hold_equal_weight"].weights
    assert np.allclose(buy_hold, drifted)


def test_execution_on_wrong_close_fails_no_trade() -> None:
    equal = tuple(np.full(len(ASSETS), 1.0 / len(ASSETS)))
    states = {
        strategy: StrategyCloseState(weights=equal, nav=1.0, peak_nav=1.0)
        for strategy in STRATEGIES
    }
    with pytest.raises(GovernanceError, match="cannot execute"):
        process_market_close(
            previous_close_date=date(2030, 1, 2),
            close_date=date(2030, 1, 3),
            asset_order=ASSETS,
            asset_log_returns=tuple(np.zeros(len(ASSETS))),
            states=states,
            signed_targets={strategy: equal for strategy in STRATEGIES},
            recommendation_hash="a" * 64,
            expected_execution_date=date(2030, 1, 4),
            flat_transaction_cost_bps=10.0,
            asset_cost_bps={ticker: 10.0 for ticker in ASSETS},
        )


def test_sealed_ledger_is_replay_safe_hash_linked_and_checkpointed(
    tmp_path: Path,
) -> None:
    keys = _keys(tmp_path)
    config = _operations_config(tmp_path, keys)
    now = datetime(2030, 1, 3, 22, tzinfo=UTC)
    ledger = tmp_path / "ledger"
    kwargs = {
        "ledger_root": ledger,
        "entry_id": "2030-01-03",
        "context_type": "certification",
        "context_id": "cert-1",
        "close_date": "2030-01-03",
        "performance_payload": {"candidate_nav": 1.01, "daily_return": 0.01},
        "bindings": {"candidate_hash": "a" * 64},
        "operations_config": config,
        "service_private_key_path": keys["service_private"],
        "service_public_key_path": keys["service_public"],
        "service_principal": "phase3b-service",
        "signed_at": now,
    }
    first = append_sealed_entry(**kwargs)
    assert append_sealed_entry(**kwargs) == first
    decrypted = decrypt_entry_for_verification(
        ciphertext_path=first / "entry.sealed",
        recipient_private_key_path=keys["seal_private"],
    )
    assert decrypted["candidate_nav"] == 1.01
    write_custodian_checkpoint(
        ledger_root=ledger,
        checkpoint_date="2030-01-03",
        custodian_private_key_path=keys["custodian_private"],
        custodian_public_key_path=keys["custodian_public"],
        custodian_principal="data-custodian",
        signed_at=now,
    )
    assert verify_sealed_ledger(
        ledger_root=ledger,
        service_public_key_path=keys["service_public"],
        custodian_public_key_path=keys["custodian_public"],
    )["entry_count"] == 1
    ciphertext = first / "entry.sealed"
    original_ciphertext = ciphertext.read_bytes()
    ciphertext.write_bytes(original_ciphertext + b"mutation")
    with pytest.raises(GovernanceError, match="ciphertext hash mismatch"):
        verify_sealed_ledger(
            ledger_root=ledger,
            service_public_key_path=keys["service_public"],
            custodian_public_key_path=keys["custodian_public"],
        )
    ciphertext.write_bytes(original_ciphertext)
    for child in first.iterdir():
        child.unlink()
    first.rmdir()
    with pytest.raises(GovernanceError, match="differs from custodian checkpoint"):
        verify_sealed_ledger(
            ledger_root=ledger,
            service_public_key_path=keys["service_public"],
            custodian_public_key_path=keys["custodian_public"],
        )


def test_restricted_state_round_trip_is_exact(tmp_path: Path) -> None:
    keys = _keys(tmp_path)
    equal = tuple(np.full(len(ASSETS), 1.0 / len(ASSETS)))
    states = {
        strategy: StrategyCloseState(weights=equal, nav=1.01, peak_nav=1.02)
        for strategy in STRATEGIES
    }
    path = tmp_path / "state.json"
    write_restricted_state(
        path=path,
        context_type="certification",
        context_id="cert-1",
        as_of_date="2030-01-03",
        asset_order=ASSETS,
        states=states,
        previous_result_sha256="a" * 64,
        ledger_tip_hash="b" * 64,
        service_private_key_path=keys["service_private"],
        service_public_key_path=keys["service_public"],
        service_principal="phase3b-service",
        signed_at=datetime(2030, 1, 3, 22, tzinfo=UTC),
    )
    assets, restored, payload = load_restricted_state(
        path=path, service_public_key_path=keys["service_public"]
    )
    assert assets == ASSETS
    assert restored == states
    assert payload["visibility"] == "internal_restricted_not_dashboard_safe"


def test_certification_requires_four_consecutive_cycles_and_resets(
    tmp_path: Path,
) -> None:
    paths = []
    for index in range(4):
        path = tmp_path / f"cycle-{index}.json"
        path.write_text(
            json.dumps(
                {
                    "official": True,
                    "scheduled_decision_missed": False,
                    "identity_sha256": "a" * 64,
                    "checks": dict.fromkeys(REQUIRED_CYCLE_CHECKS, True),
                }
            ),
            encoding="utf-8",
        )
        paths.append(path)
    status = reconstruct_certification_status(
        cycle_manifest_paths=paths,
        expected_identity_sha256="a" * 64,
        official=True,
    )
    assert status.valid
    payload = json.loads(paths[2].read_text(encoding="utf-8"))
    payload["identity_sha256"] = "b" * 64
    paths[2].write_text(json.dumps(payload), encoding="utf-8")
    reset = reconstruct_certification_status(
        cycle_manifest_paths=paths,
        expected_identity_sha256="a" * 64,
        official=True,
    )
    assert not reset.valid
    assert reset.consecutive_completed_cycles == 1


def test_operational_output_rejects_performance_and_unseal_is_logged(
    tmp_path: Path,
) -> None:
    with pytest.raises(GovernanceError, match="sealed field leaked"):
        assert_operationally_safe(
            {"candidate_nav": 1.2}, ("return", "nav", "pnl")
        )
    audit = tmp_path / "unseal.json"
    with pytest.raises(GovernanceError, match="not authorized"):
        reject_and_log_unseal_attempt(
            audit_path=audit,
            requester="decision-service",
            reason="interim review",
            timestamp=datetime(2030, 1, 3, tzinfo=UTC),
        )
    assert json.loads(audit.read_text(encoding="utf-8"))["severity"] == "fatal"


def _keys(root: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for role in ("service", "custodian"):
        private = root / f"{role}_key"
        subprocess.run(
            ["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(private)],
            check=True,
        )
        result[f"{role}_private"] = private
        result[f"{role}_public"] = Path(f"{private}.pub")
    seal = PrivateKey.generate()
    seal_private = root / "seal.private"
    seal_public = root / "seal.public"
    seal_private.write_text(
        base64.b64encode(bytes(seal)).decode("ascii"), encoding="utf-8"
    )
    seal_public.write_text(
        base64.b64encode(bytes(seal.public_key)).decode("ascii"), encoding="utf-8"
    )
    result["seal_private"] = seal_private
    result["seal_public"] = seal_public
    return result


def _operations_config(root: Path, keys: dict[str, Path]) -> OperationsConfig:
    raw_public = base64.b64decode(keys["seal_public"].read_text(encoding="utf-8"))
    placeholder = root / "operations.yaml"
    placeholder.write_text("fixture: true\n", encoding="utf-8")
    universe = root / "universe.yaml"
    universe.write_text("fixture: true\n", encoding="utf-8")
    return OperationsConfig(
        config_path=placeholder,
        config_sha256=sha256_file(placeholder),
        status="approved",
        required_consecutive_cycles=4,
        rebalance_frequency_trading_days=5,
        maximum_missed_scheduled_decisions=2,
        close_snapshot_schema_version="phase3b_market_close_v1",
        price_field="adjusted_close_total_return_proxy",
        sealing_public_key_path=keys["seal_public"],
        sealing_public_key_sha256=sha256_file(keys["seal_public"]),
        sealing_key_fingerprint=sealing_key_fingerprint(raw_public),
        sealing_approval_status="approved_for_phase3b",
        universe_config_path=universe,
        universe_config_sha256=sha256_file(universe),
        asset_classes={ticker: "fixture" for ticker in ASSETS},
        allowed_weight_types=("execution_weight",),
        forbidden_field_tokens=("return", "nav", "pnl"),
        development_output_root=root / "development",
        certification_output_root=root / "certification",
        holdout_output_root=root / "holdout",
    )

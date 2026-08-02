from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from portfolio_rl.training import pretest_freeze


def test_acceptance_config_freezes_exact_candidate() -> None:
    config = yaml.safe_load(
        Path("configs/final_candidate_acceptance.yaml").read_text(encoding="utf-8")
    )

    pretest_freeze._validate_acceptance_config(
        config,
        "ppo_v1_ensemble5_alpha025_pretest_v1",
    )
    assert config["candidate"]["seeds"] == [7, 42, 101, 202, 999]
    assert config["candidate"]["partial_rebalance_alpha"] == 0.25
    assert config["candidate"]["initial_portfolio"] == "equal_weight"
    assert config["candidate"]["turnover_definition"] == "half_l1_one_way"
    assert config["governance"]["phase3b_status"] == "blocked"
    assert config["governance"]["existing_test_independent"] is False
    assert config["governance"]["replacement_holdout_required"] is True


def test_acceptance_config_distinguishes_hard_and_advisory_gates() -> None:
    config = yaml.safe_load(
        Path("configs/final_candidate_acceptance.yaml").read_text(encoding="utf-8")
    )

    assert config["hard_gates"]["positive_active_return_vs_primary"] is True
    assert "one_close_delay_result" in config["advisory_diagnostics"]
    assert "inverse_volatility" in config["secondary_comparisons"]


def test_frozen_candidate_rejects_wrong_seed_order() -> None:
    payload = _candidate_payload()
    payload["member_seed_order"] = [42, 7, 101, 202, 999]

    with pytest.raises(pretest_freeze.PretestFreezeError, match="seed order"):
        pretest_freeze._candidate_from_payload(payload)


def test_frozen_alpha_is_exactly_0_25() -> None:
    payload = _candidate_payload()
    payload["partial_rebalance_alpha"] = 0.5

    with pytest.raises(pretest_freeze.PretestFreezeError, match="alpha"):
        pretest_freeze._candidate_from_payload(payload)


def test_frozen_initial_portfolio_is_equal_weight() -> None:
    payload = _candidate_payload()
    payload["initial_portfolio"] = "inverse_volatility"

    with pytest.raises(pretest_freeze.PretestFreezeError, match="equal weight"):
        pretest_freeze._candidate_from_payload(payload)


def test_frozen_turnover_definition_matches_half_l1() -> None:
    payload = _candidate_payload()
    payload["turnover_definition"] = "full_l1"

    with pytest.raises(pretest_freeze.PretestFreezeError, match="turnover"):
        pretest_freeze._candidate_from_payload(payload)


def test_frozen_baseline_definitions_are_complete() -> None:
    definitions = pretest_freeze._baseline_definitions()

    assert set(definitions["definitions"]) == {
        "equal_weight_weekly",
        "buy_and_hold_equal_weight",
        "inverse_volatility",
        "momentum_63d_top3_equal_weight",
        "spy_only",
        "shy_only",
    }
    momentum = definitions["definitions"]["momentum_63d_top3_equal_weight"]
    assert momentum["lookback_trading_days"] == 63
    assert momentum["tie_break"] == "canonical_asset_order_stable_sort"


def test_pm_packet_labels_2024_and_states_test_not_used() -> None:
    candidate = pretest_freeze._candidate_from_payload(
        _candidate_payload()
    )
    criteria = pretest_freeze.AcceptanceCriteria(
        primary_hurdle="equal_weight_weekly",
        hard_gates={"positive_active_return": True},
        advisory_diagnostics=("one_close_delay_result",),
        secondary_comparisons=("inverse_volatility",),
        approval_status="pending_pm_ml_signoff",
    )

    packet = pretest_freeze.build_pm_review_packet(
        evidence_paths={}, candidate=candidate, acceptance_criteria=criteria
    )

    assert "2024 is the consumed development/selection period" in packet
    assert "No final-test data were used" in packet
    assert "legacy Phase 2 model" in packet


def test_generated_command_is_blocked_and_not_executable_in_pr19() -> None:
    commands = pretest_freeze._commands("candidate_v1")

    assert "BLOCKED AND NOT EXECUTED" in commands
    assert "<approved-independent-holdout-manifest>" in commands
    assert "intentionally outside PR 19" in commands


def _candidate_payload() -> dict[str, object]:
    return {
        "model_version": "candidate_v1",
        "member_seed_order": [7, 42, 101, 202, 999],
        "member_model_paths": [f"model_{seed}.zip" for seed in [7, 42, 101, 202, 999]],
        "member_model_hashes": [str(seed) * 64 for seed in [7, 4, 1, 2, 9]],
        "action_temperatures": [0.5] * 5,
        "partial_rebalance_alpha": 0.25,
        "initial_portfolio": "equal_weight",
        "asset_order": list(pretest_freeze.EXPECTED_ASSET_ORDER),
        "feature_version": "v1",
        "feature_spec_hash": "f" * 64,
        "environment_config_hash": "e" * 64,
        "transaction_cost_bps": 10.0,
        "rebalance_frequency_trading_days": 5,
        "turnover_definition": (
            "0.5 * sum(abs(executed_target - live_drifted_current_weights))"
        ),
    }

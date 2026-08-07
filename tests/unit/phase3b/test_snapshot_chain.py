from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from portfolio_rl.phase3b.execution import load_execution_config
from portfolio_rl.phase3b.governance import GovernanceError
from portfolio_rl.phase3b.snapshot_chain import (
    dataframe_logical_sha256,
    recommendation_chain_hash,
)


def test_logical_dataframe_hash_detects_value_and_order_changes() -> None:
    frame = pd.DataFrame({"date": [pd.Timestamp("2030-01-02")], "value": [1.0]})

    original = dataframe_logical_sha256(frame)

    assert dataframe_logical_sha256(frame.copy()) == original
    assert dataframe_logical_sha256(frame.assign(value=2.0)) != original
    assert dataframe_logical_sha256(frame[["value", "date"]]) != original


def test_recommendation_chain_binds_every_input_hash() -> None:
    original = recommendation_chain_hash(
        previous_chain_hash="0" * 64,
        snapshot_sha256="1" * 64,
        state_sha256="2" * 64,
        recommendation_content_sha256="3" * 64,
    )

    changed = recommendation_chain_hash(
        previous_chain_hash="0" * 64,
        snapshot_sha256="1" * 64,
        state_sha256="2" * 64,
        recommendation_content_sha256="4" * 64,
    )

    assert original != changed


def test_repository_execution_config_remains_blocked_pending_approval() -> None:
    root = Path(__file__).resolve().parents[3]

    with pytest.raises(GovernanceError, match="still draft"):
        load_execution_config(
            root / "configs/phase3b/execution.yaml",
            repository_root=root,
        )

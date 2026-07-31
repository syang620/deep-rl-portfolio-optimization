from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio_rl.evaluation.dynamic_value import (
    RecordedTargetPolicy,
    _build_control_sequences,
)


def test_control_sequences_use_declared_executed_target_mapping() -> None:
    live = np.array(
        [[0.6, 0.4], [0.7, 0.3], [0.8, 0.2], [0.9, 0.1]],
        dtype=np.float64,
    )
    prior = np.array([[0.4, 0.6], [0.6, 0.4]], dtype=np.float64)

    controls = _build_control_sequences(
        live_targets=live,
        prior_targets=prior,
        n_assets=2,
        circular_shifts=[1],
    )

    np.testing.assert_allclose(
        controls["ex_ante_static_2023"]["targets"],
        np.tile([0.5, 0.5], (4, 1)),
    )
    np.testing.assert_allclose(
        controls["oracle_static_2024"]["targets"],
        np.tile([0.75, 0.25], (4, 1)),
    )
    np.testing.assert_allclose(
        controls["lag_1_decision"]["targets"],
        [[0.5, 0.5], [0.6, 0.4], [0.7, 0.3], [0.8, 0.2]],
    )
    np.testing.assert_allclose(
        controls["circular_shift_1"]["targets"],
        [[0.9, 0.1], [0.6, 0.4], [0.7, 0.3], [0.8, 0.2]],
    )
    np.testing.assert_array_equal(
        controls["circular_shift_1"]["source_steps"],
        [3, 0, 1, 2],
    )


def test_recorded_target_policy_rejects_date_misalignment() -> None:
    policy = RecordedTargetPolicy(
        decision_dates=[pd.Timestamp("2024-01-02")],
        targets=np.array([[0.5, 0.5]]),
    )

    with pytest.raises(ValueError, match="date mismatch"):
        policy.target_weights(
            np.ones(2),
            {"date": pd.Timestamp("2024-01-03")},
        )


@pytest.mark.parametrize(
    "targets",
    [
        np.array([[np.nan, 0.5]]),
        np.array([[-0.1, 1.1]]),
        np.array([[0.4, 0.4]]),
    ],
)
def test_recorded_target_policy_rejects_invalid_weights(
    targets: np.ndarray,
) -> None:
    with pytest.raises(ValueError, match="target weights"):
        RecordedTargetPolicy(
            decision_dates=[pd.Timestamp("2024-01-02")],
            targets=targets,
        )


def test_control_sequences_reject_shift_not_smaller_than_path() -> None:
    targets = np.tile([0.5, 0.5], (4, 1))

    with pytest.raises(ValueError, match="smaller"):
        _build_control_sequences(
            live_targets=targets,
            prior_targets=targets,
            n_assets=2,
            circular_shifts=[4],
        )

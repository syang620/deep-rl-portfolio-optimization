from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pytest

from portfolio_rl.policies.ensemble_policy import MeanWeightEnsemblePolicy


class FixedPolicy:
    def __init__(self, weights: list[float]) -> None:
        self.weights = np.asarray(weights, dtype=np.float64)
        self.calls: list[tuple[np.ndarray, Mapping[str, Any]]] = []
        self.reset_count = 0

    def reset(self) -> None:
        self.reset_count += 1

    def target_weights(
        self,
        observation: np.ndarray,
        info: Mapping[str, Any],
    ) -> np.ndarray:
        self.calls.append((observation, info))
        return self.weights.copy()


def test_ensemble_averages_member_target_weights_in_frozen_order() -> None:
    first = FixedPolicy([0.8, 0.2])
    second = FixedPolicy([0.2, 0.8])
    policy = MeanWeightEnsemblePolicy(
        member_policies={"seed_7": first, "seed_42": second}
    )
    observation = np.array([1.0, 2.0])
    info = {
        "asset_order": ["SPY", "SHY"],
        "current_weights": np.array([0.5, 0.5]),
        "date": "2024-01-02",
    }

    target = policy.target_weights(observation, info)

    np.testing.assert_allclose(target, [0.5, 0.5])
    assert policy.member_order == ["seed_7", "seed_42"]
    assert first.calls[0][0] is observation
    assert second.calls[0][0] is observation
    assert first.calls[0][1] is info
    assert second.calls[0][1] is info
    assert [record.member for record in policy.member_target_records] == [
        "seed_7",
        "seed_42",
    ]


def test_ensemble_replay_is_deterministic_and_reset_clears_records() -> None:
    first = FixedPolicy([0.7, 0.3])
    second = FixedPolicy([0.3, 0.7])
    policy = MeanWeightEnsemblePolicy(
        member_policies={"seed_7": first, "seed_42": second}
    )
    info = {
        "asset_order": ["SPY", "SHY"],
        "current_weights": np.array([0.5, 0.5]),
        "date": "2024-01-02",
    }

    first_target = policy.target_weights(np.ones(2), info)
    policy.reset()
    second_target = policy.target_weights(np.ones(2), info)

    np.testing.assert_array_equal(first_target, second_target)
    assert len(policy.member_target_records) == 2
    assert first.reset_count == 1
    assert second.reset_count == 1


@pytest.mark.parametrize(
    "weights,match",
    [
        ([1.0], "shape"),
        ([np.nan, np.nan], "finite"),
        ([1.1, -0.1], "nonnegative"),
        ([0.2, 0.2], "sum to one"),
    ],
)
def test_ensemble_rejects_invalid_member_targets(
    weights: list[float],
    match: str,
) -> None:
    policy = MeanWeightEnsemblePolicy(
        member_policies={"seed_7": FixedPolicy(weights)}
    )

    with pytest.raises(ValueError, match=match):
        policy.target_weights(
            np.ones(2),
            {
                "asset_order": ["SPY", "SHY"],
                "current_weights": np.array([0.5, 0.5]),
            },
        )


def test_ensemble_requires_members_and_asset_order() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        MeanWeightEnsemblePolicy(member_policies={})

    policy = MeanWeightEnsemblePolicy(
        member_policies={"seed_7": FixedPolicy([0.5, 0.5])}
    )
    with pytest.raises(ValueError, match="asset_order"):
        policy.target_weights(np.ones(2), {})


def test_ensemble_records_the_shared_live_current_weights() -> None:
    policy = MeanWeightEnsemblePolicy(
        member_policies={
            "seed_7": FixedPolicy([0.7, 0.3]),
            "seed_42": FixedPolicy([0.3, 0.7]),
        }
    )

    policy.target_weights(
        np.ones(2),
        {
            "asset_order": ["SPY", "SHY"],
            "current_weights": np.array([0.6, 0.4]),
        },
    )

    assert {
        record.live_current_weights for record in policy.member_target_records
    } == {(0.6, 0.4)}

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pytest

from portfolio_rl.policies.overlays import PartialRebalancePolicy


class FixedPolicy:
    def __init__(self, weights: list[float]) -> None:
        self._weights = np.asarray(weights, dtype=np.float64)
        self.reset_count = 0
        self.last_info: Mapping[str, Any] | None = None

    def reset(self) -> None:
        self.reset_count += 1

    def target_weights(
        self,
        observation: np.ndarray,
        info: Mapping[str, Any],
    ) -> np.ndarray:
        del observation
        self.last_info = info
        return self._weights.copy()


def test_partial_rebalance_interpolates_from_live_current_weights() -> None:
    base = FixedPolicy([0.8, 0.2])
    overlay = PartialRebalancePolicy(base_policy=base, alpha=0.25)
    info = {
        "asset_order": ["SPY", "SHY"],
        "current_weights": np.array([0.4, 0.6]),
        "date": "2024-01-02",
    }

    target = overlay.target_weights(np.ones(2), info)

    np.testing.assert_allclose(target, [0.5, 0.5])
    assert base.last_info is info
    record = overlay.records[0]
    assert record.current_weights == (0.4, 0.6)
    assert record.raw_policy_target == (0.8, 0.2)
    np.testing.assert_allclose(record.executed_target, [0.5, 0.5])
    assert record.raw_half_l1_turnover == pytest.approx(0.4)
    assert record.executed_half_l1_turnover == pytest.approx(0.1)


def test_alpha_one_is_exact_identity_for_raw_target() -> None:
    overlay = PartialRebalancePolicy(
        base_policy=FixedPolicy([0.73, 0.27]),
        alpha=1.0,
    )

    target = overlay.target_weights(
        np.ones(2),
        {
            "asset_order": ["SPY", "SHY"],
            "current_weights": np.array([0.5, 0.5]),
        },
    )

    np.testing.assert_array_equal(target, [0.73, 0.27])
    np.testing.assert_array_equal(
        overlay.records[0].raw_policy_target,
        overlay.records[0].executed_target,
    )


def test_alpha_zero_preserves_current_weights_and_produces_no_trade() -> None:
    overlay = PartialRebalancePolicy(
        base_policy=FixedPolicy([0.9, 0.1]),
        alpha=0.0,
    )
    current = np.array([0.35, 0.65])

    target = overlay.target_weights(
        np.ones(2),
        {
            "asset_order": ["SPY", "SHY"],
            "current_weights": current,
        },
    )

    np.testing.assert_array_equal(target, current)
    record = overlay.records[0]
    assert record.raw_policy_target == (0.9, 0.1)
    assert record.executed_target == (0.35, 0.65)
    assert record.executed_half_l1_turnover == 0.0


def test_reset_clears_records_and_resets_wrapped_policy() -> None:
    base = FixedPolicy([0.6, 0.4])
    overlay = PartialRebalancePolicy(base_policy=base, alpha=0.5)
    info = {
        "asset_order": ["SPY", "SHY"],
        "current_weights": np.array([0.5, 0.5]),
    }
    overlay.target_weights(np.ones(2), info)

    overlay.reset()

    assert overlay.records == ()
    assert base.reset_count == 1


@pytest.mark.parametrize("alpha", [-0.1, 1.1, np.nan, np.inf])
def test_partial_rebalance_rejects_invalid_alpha(alpha: float) -> None:
    with pytest.raises(ValueError, match="alpha"):
        PartialRebalancePolicy(
            base_policy=FixedPolicy([0.5, 0.5]),
            alpha=alpha,
        )

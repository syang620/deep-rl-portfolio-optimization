from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from portfolio_rl.env.action import action_to_weights
from portfolio_rl.policies.sb3_policy import SB3WeightPolicy


def test_sb3_weight_policy_predicts_deterministically() -> None:
    model = FakeSB3Model(np.array([0.2, -0.2], dtype=np.float32))
    policy = SB3WeightPolicy(model=model, action_temperature=5.0)
    observation = np.array([1.0, 2.0, 0.5, 0.5], dtype=np.float32)

    policy.target_weights(observation, {"asset_order": ["SPY", "SHY"]})

    assert len(model.predict_calls) == 1
    assert model.predict_calls[0]["deterministic"] is True
    np.testing.assert_array_equal(model.predict_calls[0]["observation"], observation)


def test_sb3_weight_policy_converts_actions_to_weights() -> None:
    action = np.array([0.2, -0.2], dtype=np.float32)
    policy = SB3WeightPolicy(
        model=FakeSB3Model(action),
        action_temperature=5.0,
    )

    weights = policy.target_weights(
        np.zeros(4, dtype=np.float32),
        {"asset_order": ["SPY", "SHY"]},
    )

    expected = action_to_weights(action, temperature=5.0)
    assert weights.dtype == np.float32
    assert np.all(weights >= 0.0)
    assert np.isclose(weights.sum(), 1.0)
    np.testing.assert_allclose(weights, expected)


def test_sb3_weight_policy_rejects_wrong_action_shape() -> None:
    policy = SB3WeightPolicy(
        model=FakeSB3Model(np.array([[0.0, 1.0]], dtype=np.float32)),
        action_temperature=5.0,
    )

    with pytest.raises(ValueError, match="one-dimensional"):
        policy.target_weights(
            np.zeros(4, dtype=np.float32),
            {"asset_order": ["SPY", "SHY"]},
        )


def test_sb3_weight_policy_rejects_action_dimension_mismatch() -> None:
    policy = SB3WeightPolicy(
        model=FakeSB3Model(np.array([0.0, 1.0], dtype=np.float32)),
        action_temperature=5.0,
    )

    with pytest.raises(ValueError, match="action dimension"):
        policy.target_weights(
            np.zeros(5, dtype=np.float32),
            {"asset_order": ["SPY", "QQQ", "SHY"]},
        )


def test_sb3_weight_policy_rejects_invalid_temperature() -> None:
    with pytest.raises(ValueError, match="action_temperature"):
        SB3WeightPolicy(model=FakeSB3Model(np.zeros(2)), action_temperature=0.0)


class FakeSB3Model:
    def __init__(self, action: np.ndarray) -> None:
        self._action = action
        self.predict_calls: list[dict[str, Any]] = []

    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool,
    ) -> tuple[np.ndarray, None]:
        self.predict_calls.append(
            {
                "observation": observation,
                "deterministic": deterministic,
            }
        )
        return self._action, None

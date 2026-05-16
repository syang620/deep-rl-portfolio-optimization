from __future__ import annotations

import numpy as np
import pytest

from portfolio_rl.env.action import action_to_weights


def test_action_to_weights_sum_to_one() -> None:
    weights = action_to_weights(np.array([-0.2, 0.0, 0.5]), temperature=5.0)

    assert weights.dtype == np.float32
    assert np.isclose(weights.sum(), 1.0)


def test_action_to_weights_nonnegative() -> None:
    weights = action_to_weights(np.array([-1.0, 0.0, 1.0]), temperature=5.0)

    assert np.all(weights >= 0.0)


def test_zero_action_maps_to_equal_weight() -> None:
    weights = action_to_weights(np.zeros(4), temperature=5.0)

    np.testing.assert_allclose(weights, np.full(4, 0.25, dtype=np.float32))


def test_action_to_weights_is_stable_for_large_logits() -> None:
    weights = action_to_weights(np.array([-100.0, 0.0, 100.0]), temperature=100.0)

    assert np.isfinite(weights).all()
    assert np.isclose(weights.sum(), 1.0)
    assert weights.argmax() == 2


def test_action_temperature_allows_concentration() -> None:
    action = np.array([0.0, 0.5])

    low_temperature = action_to_weights(action, temperature=1.0)
    high_temperature = action_to_weights(action, temperature=5.0)

    assert high_temperature[1] > low_temperature[1]
    assert high_temperature[0] < low_temperature[0]


def test_action_to_weights_rejects_multidimensional_action() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        action_to_weights(np.array([[0.0, 1.0]]), temperature=5.0)


def test_action_to_weights_rejects_empty_action() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        action_to_weights(np.array([]), temperature=5.0)


def test_action_to_weights_rejects_nonfinite_action() -> None:
    with pytest.raises(ValueError, match="values must be finite"):
        action_to_weights(np.array([0.0, np.nan]), temperature=5.0)


@pytest.mark.parametrize("temperature", [0.0, -1.0])
def test_action_to_weights_rejects_nonpositive_temperature(temperature: float) -> None:
    with pytest.raises(ValueError, match="temperature must be positive"):
        action_to_weights(np.array([0.0, 1.0]), temperature=temperature)


def test_action_to_weights_rejects_nonfinite_temperature() -> None:
    with pytest.raises(ValueError, match="temperature must be finite"):
        action_to_weights(np.array([0.0, 1.0]), temperature=np.inf)

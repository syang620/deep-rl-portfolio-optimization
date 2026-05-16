from __future__ import annotations

from math import log

import numpy as np
import pytest

from portfolio_rl.env.reward import log_growth_reward


def test_log_growth_reward_returns_scaled_log_growth() -> None:
    reward = log_growth_reward(
        period_gross_return=1.05,
        transaction_cost_fraction=0.0,
        reward_scale=100.0,
    )

    assert reward == pytest.approx(log(1.05) * 100.0)


def test_transaction_costs_reduce_reward() -> None:
    reward_without_cost = log_growth_reward(
        period_gross_return=1.05,
        transaction_cost_fraction=0.0,
    )
    reward_with_cost = log_growth_reward(
        period_gross_return=1.05,
        transaction_cost_fraction=0.01,
    )

    assert reward_with_cost < reward_without_cost


def test_bad_net_gross_returns_penalty() -> None:
    reward = log_growth_reward(
        period_gross_return=1.0,
        transaction_cost_fraction=1.0,
        bad_gross_penalty=-123.0,
    )

    assert reward == -123.0


def test_reward_scale_changes_magnitude() -> None:
    small_scale = log_growth_reward(
        period_gross_return=1.02,
        transaction_cost_fraction=0.0,
        reward_scale=10.0,
    )
    large_scale = log_growth_reward(
        period_gross_return=1.02,
        transaction_cost_fraction=0.0,
        reward_scale=100.0,
    )

    assert large_scale == pytest.approx(small_scale * 10.0)


@pytest.mark.parametrize("period_gross_return", [0.0, -1.0])
def test_log_growth_reward_rejects_nonpositive_gross_return(
    period_gross_return: float,
) -> None:
    with pytest.raises(ValueError, match="period_gross_return must be positive"):
        log_growth_reward(
            period_gross_return=period_gross_return,
            transaction_cost_fraction=0.0,
        )


def test_log_growth_reward_rejects_nonfinite_gross_return() -> None:
    with pytest.raises(ValueError, match="period_gross_return must be finite"):
        log_growth_reward(
            period_gross_return=np.inf,
            transaction_cost_fraction=0.0,
        )


def test_log_growth_reward_rejects_negative_cost_fraction() -> None:
    with pytest.raises(ValueError, match="transaction_cost_fraction must be nonnegative"):
        log_growth_reward(
            period_gross_return=1.0,
            transaction_cost_fraction=-0.1,
        )


def test_log_growth_reward_rejects_nonfinite_cost_fraction() -> None:
    with pytest.raises(ValueError, match="transaction_cost_fraction must be finite"):
        log_growth_reward(
            period_gross_return=1.0,
            transaction_cost_fraction=np.nan,
        )


@pytest.mark.parametrize("reward_scale", [0.0, -1.0])
def test_log_growth_reward_rejects_nonpositive_reward_scale(
    reward_scale: float,
) -> None:
    with pytest.raises(ValueError, match="reward_scale must be positive"):
        log_growth_reward(
            period_gross_return=1.0,
            transaction_cost_fraction=0.0,
            reward_scale=reward_scale,
        )


def test_log_growth_reward_rejects_nonfinite_penalty() -> None:
    with pytest.raises(ValueError, match="bad_gross_penalty must be finite"):
        log_growth_reward(
            period_gross_return=1.0,
            transaction_cost_fraction=0.0,
            bad_gross_penalty=np.nan,
        )

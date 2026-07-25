from __future__ import annotations

from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("stable_baselines3")

from portfolio_rl.training.callbacks import (  # noqa: E402
    ValidationCheckpointCallback,
    is_metric_improvement,
    validation_metric_value,
)


def test_validation_metric_value_reads_metric_dict() -> None:
    value = validation_metric_value(
        {"sharpe_ratio": 1.25},
        _nav_frame(),
        "sharpe_ratio",
    )

    assert value == 1.25


def test_validation_metric_value_reads_final_nav() -> None:
    value = validation_metric_value(
        {},
        _nav_frame(nav_values=[1.01, 1.03, 1.02]),
        "final_nav",
    )

    assert value == 1.02


def test_validation_metric_value_returns_none_for_empty_final_nav() -> None:
    value = validation_metric_value(
        {},
        pd.DataFrame(columns=["date", "nav"]),
        "final_nav",
    )

    assert value is None


def test_is_metric_improvement_accepts_first_finite_value() -> None:
    assert is_metric_improvement(0.1, None) is True


def test_is_metric_improvement_requires_higher_finite_value() -> None:
    assert is_metric_improvement(0.2, 0.1) is True
    assert is_metric_improvement(0.1, 0.2) is False
    assert is_metric_improvement(None, 0.2) is False
    assert is_metric_improvement(float("nan"), 0.2) is False
    assert is_metric_improvement(np.inf, 0.2) is False


def test_training_end_evaluates_a_new_final_timestep(tmp_path) -> None:
    callback = _validation_callback(tmp_path)
    callback.num_timesteps = 100
    callback._last_eval_timestep = 75
    callback._run_validation = Mock()

    callback._on_training_end()

    callback._run_validation.assert_called_once_with()


def test_training_end_skips_an_already_evaluated_timestep(tmp_path) -> None:
    callback = _validation_callback(tmp_path)
    callback.num_timesteps = 100
    callback._last_eval_timestep = 100
    callback._run_validation = Mock()

    callback._on_training_end()

    callback._run_validation.assert_not_called()


def _validation_callback(tmp_path) -> ValidationCheckpointCallback:
    return ValidationCheckpointCallback(
        validation_store=None,
        action_temperature=0.5,
        rebalance_frequency_trading_days=5,
        transaction_cost_bps=10.0,
        eval_freq_timesteps=25,
        metric_for_best_model="sharpe_ratio",
        output_dir=tmp_path,
    )


def _nav_frame(nav_values: list[float] | None = None) -> pd.DataFrame:
    values = nav_values or [1.01, 1.02]
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-02", periods=len(values), freq="B"),
            "nav": values,
        }
    )

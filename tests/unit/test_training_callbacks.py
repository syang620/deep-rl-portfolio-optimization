from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("stable_baselines3")

from portfolio_rl.training.callbacks import (  # noqa: E402
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


def _nav_frame(nav_values: list[float] | None = None) -> pd.DataFrame:
    values = nav_values or [1.01, 1.02]
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-02", periods=len(values), freq="B"),
            "nav": values,
        }
    )

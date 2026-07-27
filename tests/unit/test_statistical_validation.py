from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from portfolio_rl.config.schemas import StatisticalValidationConfig
from portfolio_rl.evaluation.statistical_validation import (
    ActiveReturnBootstrapResult,
    build_bootstrap_summary,
    calculate_bootstrap_samples,
    calculate_observed_metrics,
    circular_block_bootstrap_indices,
    format_bootstrap_report,
    prepare_validation_returns,
)
from scripts import analyze_active_return as bootstrap_script


def test_circular_block_indices_are_deterministic_and_consecutive() -> None:
    first = circular_block_bootstrap_indices(
        observation_count=7,
        iterations=3,
        block_length=3,
        random_seed=42,
    )
    second = circular_block_bootstrap_indices(
        observation_count=7,
        iterations=3,
        block_length=3,
        random_seed=42,
    )

    np.testing.assert_array_equal(first, second)
    assert first.shape == (3, 7)
    assert ((first[:, 1:3] - first[:, :2]) % 7 == 1).all()
    assert ((first[:, 4:6] - first[:, 3:5]) % 7 == 1).all()


@pytest.mark.parametrize(
    ("observation_count", "block_length", "match"),
    [
        (1, 1, "at least two"),
        (7, 0, "block_length"),
        (7, 8, "block_length"),
    ],
)
def test_circular_block_indices_reject_invalid_dimensions(
    observation_count: int,
    block_length: int,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        circular_block_bootstrap_indices(
            observation_count=observation_count,
            iterations=10,
            block_length=block_length,
            random_seed=42,
        )


def test_identical_returns_produce_zero_paired_differences() -> None:
    baseline = np.array([0.01, -0.005, 0.002, 0.004, -0.001])
    ppo = np.column_stack([baseline, baseline])
    seeds = [7, 42]

    observed = calculate_observed_metrics(ppo, baseline, seeds=seeds)
    samples = calculate_bootstrap_samples(
        ppo,
        baseline,
        seeds=seeds,
        config=_config(iterations=50, block_length=2),
    )

    for metric in (
        "active_total_return",
        "sharpe_ratio_delta",
        "max_drawdown_delta",
        "information_ratio",
    ):
        assert observed[metric].tolist() == pytest.approx([0.0, 0.0, 0.0])
        assert samples[metric].tolist() == pytest.approx([0.0] * len(samples))


def test_positive_alpha_has_predominantly_positive_active_return() -> None:
    baseline = np.array(
        [0.003, -0.002, 0.004, -0.001, 0.002] * 12,
        dtype=np.float64,
    )
    ppo = np.column_stack([baseline + 0.0010, baseline + 0.0015])
    seeds = [7, 42]
    config = _config(iterations=500, block_length=5)
    observed = calculate_observed_metrics(ppo, baseline, seeds=seeds)
    samples = calculate_bootstrap_samples(
        ppo,
        baseline,
        seeds=seeds,
        config=config,
    )

    summary = build_bootstrap_summary(
        observed=observed,
        samples=samples,
        dates=pd.bdate_range("2024-01-02", periods=len(baseline)),
        seeds=seeds,
        regime_name="validation_2024",
        config=config,
        sources={},
    )

    campaign = summary["groups"][0]["metrics"]["active_total_return"]
    assert campaign["observed"] > 0.0
    assert campaign["confidence_interval_lower"] > 0.0
    assert campaign["probability_positive"] == 1.0
    assert campaign["confidence_interval_excludes_zero"] is True
    assert "selection bias" in format_bootstrap_report(summary)


def test_prepare_validation_returns_aligns_dates_and_seeds() -> None:
    ppo_nav, baseline_nav = _nav_frames()

    dates, seeds, ppo_returns, baseline_returns = prepare_validation_returns(
        ppo_nav,
        baseline_nav,
        regime_name="validation_2024",
    )

    assert dates.tolist() == list(pd.bdate_range("2024-01-02", periods=4))
    assert seeds == [7, 42]
    assert ppo_returns.shape == (4, 2)
    assert baseline_returns.tolist() == pytest.approx(
        baseline_nav["daily_return"].tolist()
    )


def test_prepare_validation_returns_rejects_test_rows() -> None:
    ppo_nav, baseline_nav = _nav_frames()
    test_row = ppo_nav.iloc[[0]].assign(
        regime_name="final_test",
        split="test",
    )

    with pytest.raises(ValueError, match="must not contain test"):
        prepare_validation_returns(
            pd.concat([ppo_nav, test_row], ignore_index=True),
            baseline_nav,
            regime_name="validation_2024",
        )


def test_prepare_validation_returns_rejects_date_misalignment() -> None:
    ppo_nav, baseline_nav = _nav_frames()
    baseline_nav.loc[0, "date"] = pd.Timestamp("2024-02-01")

    with pytest.raises(ValueError, match="must align exactly"):
        prepare_validation_returns(
            ppo_nav,
            baseline_nav,
            regime_name="validation_2024",
        )


def test_prepare_validation_returns_rejects_duplicate_and_nonfinite_rows() -> None:
    ppo_nav, baseline_nav = _nav_frames()
    duplicate = pd.concat([ppo_nav, ppo_nav.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="duplicate seed-date"):
        prepare_validation_returns(
            duplicate,
            baseline_nav,
            regime_name="validation_2024",
        )

    ppo_nav.loc[0, "daily_return"] = np.nan
    with pytest.raises(ValueError, match="must be finite"):
        prepare_validation_returns(
            ppo_nav,
            baseline_nav,
            regime_name="validation_2024",
        )


def test_bootstrap_cli_prints_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    report = tmp_path / "bootstrap_report.md"
    fake = ActiveReturnBootstrapResult(
        outputs={"report": report},
        samples=pd.DataFrame(),
        summary={},
    )
    monkeypatch.setattr(
        bootstrap_script,
        "run_active_return_bootstrap",
        lambda **_kwargs: fake,
    )

    bootstrap_script.main(
        [
            "--ppo-nav",
            "ppo.parquet",
            "--baseline-nav",
            "equal_weight.parquet",
            "--output-dir",
            "bootstrap",
        ]
    )

    assert capsys.readouterr().out == f"report: {report}\n"


def _config(
    *,
    iterations: int,
    block_length: int,
) -> StatisticalValidationConfig:
    return StatisticalValidationConfig(
        bootstrap_iterations=iterations,
        block_length_trading_days=block_length,
        confidence_level=0.95,
        random_seed=42,
    )


def _nav_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.bdate_range("2024-01-02", periods=4)
    rows = []
    for seed, returns in {
        7: [0.01, -0.005, 0.002, 0.004],
        42: [0.012, -0.004, 0.003, 0.005],
    }.items():
        for date, daily_return in zip(dates, returns, strict=True):
            rows.append(
                {
                    "run_id": f"run_{seed}",
                    "seed": seed,
                    "selection_checkpoint": "best_checkpoint",
                    "regime_name": "validation_2024",
                    "split": "validation",
                    "in_sample": False,
                    "date": date,
                    "daily_return": daily_return,
                }
            )
    baseline = pd.DataFrame(
        {
            "date": dates,
            "daily_return": [0.009, -0.004, 0.002, 0.003],
        }
    )
    return pd.DataFrame(rows), baseline

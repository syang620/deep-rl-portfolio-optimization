from __future__ import annotations

import json
from pathlib import Path

import pytest

from portfolio_rl.evaluation.reports import (
    build_validation_report,
    load_metrics,
    write_validation_report,
)


def test_load_metrics_reads_json_values(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(
        json.dumps({"total_return": 0.1, "sharpe_ratio": None}),
        encoding="utf-8",
    )

    metrics = load_metrics(metrics_path)

    assert metrics == {"total_return": 0.1, "sharpe_ratio": None}


def test_load_metrics_rejects_non_numeric_values(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(
        json.dumps({"total_return": "bad"}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="numeric or null"):
        load_metrics(metrics_path)


def test_validation_report_formats_metrics_and_best_markers() -> None:
    report = build_validation_report(
        {
            "equal_weight_weekly": _metrics(
                total_return=0.10,
                sharpe_ratio=1.20,
                max_drawdown=-0.08,
            ),
            "spy_only": _metrics(
                total_return=0.20,
                sharpe_ratio=1.10,
                max_drawdown=-0.20,
            ),
            "shy_only": _metrics(
                total_return=0.03,
                sharpe_ratio=2.00,
                max_drawdown=-0.01,
            ),
        }
    )

    assert "| Strategy | Total Return | CAGR | Ann. Vol | Sharpe |" in report
    assert "spy_only (best return)" in report
    assert "shy_only (best Sharpe, lowest drawdown)" in report
    assert "20.00%" in report
    assert "2.000" in report


def test_validation_report_warns_when_ppo_underperforms_equal_weight() -> None:
    report = build_validation_report(
        {
            "ppo": _metrics(total_return=0.05, sharpe_ratio=0.80),
            "equal_weight_weekly": _metrics(total_return=0.10, sharpe_ratio=1.20),
        }
    )

    assert "## Warnings" in report
    assert "PPO underperformed equal_weight_weekly on total return." in report
    assert "PPO underperformed equal_weight_weekly on Sharpe ratio." in report


def test_validation_report_omits_warning_when_ppo_beats_equal_weight() -> None:
    report = build_validation_report(
        {
            "ppo": _metrics(total_return=0.15, sharpe_ratio=1.30),
            "equal_weight_weekly": _metrics(total_return=0.10, sharpe_ratio=1.20),
        }
    )

    assert "## Warnings" not in report


def test_validation_report_handles_missing_ppo_metrics() -> None:
    report = build_validation_report(
        {
            "equal_weight_weekly": _metrics(total_return=0.10, sharpe_ratio=1.20),
        }
    )

    assert "PPO metrics were not found" in report
    assert "equal_weight_weekly" in report


def test_write_validation_report_reads_artifacts_and_writes_markdown(
    tmp_path: Path,
) -> None:
    baseline_root = tmp_path / "baselines"
    _write_metrics(
        baseline_root / "equal_weight_weekly" / "metrics.json",
        _metrics(total_return=0.10, sharpe_ratio=1.20),
    )
    ppo_metrics_path = tmp_path / "ppo" / "metrics.json"
    _write_metrics(
        ppo_metrics_path,
        _metrics(total_return=0.05, sharpe_ratio=0.80),
    )

    output_path = write_validation_report(
        baseline_root=baseline_root,
        ppo_metrics_path=ppo_metrics_path,
        output_path=tmp_path / "validation_report.md",
    )

    report = output_path.read_text(encoding="utf-8")
    assert output_path.is_file()
    assert "ppo" in report
    assert "equal_weight_weekly" in report
    assert "Total Return" in report
    assert "PPO underperformed" in report


def _metrics(
    *,
    total_return: float,
    sharpe_ratio: float,
    max_drawdown: float = -0.05,
) -> dict[str, float | None]:
    return {
        "total_return": total_return,
        "cagr": total_return,
        "annualized_volatility": 0.10,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "average_weekly_turnover": 0.02,
        "transaction_cost_drag": 0.001,
    }


def _write_metrics(path: Path, metrics: dict[str, float | None]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics), encoding="utf-8")

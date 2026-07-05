from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.diagnose_ppo_run import (
    build_ppo_diagnostics,
    format_diagnostics_report,
    main,
)


def test_build_ppo_diagnostics_summarizes_validation_artifacts(
    tmp_path: Path,
) -> None:
    experiment_dir = tmp_path / "experiment"
    baseline_root = tmp_path / "baselines"
    _write_experiment_artifacts(experiment_dir)
    _write_json(
        baseline_root / "equal_weight_weekly" / "metrics.json",
        {
            "total_return": 0.10,
            "sharpe_ratio": 1.20,
            "average_weekly_turnover": 0.02,
            "transaction_cost_drag": 0.002,
        },
    )

    diagnostics = build_ppo_diagnostics(
        experiment_dir=experiment_dir,
        baseline_root=baseline_root,
        top_n=2,
    )

    assert diagnostics["metrics"]["total_return"] == 0.05
    assert diagnostics["turnover_summary"]["mean"] == 0.30000000000000004
    assert diagnostics["weight_concentration"]["max_weight"]["max"] == 0.8
    assert diagnostics["top_asset_counts"] == {"SPY": 1, "SHY": 1}
    assert diagnostics["largest_trades"][0]["ticker"] == "SPY"
    comparison = diagnostics["baseline_comparison"]["equal_weight_weekly"]
    assert comparison["total_return_delta"] == pytest.approx(-0.05)
    assert comparison["sharpe_ratio_delta"] == pytest.approx(-0.40)
    assert comparison["transaction_cost_drag_delta"] == pytest.approx(0.008)
    assert comparison["turnover_ratio"] == pytest.approx(15.0)


def test_format_diagnostics_report_includes_key_sections(tmp_path: Path) -> None:
    experiment_dir = tmp_path / "experiment"
    _write_experiment_artifacts(experiment_dir)
    diagnostics = build_ppo_diagnostics(
        experiment_dir=experiment_dir,
        baseline_root=None,
        top_n=1,
    )

    report = format_diagnostics_report(diagnostics)

    assert "# PPO Diagnostics" in report
    assert "## Turnover And Costs" in report
    assert "## Concentration" in report
    assert "## Largest Trades" in report


def test_diagnose_ppo_run_cli_writes_json(tmp_path: Path) -> None:
    experiment_dir = tmp_path / "experiment"
    baseline_root = tmp_path / "baselines"
    output_json = tmp_path / "diagnostics.json"
    _write_experiment_artifacts(experiment_dir)
    _write_json(
        baseline_root / "equal_weight_weekly" / "metrics.json",
        {
            "total_return": 0.10,
            "sharpe_ratio": 1.20,
            "average_weekly_turnover": 0.02,
            "transaction_cost_drag": 0.002,
        },
    )

    main(
        [
            "--experiment-dir",
            str(experiment_dir),
            "--baseline-root",
            str(baseline_root),
            "--output-json",
            str(output_json),
            "--top-n",
            "2",
        ]
    )

    loaded = json.loads(output_json.read_text(encoding="utf-8"))
    assert loaded["experiment_dir"] == str(experiment_dir)
    assert loaded["metrics"]["sharpe_ratio"] == 0.8


def _write_experiment_artifacts(experiment_dir: Path) -> None:
    experiment_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        experiment_dir / "metrics_validation.json",
        {
            "total_return": 0.05,
            "cagr": 0.05,
            "annualized_volatility": 0.10,
            "sharpe_ratio": 0.80,
            "max_drawdown": -0.04,
            "average_weekly_turnover": 0.30,
            "transaction_cost_drag": 0.01,
        },
    )
    pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-09"]),
            "strategy": ["ppo", "ppo"],
            "turnover": [0.2, 0.4],
            "transaction_cost_fraction": [0.0002, 0.0004],
        }
    ).to_parquet(experiment_dir / "validation_costs.parquet", index=False)
    pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2024-01-02", "2024-01-02", "2024-01-09", "2024-01-09"]
            ),
            "strategy": ["ppo", "ppo", "ppo", "ppo"],
            "ticker": ["SPY", "SHY", "SPY", "SHY"],
            "target_weight": [0.8, 0.2, 0.4, 0.6],
        }
    ).to_parquet(experiment_dir / "validation_weights.parquet", index=False)
    pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2024-01-02", "2024-01-02", "2024-01-09", "2024-01-09"]
            ),
            "strategy": ["ppo", "ppo", "ppo", "ppo"],
            "ticker": ["SPY", "SHY", "SPY", "SHY"],
            "pre_trade_weight": [0.5, 0.5, 0.8, 0.2],
            "target_weight": [0.8, 0.2, 0.4, 0.6],
            "trade_weight": [0.3, -0.3, -0.4, 0.4],
        }
    ).to_parquet(experiment_dir / "validation_trades.parquet", index=False)
    pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-03", "2024-01-04"]),
            "strategy": ["ppo", "ppo"],
            "nav": [1.01, 1.02],
            "daily_return": [0.01, 0.0099],
            "drawdown": [0.0, -0.01],
        }
    ).to_parquet(experiment_dir / "validation_nav.parquet", index=False)


def _write_json(path: Path, values: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(values), encoding="utf-8")

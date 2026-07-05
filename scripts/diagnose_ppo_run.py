"""CLI diagnostics for trained PPO validation artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SUMMARY_PERCENTILES = [0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
METRIC_KEYS = [
    "total_return",
    "cagr",
    "annualized_volatility",
    "sharpe_ratio",
    "max_drawdown",
    "average_weekly_turnover",
    "transaction_cost_drag",
]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose PPO validation turnover, concentration, and metrics.",
    )
    parser.add_argument(
        "--experiment-dir",
        required=True,
        help="Experiment directory containing PPO validation artifacts.",
    )
    parser.add_argument(
        "--baseline-root",
        default="artifacts/backtests/baselines_validation",
        help="Directory containing baseline strategy metrics artifacts.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to write diagnostics JSON.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of largest trades to include.",
    )
    args = parser.parse_args(argv)

    diagnostics = build_ppo_diagnostics(
        experiment_dir=Path(args.experiment_dir),
        baseline_root=Path(args.baseline_root),
        top_n=args.top_n,
    )
    print(format_diagnostics_report(diagnostics))
    if args.output_json is not None:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(diagnostics, indent=2, sort_keys=True),
            encoding="utf-8",
        )


def build_ppo_diagnostics(
    *,
    experiment_dir: str | Path,
    baseline_root: str | Path | None = "artifacts/backtests/baselines_validation",
    top_n: int = 10,
) -> dict[str, Any]:
    """Build JSON-serializable PPO validation diagnostics."""
    if top_n <= 0:
        raise ValueError("top_n must be positive")

    experiment_path = Path(experiment_dir)
    metrics = _read_json(experiment_path / "metrics_validation.json")
    costs = pd.read_parquet(experiment_path / "validation_costs.parquet")
    weights = pd.read_parquet(experiment_path / "validation_weights.parquet")
    trades = pd.read_parquet(experiment_path / "validation_trades.parquet")
    nav = pd.read_parquet(experiment_path / "validation_nav.parquet")

    diagnostics = {
        "experiment_dir": str(experiment_path),
        "metrics": _select_metrics(metrics),
        "nav_summary": _nav_summary(nav),
        "turnover_summary": _series_summary(costs["turnover"]),
        "cost_summary": _series_summary(costs["transaction_cost_fraction"]),
        "weight_concentration": _weight_concentration(weights),
        "top_asset_counts": _top_asset_counts(weights),
        "largest_trades": _largest_trades(trades, top_n=top_n),
    }
    if baseline_root is not None:
        diagnostics["baseline_comparison"] = _baseline_comparison(
            ppo_metrics=metrics,
            baseline_root=Path(baseline_root),
        )
    return diagnostics


def format_diagnostics_report(diagnostics: dict[str, Any]) -> str:
    """Format diagnostics as a concise Markdown report."""
    lines = [
        "# PPO Diagnostics",
        "",
        f"Experiment: `{diagnostics['experiment_dir']}`",
        "",
        "## Validation Metrics",
        "",
    ]
    for key, value in diagnostics["metrics"].items():
        lines.append(f"- `{key}`: {_format_value(value)}")

    lines.extend(
        [
            "",
            "## Turnover And Costs",
            "",
            f"- Average weekly turnover: "
            f"{_format_value(diagnostics['turnover_summary']['mean'])}",
            f"- Median weekly turnover: "
            f"{_format_value(diagnostics['turnover_summary']['p50'])}",
            f"- 95th percentile turnover: "
            f"{_format_value(diagnostics['turnover_summary']['p95'])}",
            f"- Mean transaction cost fraction: "
            f"{_format_value(diagnostics['cost_summary']['mean'])}",
            "",
            "## Concentration",
            "",
            f"- Mean max target weight: "
            f"{_format_value(diagnostics['weight_concentration']['max_weight']['mean'])}",
            f"- 95th percentile max target weight: "
            f"{_format_value(diagnostics['weight_concentration']['max_weight']['p95'])}",
            f"- Mean HHI: "
            f"{_format_value(diagnostics['weight_concentration']['hhi']['mean'])}",
            "",
            "## Top Assets",
            "",
        ]
    )
    for ticker, count in diagnostics["top_asset_counts"].items():
        lines.append(f"- `{ticker}`: {count}")

    if "baseline_comparison" in diagnostics:
        lines.extend(["", "## Baseline Comparison", ""])
        for strategy, values in diagnostics["baseline_comparison"].items():
            lines.append(
                "- "
                f"`{strategy}`: return delta={_format_value(values['total_return_delta'])}, "
                f"Sharpe delta={_format_value(values['sharpe_ratio_delta'])}, "
                f"cost drag delta={_format_value(values['transaction_cost_drag_delta'])}, "
                f"turnover ratio={_format_value(values['turnover_ratio'])}"
            )

    lines.extend(["", "## Largest Trades", ""])
    for trade in diagnostics["largest_trades"]:
        lines.append(
            "- "
            f"{trade['date']} `{trade['ticker']}` "
            f"trade={_format_value(trade['trade_weight'])} "
            f"pre={_format_value(trade['pre_trade_weight'])} "
            f"target={_format_value(trade['target_weight'])}"
        )
    return "\n".join(lines) + "\n"


def _select_metrics(metrics: dict[str, Any]) -> dict[str, float | None]:
    return {key: _optional_float(metrics.get(key)) for key in METRIC_KEYS}


def _nav_summary(nav: pd.DataFrame) -> dict[str, Any]:
    if nav.empty:
        raise ValueError("validation_nav.parquet must not be empty")
    nav_sorted = nav.sort_values("date").reset_index(drop=True)
    worst_drawdown_idx = int(nav_sorted["drawdown"].idxmin())
    return {
        "start_date": _to_iso(nav_sorted.loc[0, "date"]),
        "end_date": _to_iso(nav_sorted.loc[len(nav_sorted) - 1, "date"]),
        "start_nav": float(nav_sorted.loc[0, "nav"]),
        "end_nav": float(nav_sorted.loc[len(nav_sorted) - 1, "nav"]),
        "worst_drawdown_date": _to_iso(nav_sorted.loc[worst_drawdown_idx, "date"]),
        "worst_drawdown": float(nav_sorted.loc[worst_drawdown_idx, "drawdown"]),
    }


def _weight_concentration(weights: pd.DataFrame) -> dict[str, dict[str, float | None]]:
    concentration = weights.groupby("date")["target_weight"].agg(
        max_weight="max",
        hhi=lambda values: float(np.square(values).sum()),
    )
    return {
        "max_weight": _series_summary(concentration["max_weight"]),
        "hhi": _series_summary(concentration["hhi"]),
    }


def _top_asset_counts(weights: pd.DataFrame) -> dict[str, int]:
    sorted_weights = weights.sort_values(
        ["date", "target_weight"],
        ascending=[True, False],
    )
    top_assets = sorted_weights.groupby("date").head(1)
    counts = top_assets["ticker"].value_counts().sort_values(ascending=False)
    return {str(ticker): int(count) for ticker, count in counts.items()}


def _largest_trades(trades: pd.DataFrame, top_n: int) -> list[dict[str, Any]]:
    largest = trades.assign(abs_trade=trades["trade_weight"].abs()).nlargest(
        top_n,
        "abs_trade",
    )
    rows = []
    for row in largest.itertuples(index=False):
        rows.append(
            {
                "date": _to_iso(row.date),
                "ticker": str(row.ticker),
                "pre_trade_weight": float(row.pre_trade_weight),
                "target_weight": float(row.target_weight),
                "trade_weight": float(row.trade_weight),
                "abs_trade": float(row.abs_trade),
            }
        )
    return rows


def _baseline_comparison(
    *,
    ppo_metrics: dict[str, Any],
    baseline_root: Path,
) -> dict[str, dict[str, float | None]]:
    comparisons = {}
    ppo_return = _optional_float(ppo_metrics.get("total_return"))
    ppo_sharpe = _optional_float(ppo_metrics.get("sharpe_ratio"))
    ppo_turnover = _optional_float(ppo_metrics.get("average_weekly_turnover"))
    ppo_cost_drag = _optional_float(ppo_metrics.get("transaction_cost_drag"))

    for metrics_path in sorted(baseline_root.glob("*/metrics.json")):
        baseline = _read_json(metrics_path)
        baseline_return = _optional_float(baseline.get("total_return"))
        baseline_sharpe = _optional_float(baseline.get("sharpe_ratio"))
        baseline_turnover = _optional_float(baseline.get("average_weekly_turnover"))
        baseline_cost_drag = _optional_float(baseline.get("transaction_cost_drag"))
        comparisons[metrics_path.parent.name] = {
            "total_return_delta": _subtract(ppo_return, baseline_return),
            "sharpe_ratio_delta": _subtract(ppo_sharpe, baseline_sharpe),
            "transaction_cost_drag_delta": _subtract(
                ppo_cost_drag,
                baseline_cost_drag,
            ),
            "turnover_ratio": _divide(ppo_turnover, baseline_turnover),
        }
    return comparisons


def _series_summary(series: pd.Series) -> dict[str, float | None]:
    values = pd.to_numeric(series, errors="raise")
    summary = {
        "count": float(values.count()),
        "mean": float(values.mean()),
        "std": _optional_float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
    }
    for percentile in SUMMARY_PERCENTILES:
        summary[f"p{int(percentile * 100)}"] = float(values.quantile(percentile))
    return summary


def _read_json(path: Path) -> dict[str, Any]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"expected JSON object: {path}")
    return loaded


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _subtract(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return left - right


def _divide(left: float | None, right: float | None) -> float | None:
    if left is None or right is None or right == 0.0:
        return None
    return left / right


def _to_iso(value: Any) -> str:
    return pd.Timestamp(value).isoformat()


def _format_value(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


if __name__ == "__main__":
    main()

"""Markdown validation reports for Phase 2 backtests."""

from __future__ import annotations

import json
from pathlib import Path


STRATEGY_ORDER = [
    "ppo",
    "equal_weight_weekly",
    "buy_and_hold_equal_weight",
    "spy_only",
    "shy_only",
    "inverse_volatility",
]
METRIC_COLUMNS = [
    ("total_return", "Total Return", "percent"),
    ("cagr", "CAGR", "percent"),
    ("annualized_volatility", "Ann. Vol", "percent"),
    ("sharpe_ratio", "Sharpe", "decimal"),
    ("max_drawdown", "Max Drawdown", "percent"),
    ("average_weekly_turnover", "Avg Weekly Turnover", "percent"),
    ("transaction_cost_drag", "Cost Drag", "percent"),
]


def load_metrics(metrics_path: str | Path) -> dict[str, float | None]:
    """Load a metrics JSON artifact."""
    loaded = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"metrics file must contain a JSON object: {metrics_path}")
    return {
        str(key): _coerce_optional_float(value)
        for key, value in loaded.items()
    }


def collect_baseline_metrics(
    baseline_root: str | Path,
) -> dict[str, dict[str, float | None]]:
    """Collect baseline strategy metrics from one artifact root."""
    root = Path(baseline_root)
    metrics_by_strategy: dict[str, dict[str, float | None]] = {}
    for metrics_path in sorted(root.glob("*/metrics.json")):
        metrics_by_strategy[metrics_path.parent.name] = load_metrics(metrics_path)
    return metrics_by_strategy


def build_validation_report(
    metrics_by_strategy: dict[str, dict[str, float | None]],
) -> str:
    """Build a Markdown validation comparison report."""
    ordered_metrics = _ordered_metrics(metrics_by_strategy)
    lines = [
        "# Validation Backtest Comparison",
        "",
        "Generated from deterministic backtest metrics after transaction costs.",
        "",
    ]
    if "ppo" not in ordered_metrics:
        lines.extend(
            [
                "> PPO metrics were not found. This report currently summarizes "
                "baseline validation backtests only.",
                "",
            ]
        )

    lines.extend(_metrics_table(ordered_metrics))
    warnings = _underperformance_warnings(ordered_metrics)
    if warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in warnings)
    return "\n".join(lines) + "\n"


def write_validation_report(
    *,
    baseline_root: str | Path = "artifacts/backtests/baselines_validation",
    ppo_metrics_path: str | Path = "artifacts/backtests/ppo_validation/metrics.json",
    output_path: str | Path = "artifacts/backtests/validation_report.md",
) -> Path:
    """Write a Markdown validation report from baseline and optional PPO metrics."""
    metrics_by_strategy = collect_baseline_metrics(baseline_root)
    ppo_path = Path(ppo_metrics_path)
    if ppo_path.exists():
        metrics_by_strategy["ppo"] = load_metrics(ppo_path)

    report_path = Path(output_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        build_validation_report(metrics_by_strategy),
        encoding="utf-8",
    )
    return report_path


def _ordered_metrics(
    metrics_by_strategy: dict[str, dict[str, float | None]],
) -> dict[str, dict[str, float | None]]:
    ordered: dict[str, dict[str, float | None]] = {}
    for strategy in STRATEGY_ORDER:
        if strategy in metrics_by_strategy:
            ordered[strategy] = metrics_by_strategy[strategy]
    for strategy in sorted(metrics_by_strategy):
        if strategy not in ordered:
            ordered[strategy] = metrics_by_strategy[strategy]
    return ordered


def _metrics_table(
    metrics_by_strategy: dict[str, dict[str, float | None]],
) -> list[str]:
    best_markers = _best_strategy_markers(metrics_by_strategy)
    headers = ["Strategy", *[label for _key, label, _kind in METRIC_COLUMNS]]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---", *[":---:" for _ in METRIC_COLUMNS]]) + " |",
    ]
    for strategy, metrics in metrics_by_strategy.items():
        row = [_format_strategy(strategy, best_markers)]
        row.extend(
            _format_metric(metrics.get(key), kind)
            for key, _label, kind in METRIC_COLUMNS
        )
        lines.append("| " + " | ".join(row) + " |")
    return lines


def _best_strategy_markers(
    metrics_by_strategy: dict[str, dict[str, float | None]],
) -> dict[str, list[str]]:
    markers: dict[str, list[str]] = {strategy: [] for strategy in metrics_by_strategy}
    _add_best_marker(
        metrics_by_strategy,
        markers,
        metric="total_return",
        marker="best return",
        higher_is_better=True,
    )
    _add_best_marker(
        metrics_by_strategy,
        markers,
        metric="sharpe_ratio",
        marker="best Sharpe",
        higher_is_better=True,
    )
    _add_best_marker(
        metrics_by_strategy,
        markers,
        metric="max_drawdown",
        marker="lowest drawdown",
        higher_is_better=True,
    )
    return markers


def _add_best_marker(
    metrics_by_strategy: dict[str, dict[str, float | None]],
    markers: dict[str, list[str]],
    *,
    metric: str,
    marker: str,
    higher_is_better: bool,
) -> None:
    candidates = {
        strategy: values.get(metric)
        for strategy, values in metrics_by_strategy.items()
        if values.get(metric) is not None
    }
    if not candidates:
        return
    best_strategy = (
        max(candidates, key=lambda strategy: candidates[strategy])
        if higher_is_better
        else min(candidates, key=lambda strategy: candidates[strategy])
    )
    markers[best_strategy].append(marker)


def _underperformance_warnings(
    metrics_by_strategy: dict[str, dict[str, float | None]],
) -> list[str]:
    if "ppo" not in metrics_by_strategy:
        return []
    if "equal_weight_weekly" not in metrics_by_strategy:
        return []

    warnings = []
    ppo = metrics_by_strategy["ppo"]
    equal_weight = metrics_by_strategy["equal_weight_weekly"]
    if _is_less(ppo.get("total_return"), equal_weight.get("total_return")):
        warnings.append("PPO underperformed equal_weight_weekly on total return.")
    if _is_less(ppo.get("sharpe_ratio"), equal_weight.get("sharpe_ratio")):
        warnings.append("PPO underperformed equal_weight_weekly on Sharpe ratio.")
    return warnings


def _format_strategy(
    strategy: str,
    best_markers: dict[str, list[str]],
) -> str:
    markers = best_markers.get(strategy, [])
    if not markers:
        return strategy
    return f"{strategy} ({', '.join(markers)})"


def _format_metric(value: float | None, kind: str) -> str:
    if value is None:
        return "n/a"
    if kind == "percent":
        return f"{value:.2%}"
    if kind == "decimal":
        return f"{value:.3f}"
    raise ValueError(f"unsupported metric format kind: {kind}")


def _coerce_optional_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, int | float):
        return float(value)
    raise ValueError(f"metric values must be numeric or null: {value!r}")


def _is_less(left: float | None, right: float | None) -> bool:
    return left is not None and right is not None and left < right

"""Paired block-bootstrap evidence for PPO versus equal weight."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from portfolio_rl.config.loader import load_phase3_evaluation_config
from portfolio_rl.config.schemas import StatisticalValidationConfig

TRADING_DAYS_PER_YEAR = 252
BOOTSTRAP_METRICS = (
    "active_total_return",
    "sharpe_ratio_delta",
    "max_drawdown_delta",
    "information_ratio",
)
OUTPUT_FILENAMES = {
    "summary": "bootstrap_summary.json",
    "samples": "bootstrap_samples.parquet",
    "report": "bootstrap_report.md",
}


@dataclass(frozen=True)
class ActiveReturnBootstrapResult:
    """Written bootstrap artifacts and their in-memory values."""

    outputs: dict[str, Path]
    samples: pd.DataFrame
    summary: dict[str, Any]


def run_active_return_bootstrap(
    *,
    ppo_nav_path: str | Path,
    baseline_nav_path: str | Path,
    evaluation_config_path: str | Path,
    output_dir: str | Path,
    regime_name: str = "validation_2024",
    root: str | Path = ".",
) -> ActiveReturnBootstrapResult:
    """Analyze selected PPO returns against a paired validation benchmark."""
    root_path = Path(root)
    resolved_ppo_path = _resolve_path(root_path, ppo_nav_path)
    resolved_baseline_path = _resolve_path(root_path, baseline_nav_path)
    resolved_config_path = _resolve_path(root_path, evaluation_config_path)
    destination = _resolve_path(root_path, output_dir)

    evaluation_config = load_phase3_evaluation_config(resolved_config_path)
    ppo_nav = pd.read_parquet(resolved_ppo_path)
    baseline_nav = pd.read_parquet(resolved_baseline_path)
    dates, seeds, ppo_returns, baseline_returns = prepare_validation_returns(
        ppo_nav,
        baseline_nav,
        regime_name=regime_name,
    )
    observed = calculate_observed_metrics(
        ppo_returns,
        baseline_returns,
        seeds=seeds,
    )
    samples = calculate_bootstrap_samples(
        ppo_returns,
        baseline_returns,
        seeds=seeds,
        config=evaluation_config.statistical_validation,
    )
    summary = build_bootstrap_summary(
        observed=observed,
        samples=samples,
        dates=dates,
        seeds=seeds,
        regime_name=regime_name,
        config=evaluation_config.statistical_validation,
        sources={
            "ppo_nav": _source(resolved_ppo_path, root_path),
            "baseline_nav": _source(resolved_baseline_path, root_path),
            "evaluation_config": _source(resolved_config_path, root_path),
        },
    )

    destination.mkdir(parents=True, exist_ok=True)
    outputs = {
        key: destination / filename for key, filename in OUTPUT_FILENAMES.items()
    }
    samples.to_parquet(outputs["samples"], index=False)
    outputs["summary"].write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    outputs["report"].write_text(
        format_bootstrap_report(summary),
        encoding="utf-8",
    )
    return ActiveReturnBootstrapResult(
        outputs=outputs,
        samples=samples,
        summary=summary,
    )


def prepare_validation_returns(
    ppo_nav: pd.DataFrame,
    baseline_nav: pd.DataFrame,
    *,
    regime_name: str,
) -> tuple[pd.DatetimeIndex, list[int], np.ndarray, np.ndarray]:
    """Validate and align selected PPO and benchmark daily returns."""
    _require_columns(
        ppo_nav,
        [
            "run_id",
            "seed",
            "selection_checkpoint",
            "regime_name",
            "split",
            "in_sample",
            "date",
            "daily_return",
        ],
        "PPO NAV",
    )
    _require_columns(
        baseline_nav,
        ["date", "daily_return"],
        "baseline NAV",
    )
    if (ppo_nav["split"] == "test").any():
        raise ValueError("PPO NAV must not contain test-split rows")

    selected = ppo_nav[ppo_nav["regime_name"] == regime_name].copy()
    if selected.empty:
        raise ValueError(f"PPO NAV missing regime: {regime_name}")
    if set(selected["split"]) != {"validation"}:
        raise ValueError("bootstrap regime must contain only validation rows")
    if selected["in_sample"].astype(bool).any():
        raise ValueError("bootstrap regime must be out of sample")
    if set(selected["selection_checkpoint"]) != {"best_checkpoint"}:
        raise ValueError("bootstrap requires selected best checkpoints")
    if selected.duplicated(["seed", "date"]).any():
        raise ValueError("PPO NAV contains duplicate seed-date rows")
    _validate_return_values(
        selected["daily_return"].to_numpy(dtype=np.float64),
        label="PPO",
    )
    run_counts = selected.groupby("seed")["run_id"].nunique()
    if not (run_counts == 1).all():
        raise ValueError("each seed must map to exactly one selected run")

    selected["date"] = pd.to_datetime(selected["date"])
    pivot = selected.pivot(
        index="date",
        columns="seed",
        values="daily_return",
    ).sort_index()
    if pivot.isna().any().any():
        raise ValueError("PPO seeds must share identical validation dates")
    if pivot.shape[1] < 2:
        raise ValueError("bootstrap requires at least two PPO seeds")

    baseline = baseline_nav[["date", "daily_return"]].copy()
    baseline["date"] = pd.to_datetime(baseline["date"])
    if baseline["date"].duplicated().any():
        raise ValueError("baseline NAV contains duplicate dates")
    baseline = baseline.sort_values("date").set_index("date")
    if not pivot.index.equals(baseline.index):
        raise ValueError("PPO and baseline validation dates must align exactly")

    ppo_returns = pivot.to_numpy(dtype=np.float64)
    baseline_returns = baseline["daily_return"].to_numpy(dtype=np.float64)
    _validate_return_values(ppo_returns, label="PPO")
    _validate_return_values(baseline_returns, label="baseline")
    if len(pivot) < 2:
        raise ValueError("bootstrap requires at least two validation observations")
    return (
        pd.DatetimeIndex(pivot.index),
        [int(seed) for seed in pivot.columns],
        ppo_returns,
        baseline_returns,
    )


def circular_block_bootstrap_indices(
    *,
    observation_count: int,
    iterations: int,
    block_length: int,
    random_seed: int,
) -> np.ndarray:
    """Generate reproducible circular moving-block bootstrap indices."""
    if observation_count < 2:
        raise ValueError("observation_count must be at least two")
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if block_length <= 0 or block_length > observation_count:
        raise ValueError(
            "block_length must be positive and no greater than observation_count"
        )
    block_count = int(np.ceil(observation_count / block_length))
    generator = np.random.default_rng(random_seed)
    starts = generator.integers(
        0,
        observation_count,
        size=(iterations, block_count),
    )
    offsets = np.arange(block_length)
    return (
        (starts[:, :, None] + offsets) % observation_count
    ).reshape(iterations, -1)[:, :observation_count]


def calculate_observed_metrics(
    ppo_returns: np.ndarray,
    baseline_returns: np.ndarray,
    *,
    seeds: list[int],
) -> pd.DataFrame:
    """Calculate observed paired metrics by seed and across-seed median."""
    _validate_return_matrices(ppo_returns, baseline_returns, seeds=seeds)
    metrics = _paired_metrics(
        ppo_returns[None, :, :],
        baseline_returns[None, :],
    )
    seed_frame = pd.DataFrame(
        {
            "aggregation": "seed",
            "seed": pd.array(seeds, dtype="Int64"),
            **{
                metric: values[0]
                for metric, values in metrics.items()
            },
        }
    )
    campaign = {
        "aggregation": "campaign_median",
        "seed": pd.NA,
        **{
            metric: float(np.median(seed_frame[metric]))
            for metric in BOOTSTRAP_METRICS
        },
    }
    result = pd.concat(
        [pd.DataFrame([campaign]), seed_frame],
        ignore_index=True,
    )
    result["seed"] = result["seed"].astype("Int64")
    return result


def calculate_bootstrap_samples(
    ppo_returns: np.ndarray,
    baseline_returns: np.ndarray,
    *,
    seeds: list[int],
    config: StatisticalValidationConfig,
) -> pd.DataFrame:
    """Generate paired bootstrap metric samples by seed and campaign median."""
    _validate_return_matrices(ppo_returns, baseline_returns, seeds=seeds)
    indices = circular_block_bootstrap_indices(
        observation_count=len(baseline_returns),
        iterations=config.bootstrap_iterations,
        block_length=config.block_length_trading_days,
        random_seed=config.random_seed,
    )
    baseline_resampled = baseline_returns[indices]
    ppo_resampled = ppo_returns[indices, :]
    metrics = _paired_metrics(ppo_resampled, baseline_resampled)
    iterations = config.bootstrap_iterations
    seed_count = len(seeds)
    seed_frame = pd.DataFrame(
        {
            "replicate": np.repeat(np.arange(iterations), seed_count),
            "aggregation": "seed",
            "seed": np.tile(seeds, iterations),
            **{
                metric: values.reshape(-1)
                for metric, values in metrics.items()
            },
        }
    )
    campaign_frame = pd.DataFrame(
        {
            "replicate": np.arange(iterations),
            "aggregation": "campaign_median",
            "seed": pd.NA,
            **{
                metric: np.median(values, axis=1)
                for metric, values in metrics.items()
            },
        }
    )
    result = pd.concat(
        [campaign_frame, seed_frame],
        ignore_index=True,
    )
    result["seed"] = result["seed"].astype("Int64")
    if not np.isfinite(result[list(BOOTSTRAP_METRICS)].to_numpy()).all():
        raise ValueError("bootstrap produced non-finite metrics")
    return result


def build_bootstrap_summary(
    *,
    observed: pd.DataFrame,
    samples: pd.DataFrame,
    dates: pd.DatetimeIndex,
    seeds: list[int],
    regime_name: str,
    config: StatisticalValidationConfig,
    sources: dict[str, dict[str, str]],
) -> dict[str, Any]:
    """Summarize point estimates and percentile bootstrap intervals."""
    alpha = (1.0 - config.confidence_level) / 2.0
    groups = []
    for observed_row in observed.to_dict(orient="records"):
        aggregation = str(observed_row["aggregation"])
        raw_seed = observed_row["seed"]
        seed = None if pd.isna(raw_seed) else int(raw_seed)
        group_samples = samples[samples["aggregation"] == aggregation]
        if seed is not None:
            group_samples = group_samples[group_samples["seed"] == seed]
        if group_samples.empty:
            raise ValueError(
                f"bootstrap samples missing group: {aggregation}, seed={seed}"
            )
        metric_results = {}
        for metric in BOOTSTRAP_METRICS:
            values = group_samples[metric].to_numpy(dtype=np.float64)
            lower, upper = np.quantile(values, [alpha, 1.0 - alpha])
            metric_results[metric] = {
                "observed": float(observed_row[metric]),
                "confidence_interval_lower": float(lower),
                "confidence_interval_upper": float(upper),
                "probability_positive": float(np.mean(values > 0.0)),
                "confidence_interval_excludes_zero": bool(
                    lower > 0.0 or upper < 0.0
                ),
            }
        groups.append(
            {
                "aggregation": aggregation,
                "seed": seed,
                "metrics": metric_results,
            }
        )
    return {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "analysis_type": "paired_circular_moving_block_bootstrap",
        "validation_only": True,
        "test_split_used": False,
        "advisory_only": True,
        "regime_name": regime_name,
        "observation_count": len(dates),
        "start_date": dates.min().date().isoformat(),
        "end_date": dates.max().date().isoformat(),
        "seed_count": len(seeds),
        "seeds": seeds,
        "methodology": {
            **config.model_dump(mode="json"),
            "paired_sampling": True,
            "shared_indices_across_seeds": True,
            "confidence_interval_method": "percentile",
            "positive_max_drawdown_delta_meaning": (
                "PPO maximum drawdown was less severe than equal weight"
            ),
        },
        "groups": groups,
        "warnings": [
            (
                "Validation informed model and checkpoint selection, so these "
                "intervals do not remove selection bias."
            ),
            (
                "The five policy seeds share one market history and are not "
                "treated as independent return samples."
            ),
            "This advisory analysis does not authorize final-test access.",
        ],
        "sources": sources,
    }


def format_bootstrap_report(summary: dict[str, Any]) -> str:
    """Format bootstrap evidence as a concise Markdown report."""
    confidence = float(summary["methodology"]["confidence_level"])
    lines = [
        "# PPO Active-Return Bootstrap Evidence",
        "",
        f"Regime: `{summary['regime_name']}`",
        "",
        (
            f"Observations: {summary['observation_count']}; seeds: "
            f"{summary['seed_count']}; bootstrap replications: "
            f"{summary['methodology']['bootstrap_iterations']}."
        ),
        "",
        (
            "Paired circular moving blocks preserve short-horizon dependence "
            "and apply the same sampled dates to every policy and benchmark."
        ),
        "",
        "## Results",
        "",
        (
            f"| Group | Metric | Observed | {confidence:.0%} interval | "
            "Probability positive |"
        ),
        "|---|---|---:|---:|---:|",
    ]
    for group in summary["groups"]:
        group_name = (
            "Campaign median"
            if group["aggregation"] == "campaign_median"
            else f"Seed {group['seed']}"
        )
        for metric, result in group["metrics"].items():
            lines.append(
                f"| {group_name} | `{metric}` | "
                f"{_format_metric(metric, result['observed'])} | "
                f"[{_format_metric(metric, result['confidence_interval_lower'])}, "
                f"{_format_metric(metric, result['confidence_interval_upper'])}] | "
                f"{result['probability_positive']:.1%} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "Positive active return, Sharpe delta, information ratio, and "
                "maximum-drawdown delta favor PPO."
            ),
            "",
            "This is advisory validation evidence and does not change selection.",
            "",
            "## Limitations",
            "",
            *[f"- {warning}" for warning in summary["warnings"]],
            "",
            "The test split was not accessed.",
            "",
        ]
    )
    return "\n".join(lines)


def _paired_metrics(
    ppo_returns: np.ndarray,
    baseline_returns: np.ndarray,
) -> dict[str, np.ndarray]:
    baseline_total_return = np.prod(1.0 + baseline_returns, axis=1) - 1.0
    ppo_total_return = np.prod(1.0 + ppo_returns, axis=1) - 1.0
    active_returns = ppo_returns - baseline_returns[:, :, None]
    return {
        "active_total_return": (
            ppo_total_return - baseline_total_return[:, None]
        ),
        "sharpe_ratio_delta": (
            _annualized_ratio(ppo_returns)
            - _annualized_ratio(baseline_returns)[:, None]
        ),
        "max_drawdown_delta": (
            _maximum_drawdown(ppo_returns)
            - _maximum_drawdown(baseline_returns)[:, None]
        ),
        "information_ratio": _annualized_ratio(active_returns),
    }


def _annualized_ratio(returns: np.ndarray) -> np.ndarray:
    mean = np.mean(returns, axis=1)
    standard_deviation = np.std(returns, axis=1, ddof=1)
    return np.divide(
        mean,
        standard_deviation,
        out=np.zeros_like(mean),
        where=standard_deviation > np.finfo(np.float64).eps,
    ) * np.sqrt(TRADING_DAYS_PER_YEAR)


def _maximum_drawdown(returns: np.ndarray) -> np.ndarray:
    if returns.ndim == 2:
        wealth = np.cumprod(1.0 + returns, axis=1)
        peak = np.maximum.accumulate(np.maximum(wealth, 1.0), axis=1)
        return np.min(wealth / peak - 1.0, axis=1)
    if returns.ndim != 3:
        raise ValueError("returns must be a two- or three-dimensional array")
    iteration_count, observation_count, seed_count = returns.shape
    flattened = returns.transpose(0, 2, 1).reshape(
        iteration_count * seed_count,
        observation_count,
    )
    drawdowns = _maximum_drawdown(flattened)
    return drawdowns.reshape(iteration_count, seed_count)


def _validate_return_matrices(
    ppo_returns: np.ndarray,
    baseline_returns: np.ndarray,
    *,
    seeds: list[int],
) -> None:
    if ppo_returns.ndim != 2:
        raise ValueError("ppo_returns must be a two-dimensional array")
    if baseline_returns.ndim != 1:
        raise ValueError("baseline_returns must be a one-dimensional array")
    if ppo_returns.shape != (len(baseline_returns), len(seeds)):
        raise ValueError("return matrix dimensions do not match dates and seeds")
    if len(seeds) != len(set(seeds)):
        raise ValueError("seeds must be unique")
    _validate_return_values(ppo_returns, label="PPO")
    _validate_return_values(baseline_returns, label="baseline")


def _validate_return_values(values: np.ndarray, *, label: str) -> None:
    if not np.isfinite(values).all():
        raise ValueError(f"{label} returns must be finite")
    if (values <= -1.0).any():
        raise ValueError(f"{label} returns must be greater than -1")


def _format_metric(metric: str, value: float) -> str:
    if metric in {"active_total_return", "max_drawdown_delta"}:
        return f"{value:.2%}"
    return f"{value:.3f}"


def _require_columns(
    frame: pd.DataFrame,
    columns: list[str],
    label: str,
) -> None:
    missing = [column for column in columns if column not in frame]
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def _source(path: Path, root: Path) -> dict[str, str]:
    return {
        "path": _display_path(path, root),
        "sha256": _sha256_file(path),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _display_path(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _resolve_path(root: Path, path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return root / candidate

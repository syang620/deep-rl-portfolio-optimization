"""Data quality report artifact for Phase 1 outputs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from math import isfinite
from pathlib import Path
from typing import Any

import pandas as pd

from portfolio_rl.config.schemas import DataConfig, FeaturesConfig, UniverseConfig
from portfolio_rl.features.feature_spec import FeatureSpec


DEFAULT_DATA_QUALITY_REPORT_PATH = Path(
    "artifacts/reports/data_quality_report_v1.json"
)


@dataclass(frozen=True)
class DataQualityReport:
    """Summary checks for the final Phase 1 model matrix."""

    universe_name: str
    feature_version: str
    n_assets: int
    model_start_date: str
    train_end_date: str
    validation_start_date: str
    test_start_date: str
    nan_count_final: int
    inf_count_final: int
    normalization_fit_split: str
    model_matrix_row_count: int
    observation_dim: int
    model_matrix_start_date: str
    model_matrix_end_date: str
    raw_prices: dict[str, Any]
    raw_macro: dict[str, Any]
    processed_artifacts: dict[str, dict[str, Any]]


def build_data_quality_report(
    model_matrix: pd.DataFrame,
    data_config: DataConfig,
    feature_config: FeaturesConfig,
    universe_config: UniverseConfig,
    feature_spec: FeatureSpec,
    raw_prices: pd.DataFrame | None = None,
    raw_macro: pd.DataFrame | None = None,
    processed_artifacts: dict[str, pd.DataFrame] | None = None,
) -> DataQualityReport:
    """Build a Phase 1 data quality report from the final model matrix."""
    if model_matrix.empty:
        raise ValueError("model matrix is empty")

    numeric = model_matrix.select_dtypes(include="number")
    nan_count = int(numeric.isna().sum().sum())
    inf_count = int((~numeric.map(isfinite)).sum().sum()) - nan_count
    dates = pd.to_datetime(model_matrix["date"])

    return DataQualityReport(
        universe_name=universe_config.universe_name,
        feature_version=feature_config.feature_version,
        n_assets=len(universe_config.tickers),
        model_start_date=data_config.model_start_date.isoformat(),
        train_end_date=data_config.train_end_date.isoformat(),
        validation_start_date=data_config.validation_start_date.isoformat(),
        test_start_date=data_config.test_start_date.isoformat(),
        nan_count_final=nan_count,
        inf_count_final=inf_count,
        normalization_fit_split=feature_config.normalization.fit_split,
        model_matrix_row_count=len(model_matrix),
        observation_dim=feature_spec.observation_dim,
        model_matrix_start_date=dates.min().date().isoformat(),
        model_matrix_end_date=dates.max().date().isoformat(),
        raw_prices=_price_summary(raw_prices),
        raw_macro=_macro_summary(raw_macro),
        processed_artifacts=_processed_artifact_summaries(
            model_matrix,
            processed_artifacts,
        ),
    )


def save_data_quality_report(
    report: DataQualityReport,
    path: str | Path = DEFAULT_DATA_QUALITY_REPORT_PATH,
) -> None:
    """Save a Phase 1 data quality report as stable JSON."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as report_file:
        json.dump(asdict(report), report_file, indent=2)
        report_file.write("\n")


def _price_summary(prices: pd.DataFrame | None) -> dict[str, Any]:
    if prices is None:
        return {}

    summary = _frame_summary(prices)
    if "ticker" in prices.columns:
        summary["missing_count_by_ticker"] = _missing_count_by_group(prices, "ticker")
    else:
        summary["missing_count_by_ticker"] = {}
    summary["missing_count_by_column"] = _missing_count_by_column(prices)
    return summary


def _macro_summary(macro: pd.DataFrame | None) -> dict[str, Any]:
    if macro is None:
        return {}

    summary = _frame_summary(macro)
    if {"series_id", "value"}.issubset(macro.columns):
        summary["missing_value_count_by_series"] = _missing_value_count_by_series(macro)
    else:
        summary["missing_value_count_by_series"] = {}
    summary["missing_count_by_column"] = _missing_count_by_column(macro)
    return summary


def _processed_artifact_summaries(
    model_matrix: pd.DataFrame,
    processed_artifacts: dict[str, pd.DataFrame] | None,
) -> dict[str, dict[str, Any]]:
    artifacts = dict(processed_artifacts or {})
    artifacts.setdefault("model_matrix_daily", model_matrix)
    return {name: _frame_summary(frame) for name, frame in artifacts.items()}


def _frame_summary(frame: pd.DataFrame) -> dict[str, Any]:
    numeric = frame.select_dtypes(include="number")
    inf_count = int((~numeric.map(isfinite) & numeric.notna()).sum().sum())
    summary: dict[str, Any] = {
        "row_count": int(len(frame)),
        "column_count": int(len(frame.columns)),
        "missing_cell_count": int(frame.isna().sum().sum()),
        "inf_count": inf_count,
    }
    if "date" in frame.columns and not frame.empty:
        dates = pd.to_datetime(frame["date"])
        summary["start_date"] = dates.min().date().isoformat()
        summary["end_date"] = dates.max().date().isoformat()
    else:
        summary["start_date"] = None
        summary["end_date"] = None
    return summary


def _missing_count_by_column(frame: pd.DataFrame) -> dict[str, int]:
    missing = frame.isna().sum()
    return {
        str(column): int(count)
        for column, count in missing.items()
        if int(count) > 0
    }


def _missing_count_by_group(frame: pd.DataFrame, group_column: str) -> dict[str, int]:
    grouped = frame.groupby(group_column, dropna=False, sort=True)
    counts = grouped.apply(lambda group: int(group.isna().sum().sum()))
    return {
        str(group): int(count)
        for group, count in counts.items()
        if int(count) > 0
    }


def _missing_value_count_by_series(macro: pd.DataFrame) -> dict[str, int]:
    grouped = macro.groupby("series_id", dropna=False, sort=True)["value"]
    counts = grouped.apply(lambda values: int(values.isna().sum()))
    return {
        str(series_id): int(count)
        for series_id, count in counts.items()
        if int(count) > 0
    }

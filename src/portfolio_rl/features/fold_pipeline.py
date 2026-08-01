"""Deterministic, leakage-safe walk-forward feature artifact construction."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any

import pandas as pd

from portfolio_rl.config.loader import load_features_config
from portfolio_rl.data.dataset import build_portfolio_dataset
from portfolio_rl.data.feature_store import PortfolioFeatureStore
from portfolio_rl.data.walk_forward import (
    WalkForwardConfig,
    WalkForwardFold,
    assign_fold_splits,
    load_walk_forward_config,
)
from portfolio_rl.features.feature_spec import load_feature_spec
from portfolio_rl.features.model_matrix import build_model_matrix
from portfolio_rl.features.normalization import (
    NormalizationArtifactBundle,
    fit_normalization_artifact,
    save_normalization_artifact,
    transform_features,
)

GLOBAL_IDENTIFIERS = ("date", "split", "feature_version")


@dataclass(frozen=True)
class WalkForwardBuildResult:
    output_root: Path
    campaign_manifest: Path
    fold_directories: dict[str, Path]


def build_walk_forward_artifacts(
    *,
    config_path: str | Path,
    root: str | Path = ".",
    output_root: str | Path | None = None,
) -> WalkForwardBuildResult:
    """Build all four fold bundles atomically from pre-normalization panels."""
    root_path = Path(root).resolve()
    resolved_config = _resolve(root_path, config_path)
    config = load_walk_forward_config(resolved_config)
    destination = _resolve(root_path, output_root or config.output_root)
    if destination.exists():
        raise FileExistsError(f"walk-forward output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    sources = _load_and_verify_sources(root_path, config)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    try:
        fold_manifest_hashes = {}
        for fold in config.folds:
            fold_dir = temporary / fold.fold_id
            _build_fold(
                fold=fold,
                fold_dir=fold_dir,
                config=config,
                sources=sources,
            )
            fold_manifest_hashes[fold.fold_id] = _sha256(
                fold_dir / "fold_manifest.json"
            )
        campaign = {
            "schema_version": 1,
            "study": "walk_forward_data_artifacts",
            "fold_order": [fold.fold_id for fold in config.folds],
            "fold_manifest_sha256": fold_manifest_hashes,
            "checkpoint_selection_performed": False,
            "ppo_training_performed": False,
            "latest_outer_evaluation_end": "2023-12-31",
            "contains_2024_or_later": False,
            "config_sha256": _sha256(resolved_config),
        }
        _write_json(temporary / "walk_forward_manifest.json", campaign)
        os.replace(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return WalkForwardBuildResult(
        output_root=destination,
        campaign_manifest=destination / "walk_forward_manifest.json",
        fold_directories={
            fold.fold_id: destination / fold.fold_id for fold in config.folds
        },
    )


def _build_fold(
    *,
    fold: WalkForwardFold,
    fold_dir: Path,
    config: WalkForwardConfig,
    sources: dict[str, Any],
) -> None:
    fold_dir.mkdir(parents=True)
    asset = assign_fold_splits(sources["asset_features"], fold)
    global_frame = assign_fold_splits(sources["global_features"], fold)
    _validate_panel_alignment(asset, global_frame)
    feature_config = sources["features_config"]
    feature_spec = sources["feature_spec"]
    asset_scaler = fit_normalization_artifact(
        asset,
        feature_config,
        fit_split="inner_train",
    )
    global_scaler = fit_normalization_artifact(
        global_frame,
        feature_config,
        identifier_columns=GLOBAL_IDENTIFIERS,
        fit_split="inner_train",
    )
    normalized_asset = transform_features(asset, asset_scaler)
    normalized_global = transform_features(
        global_frame,
        global_scaler,
        identifier_columns=GLOBAL_IDENTIFIERS,
    )
    model_matrix = build_model_matrix(
        normalized_asset,
        normalized_global,
        asset,
        feature_spec,
    )
    if pd.to_datetime(model_matrix["date"]).max() >= pd.Timestamp("2024-01-01"):
        raise ValueError(f"fold contains prohibited 2024+ rows: {fold.fold_id}")

    training_view = model_matrix[
        model_matrix["split"].isin(["inner_train", "inner_validation"])
    ].reset_index(drop=True)
    outer_view = model_matrix[
        model_matrix["split"] == "outer_evaluation"
    ].reset_index(drop=True)
    if "outer_evaluation" in set(training_view["split"]):
        raise ValueError("training-selection view contains outer rows")

    paths = {
        "model_matrix": fold_dir / "model_matrix_daily.parquet",
        "training_selection_matrix": (
            fold_dir / "training_selection_matrix_daily.parquet"
        ),
        "outer_evaluation_matrix": (
            fold_dir / "outer_evaluation_matrix_daily.parquet"
        ),
        "feature_spec": fold_dir / "feature_spec.json",
        "scaler": fold_dir / "scaler.pkl",
        "quality": fold_dir / "data_quality_report.json",
        "summary": fold_dir / "split_summary.json",
    }
    model_matrix.to_parquet(paths["model_matrix"], index=False)
    training_view.to_parquet(paths["training_selection_matrix"], index=False)
    outer_view.to_parquet(paths["outer_evaluation_matrix"], index=False)
    shutil.copyfile(sources["feature_spec_path"], paths["feature_spec"])
    save_normalization_artifact(
        NormalizationArtifactBundle(
            asset_features=asset_scaler,
            global_features=global_scaler,
        ),
        paths["scaler"],
    )
    split_summary = _split_summary(model_matrix, asset)
    _write_json(paths["summary"], split_summary)
    quality = _quality_report(
        fold=fold,
        model_matrix=model_matrix,
        normalized_asset=normalized_asset,
        normalized_global=normalized_global,
        asset_scaler=asset_scaler,
        global_scaler=global_scaler,
        feature_spec=feature_spec,
    )
    _write_json(paths["quality"], quality)
    artifact_hashes = {
        name: {"path": path.name, "sha256": _sha256(path)}
        for name, path in paths.items()
    }
    manifest = {
        "schema_version": 1,
        "fold_id": fold.fold_id,
        "periods": _fold_periods(fold),
        "realized_splits": split_summary,
        "source_artifacts": {
            name: {
                "path": config.source_paths[name],
                "sha256": config.source_hashes[name],
            }
            for name in config.source_paths
        },
        "raw_history_start_date": config.raw_history_start_date,
        "maximum_feature_lookback_trading_days": (
            config.maximum_feature_lookback_trading_days
        ),
        "warmup_history_precedes_model_period": True,
        "normalization": {
            "fit_split": "inner_train",
            "winsorization_fit_split": "inner_train",
            "asset_fit_rows": int((asset["split"] == "inner_train").sum()),
            "global_fit_rows": int(
                (global_frame["split"] == "inner_train").sum()
            ),
        },
        "access_contract": {
            "training_split": "inner_train",
            "checkpoint_selection_split": "inner_validation",
            "outer_evaluation_split": "outer_evaluation",
            "training_selection_input": paths[
                "training_selection_matrix"
            ].name,
            "outer_evaluation_input": paths["outer_evaluation_matrix"].name,
            "checkpoint_selection_performed": False,
            "outer_accessed_during_training_or_selection": False,
        },
        "feature_spec_matches_canonical": (
            _sha256(paths["feature_spec"])
            == config.source_hashes["feature_spec"]
        ),
        "model_matrix_column_order": list(model_matrix.columns),
        "contains_2024_or_later": False,
        "artifact_hashes": artifact_hashes,
    }
    _write_json(fold_dir / "fold_manifest.json", manifest)


def _load_and_verify_sources(
    root: Path,
    config: WalkForwardConfig,
) -> dict[str, Any]:
    paths = {name: _resolve(root, path) for name, path in config.source_paths.items()}
    for name, path in paths.items():
        actual = _sha256(path)
        expected = config.source_hashes[name]
        if actual != expected:
            raise ValueError(
                f"walk-forward source hash mismatch: {name}; "
                f"expected={expected}, actual={actual}"
            )
    return {
        "asset_features": pd.read_parquet(paths["asset_features"]),
        "global_features": pd.read_parquet(paths["global_features"]),
        "feature_spec": load_feature_spec(paths["feature_spec"]),
        "feature_spec_path": paths["feature_spec"],
        "features_config": load_features_config(paths["features_config"]),
    }


def _validate_panel_alignment(asset: pd.DataFrame, global_frame: pd.DataFrame) -> None:
    asset_dates = pd.DatetimeIndex(pd.to_datetime(asset["date"].unique())).sort_values()
    global_dates = pd.DatetimeIndex(pd.to_datetime(global_frame["date"])).sort_values()
    if not asset_dates.equals(global_dates):
        raise ValueError("fold asset/global dates do not align")
    asset_splits = asset[["date", "split"]].drop_duplicates().sort_values("date")
    global_splits = global_frame[["date", "split"]].sort_values("date")
    if not asset_splits.reset_index(drop=True).equals(
        global_splits.reset_index(drop=True)
    ):
        raise ValueError("fold asset/global split labels do not align")


def _quality_report(**kwargs: Any) -> dict[str, Any]:
    fold = kwargs["fold"]
    matrix = kwargs["model_matrix"]
    normalized_asset = kwargs["normalized_asset"]
    normalized_global = kwargs["normalized_global"]
    asset_scaler = kwargs["asset_scaler"]
    global_scaler = kwargs["global_scaler"]
    feature_spec = kwargs["feature_spec"]
    numeric = matrix.select_dtypes(include="number")
    dataset = build_portfolio_dataset(matrix, feature_spec)
    outer_store = PortfolioFeatureStore(dataset, "outer_evaluation")
    boundary_enforced = False
    try:
        outer_store.get_forward_log_returns(outer_store.n_rows - 1, 1)
    except IndexError:
        boundary_enforced = True
    asset_train = normalized_asset[normalized_asset["split"] == "inner_train"]
    global_train = normalized_global[
        normalized_global["split"] == "inner_train"
    ]
    return {
        "schema_version": 1,
        "fold_id": fold.fold_id,
        "row_count": len(matrix),
        "column_count": len(matrix.columns),
        "nan_count": int(numeric.isna().sum().sum()),
        "inf_count": int((~numeric.map(isfinite) & numeric.notna()).sum().sum()),
        "feature_version": feature_spec.feature_version,
        "observation_dim": feature_spec.observation_dim,
        "normalization_fit_split": "inner_train",
        "asset_fit_rows": len(asset_train),
        "global_fit_rows": len(global_train),
        "asset_inner_train_max_abs_mean": float(
            asset_train[asset_scaler.feature_columns].mean().abs().max()
        ),
        "global_inner_train_max_abs_mean": float(
            global_train[global_scaler.feature_columns].mean().abs().max()
        ),
        "feature_order_matches_spec": True,
        "return_columns_use_unnormalized_same_date_ret_1d": True,
        "outer_forward_boundary_enforced": boundary_enforced,
        "contains_2024_or_later": False,
    }


def _split_summary(matrix: pd.DataFrame, asset: pd.DataFrame) -> dict[str, Any]:
    summaries = {}
    for split, frame in matrix.groupby("split", sort=False):
        dates = pd.to_datetime(frame["date"])
        summaries[str(split)] = {
            "model_matrix_rows": len(frame),
            "asset_feature_rows": int((asset["split"] == split).sum()),
            "start_date": dates.min().date().isoformat(),
            "end_date": dates.max().date().isoformat(),
        }
    return summaries


def _fold_periods(fold: WalkForwardFold) -> dict[str, dict[str, str]]:
    return {
        name: {
            "start": period.start.date().isoformat(),
            "end": period.end.date().isoformat(),
        }
        for name, period in (
            ("inner_train", fold.inner_train),
            ("inner_validation", fold.inner_validation),
            ("outer_evaluation", fold.outer_evaluation),
        )
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _resolve(root: Path, path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

"""Past-only reconciliation of the frozen training scaler for Phase 3B."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from portfolio_rl.features.feature_spec import load_feature_spec
from portfolio_rl.features.model_matrix import build_model_matrix
from portfolio_rl.features.normalization import (
    NormalizationArtifactBundle,
    load_normalization_artifact,
    transform_features,
)
from portfolio_rl.phase3b.execution import (
    EXPECTED_FEATURE_SPEC_SHA256,
    EXPECTED_SCALER_SHA256,
)
from portfolio_rl.phase3b.governance import GovernanceError, sha256_file

DEVELOPMENT_CUTOFF = pd.Timestamp("2023-12-31")


@dataclass(frozen=True)
class ScalerReconciliation:
    """Evidence that the observed scaler reproduces frozen pre-2024 inputs."""

    scaler_sha256: str
    feature_spec_sha256: str
    feature_version: str
    fit_split: str
    asset_feature_count: int
    global_feature_count: int
    market_feature_count: int
    compared_model_rows: int
    maximum_asset_normalization_error: float
    maximum_global_normalization_error: float
    maximum_model_matrix_error: float
    cutoff_date: str
    refit_performed: bool
    reconciled: bool

    def to_payload(self) -> dict[str, object]:
        """Return a JSON-compatible report payload."""
        return {"schema_version": 1, **asdict(self)}


def reconcile_frozen_scaler(
    *,
    scaler_path: Path,
    feature_spec_path: Path,
    raw_asset_features_path: Path,
    normalized_asset_features_path: Path,
    raw_global_features_path: Path,
    normalized_global_features_path: Path,
    model_matrix_path: Path,
    cutoff: pd.Timestamp = DEVELOPMENT_CUTOFF,
) -> ScalerReconciliation:
    """Verify the existing artifact without fitting or reading post-cutoff rows."""
    scaler_sha = sha256_file(scaler_path)
    feature_sha = sha256_file(feature_spec_path)
    if scaler_sha != EXPECTED_SCALER_SHA256:
        raise GovernanceError("observed scaler hash differs from the configured scaler")
    if feature_sha != EXPECTED_FEATURE_SPEC_SHA256:
        raise GovernanceError("feature specification hash differs from the frozen hash")
    feature_spec = load_feature_spec(feature_spec_path)
    artifact = load_normalization_artifact(scaler_path)
    if not isinstance(artifact, NormalizationArtifactBundle):
        raise GovernanceError("Phase 3B scaler must be a NormalizationArtifactBundle")
    if artifact.asset_features.feature_version != feature_spec.feature_version:
        raise GovernanceError("asset scaler feature version mismatch")
    if artifact.global_features.feature_version != feature_spec.feature_version:
        raise GovernanceError("global scaler feature version mismatch")
    if artifact.asset_features.fit_split != "train":
        raise GovernanceError("asset scaler was not fitted on the train split")
    if artifact.global_features.fit_split != "train":
        raise GovernanceError("global scaler was not fitted on the train split")
    _same_feature_set(
        artifact.asset_features.feature_columns,
        feature_spec.per_asset_features,
        "asset scaler",
    )
    _same_feature_set(
        artifact.global_features.feature_columns,
        feature_spec.global_features,
        "global scaler",
    )
    market_feature_count = (
        len(feature_spec.asset_order) * len(feature_spec.per_asset_features)
        + len(feature_spec.global_features)
    )
    if market_feature_count != 302 or feature_spec.observation_dim != 316:
        raise GovernanceError("frozen feature dimensions do not reconcile")

    raw_asset = _read_past(raw_asset_features_path, cutoff)
    normalized_asset = _read_past(normalized_asset_features_path, cutoff)
    raw_global = _read_past(raw_global_features_path, cutoff)
    normalized_global = _read_past(normalized_global_features_path, cutoff)
    frozen_matrix = _read_past(model_matrix_path, cutoff)
    transformed_asset = transform_features(raw_asset, artifact.asset_features)
    transformed_global = transform_features(
        raw_global,
        artifact.global_features,
        identifier_columns=("date", "feature_version", "split"),
    )
    asset_error = _frame_error(
        transformed_asset,
        normalized_asset,
        keys=["date", "ticker"],
        columns=feature_spec.per_asset_features,
        label="asset normalization",
    )
    global_error = _frame_error(
        transformed_global,
        normalized_global,
        keys=["date"],
        columns=feature_spec.global_features,
        label="global normalization",
    )
    rebuilt = build_model_matrix(
        transformed_asset,
        transformed_global,
        raw_asset,
        feature_spec,
    )
    observation_columns = [f"obs_{index:03d}" for index in range(316)]
    model_error = _frame_error(
        rebuilt,
        frozen_matrix,
        keys=["date"],
        columns=observation_columns,
        label="model matrix",
    )
    return ScalerReconciliation(
        scaler_sha256=scaler_sha,
        feature_spec_sha256=feature_sha,
        feature_version=feature_spec.feature_version,
        fit_split="train",
        asset_feature_count=len(feature_spec.per_asset_features),
        global_feature_count=len(feature_spec.global_features),
        market_feature_count=market_feature_count,
        compared_model_rows=len(rebuilt),
        maximum_asset_normalization_error=asset_error,
        maximum_global_normalization_error=global_error,
        maximum_model_matrix_error=model_error,
        cutoff_date=cutoff.date().isoformat(),
        refit_performed=False,
        reconciled=True,
    )


def _read_past(path: Path, cutoff: pd.Timestamp) -> pd.DataFrame:
    frame = pd.read_parquet(path, filters=[("date", "<=", cutoff)])
    if frame.empty or pd.to_datetime(frame["date"]).max() > cutoff:
        raise GovernanceError(f"past-only scaler input is invalid: {path.name}")
    return frame


def _same_feature_set(observed: list[str], expected: list[str], label: str) -> None:
    if len(observed) != len(set(observed)) or set(observed) != set(expected):
        raise GovernanceError(f"{label} feature names differ from the feature spec")


def _frame_error(
    actual: pd.DataFrame,
    expected: pd.DataFrame,
    *,
    keys: list[str],
    columns: list[str],
    label: str,
) -> float:
    actual_sorted = actual.sort_values(keys, ignore_index=True)
    expected_sorted = expected.sort_values(keys, ignore_index=True)
    if len(actual_sorted) != len(expected_sorted):
        raise GovernanceError(f"{label} row count mismatch")
    for key in keys:
        left = actual_sorted[key].astype(str).to_numpy()
        right = expected_sorted[key].astype(str).to_numpy()
        if not np.array_equal(left, right):
            raise GovernanceError(f"{label} key mismatch")
    left_values = actual_sorted.loc[:, columns].to_numpy(dtype=np.float64)
    right_values = expected_sorted.loc[:, columns].to_numpy(dtype=np.float64)
    error = float(np.max(np.abs(left_values - right_values)))
    if not np.allclose(left_values, right_values, rtol=0.0, atol=1e-12):
        raise GovernanceError(f"{label} values do not reconcile")
    return error

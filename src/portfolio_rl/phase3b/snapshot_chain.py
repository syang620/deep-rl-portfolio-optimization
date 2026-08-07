"""Point-in-time snapshot validation and immutable recommendation chaining."""

from __future__ import annotations

import subprocess
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from portfolio_rl.phase3b.execution import (
    RECOMMENDATION_SIGNATURE_NAMESPACE,
    ExecutionConfig,
)
from portfolio_rl.phase3b.governance import (
    GovernanceError,
    canonical_json_bytes,
    logical_json_sha256,
    read_json,
    resolve_path,
    sha256_file,
    ssh_public_key_fingerprint,
)

STRATEGIES = (
    "candidate",
    "equal_weight_weekly",
    "buy_and_hold_equal_weight",
    "inverse_volatility",
    "momentum_63d_top3_equal_weight",
    "spy_only",
    "shy_only",
)
GENESIS_CHAIN_HASH = "0" * 64


@dataclass(frozen=True)
class PointInTimeSnapshot:
    """Verified normalized inputs available at one decision close."""

    snapshot_id: str
    decision_date: date
    as_of_close: datetime
    next_trading_date: date
    generated_at: datetime
    market_features: tuple[float, ...]
    trailing_return_dates: tuple[date, ...]
    trailing_log_returns: tuple[tuple[float, ...], ...]
    asset_order: tuple[str, ...]
    feature_version: str
    feature_spec_sha256: str
    normalization_artifact_sha256: str
    snapshot_sha256: str
    manifest_path: Path
    feature_payload_path: Path
    trailing_returns_path: Path


@dataclass(frozen=True)
class LivePortfolioState:
    """One candidate state and one state for every frozen baseline."""

    state_id: str
    as_of_date: date
    asset_order: tuple[str, ...]
    weights: Mapping[str, tuple[float, ...]]
    previous_chain_hash: str
    initial_endowment: bool
    state_sha256: str
    manifest_path: Path
    weights_path: Path


def load_point_in_time_snapshot(
    *,
    manifest_path: Path,
    repository_root: Path,
    config: ExecutionConfig,
    expected_asset_order: tuple[str, ...],
    expected_feature_version: str,
    expected_feature_spec_sha256: str,
) -> PointInTimeSnapshot:
    """Load one normalized snapshot and reject any future or mixed input."""
    root = repository_root.resolve()
    resolved_manifest = resolve_path(root, manifest_path)
    payload = read_json(resolved_manifest)
    _keys(
        payload,
        {
            "schema_version",
            "snapshot_id",
            "feature_payload_schema_version",
            "decision_date",
            "as_of_close",
            "next_trading_date",
            "generated_at",
            "feature_version",
            "feature_spec_sha256",
            "normalization_artifact_sha256",
            "asset_order",
            "source_inventory",
            "files",
            "snapshot_payload_sha256",
        },
        "snapshot manifest",
    )
    if payload["schema_version"] != 1:
        raise GovernanceError("unsupported snapshot manifest schema")
    if (
        payload["feature_payload_schema_version"]
        != config.feature_payload_schema_version
    ):
        raise GovernanceError("snapshot feature payload schema mismatch")
    snapshot_hash = _verify_payload_hash(payload, "snapshot_payload_sha256", "snapshot")
    decision_date = _date(payload["decision_date"], "decision_date")
    next_trading_date = _date(payload["next_trading_date"], "next_trading_date")
    if next_trading_date <= decision_date:
        raise GovernanceError("next trading date must follow the decision date")
    as_of_close = _datetime(payload["as_of_close"], "as_of_close")
    generated_at = _datetime(payload["generated_at"], "snapshot generated_at")
    if as_of_close.date() != decision_date or generated_at < as_of_close:
        raise GovernanceError("snapshot chronology is invalid")
    asset_order = tuple(str(value) for value in payload["asset_order"])
    if asset_order != expected_asset_order:
        raise GovernanceError("snapshot asset order mismatch")
    if payload["feature_version"] != expected_feature_version:
        raise GovernanceError("snapshot feature version mismatch")
    if payload["feature_spec_sha256"] != expected_feature_spec_sha256:
        raise GovernanceError("snapshot feature specification hash mismatch")
    if payload["normalization_artifact_sha256"] != config.normalization_artifact_sha256:
        raise GovernanceError("snapshot normalization artifact hash mismatch")
    _validate_source_inventory(
        payload["source_inventory"],
        decision_date=decision_date,
        snapshot_generated_at=generated_at,
    )
    files = payload["files"]
    _keys(files, {"feature_payload", "trailing_log_returns"}, "snapshot files")
    feature_path = _verified_file(root, files["feature_payload"], "feature payload")
    trailing_path = _verified_file(
        root, files["trailing_log_returns"], "trailing returns"
    )
    feature_frame = pd.read_parquet(feature_path)
    trailing_frame = pd.read_parquet(trailing_path)
    if dataframe_logical_sha256(feature_frame) != files["feature_payload"].get(
        "logical_sha256"
    ):
        raise GovernanceError("feature payload logical hash mismatch")
    if dataframe_logical_sha256(trailing_frame) != files["trailing_log_returns"].get(
        "logical_sha256"
    ):
        raise GovernanceError("trailing returns logical hash mismatch")
    market_features = _market_features(
        feature_frame,
        decision_date=decision_date,
        feature_version=expected_feature_version,
    )
    dates, returns = _trailing_returns(
        trailing_frame,
        decision_date=decision_date,
        asset_order=asset_order,
    )
    return PointInTimeSnapshot(
        snapshot_id=_identifier(payload["snapshot_id"], "snapshot_id"),
        decision_date=decision_date,
        as_of_close=as_of_close,
        next_trading_date=next_trading_date,
        generated_at=generated_at,
        market_features=market_features,
        trailing_return_dates=dates,
        trailing_log_returns=returns,
        asset_order=asset_order,
        feature_version=expected_feature_version,
        feature_spec_sha256=expected_feature_spec_sha256,
        normalization_artifact_sha256=config.normalization_artifact_sha256,
        snapshot_sha256=snapshot_hash,
        manifest_path=resolved_manifest,
        feature_payload_path=feature_path,
        trailing_returns_path=trailing_path,
    )


def load_live_portfolio_state(
    *,
    manifest_path: Path,
    repository_root: Path,
    config: ExecutionConfig,
    expected_asset_order: tuple[str, ...],
    decision_date: date,
) -> LivePortfolioState:
    """Load the one live state bundle used by candidate and baselines."""
    root = repository_root.resolve()
    resolved_manifest = resolve_path(root, manifest_path)
    payload = read_json(resolved_manifest)
    _keys(
        payload,
        {
            "schema_version",
            "state_id",
            "live_state_schema_version",
            "as_of_date",
            "asset_order",
            "previous_chain_hash",
            "initial_endowment",
            "weights_file",
            "state_payload_sha256",
        },
        "live-state manifest",
    )
    if payload["schema_version"] != 1:
        raise GovernanceError("unsupported live-state manifest schema")
    if payload["live_state_schema_version"] != config.live_state_schema_version:
        raise GovernanceError("live-state schema version mismatch")
    state_hash = _verify_payload_hash(payload, "state_payload_sha256", "live state")
    as_of_date = _date(payload["as_of_date"], "live-state as_of_date")
    if as_of_date != decision_date:
        raise GovernanceError("live-state date does not match snapshot decision date")
    asset_order = tuple(str(value) for value in payload["asset_order"])
    if asset_order != expected_asset_order:
        raise GovernanceError("live-state asset order mismatch")
    previous_hash = _sha(payload["previous_chain_hash"], "previous chain")
    initial_endowment = payload["initial_endowment"]
    if not isinstance(initial_endowment, bool):
        raise GovernanceError("initial_endowment must be boolean")
    weights_path = _verified_file(root, payload["weights_file"], "live weights")
    frame = pd.read_parquet(weights_path)
    if dataframe_logical_sha256(frame) != payload["weights_file"].get("logical_sha256"):
        raise GovernanceError("live weights logical hash mismatch")
    weights = _live_weights(frame, asset_order)
    if initial_endowment:
        if previous_hash != GENESIS_CHAIN_HASH:
            raise GovernanceError("initial endowment must use the genesis chain hash")
        equal = np.full(len(asset_order), 1.0 / len(asset_order))
        for strategy_weights in weights.values():
            if not np.allclose(strategy_weights, equal):
                raise GovernanceError("initial live portfolios must be equal weight")
    elif previous_hash == GENESIS_CHAIN_HASH:
        raise GovernanceError("noninitial state cannot use the genesis chain hash")
    return LivePortfolioState(
        state_id=_identifier(payload["state_id"], "state_id"),
        as_of_date=as_of_date,
        asset_order=asset_order,
        weights=weights,
        previous_chain_hash=previous_hash,
        initial_endowment=initial_endowment,
        state_sha256=state_hash,
        manifest_path=resolved_manifest,
        weights_path=weights_path,
    )


def dataframe_logical_sha256(frame: pd.DataFrame) -> str:
    """Hash ordered column names, dtypes, and values independently of Parquet bytes."""
    normalized = frame.copy()
    for column in normalized.columns:
        if pd.api.types.is_datetime64_any_dtype(normalized[column]):
            normalized[column] = pd.to_datetime(normalized[column]).map(
                lambda value: value.isoformat()
            )
    payload = {
        "columns": list(normalized.columns),
        "dtypes": [str(dtype) for dtype in frame.dtypes],
        "rows": [
            [_json_scalar(value) for value in row]
            for row in normalized.itertuples(index=False, name=None)
        ],
    }
    return logical_json_sha256(payload)


def recommendation_chain_hash(
    *,
    previous_chain_hash: str,
    snapshot_sha256: str,
    state_sha256: str,
    recommendation_content_sha256: str,
) -> str:
    """Link one immutable recommendation to its exact predecessor and inputs."""
    return logical_json_sha256(
        {
            "previous_chain_hash": _sha(previous_chain_hash, "previous chain"),
            "snapshot_sha256": _sha(snapshot_sha256, "snapshot"),
            "state_sha256": _sha(state_sha256, "live state"),
            "recommendation_content_sha256": _sha(
                recommendation_content_sha256, "recommendation content"
            ),
        }
    )


def sign_recommendation_manifest(
    *, payload: dict[str, Any], private_key_path: Path, config: ExecutionConfig
) -> str:
    """Sign canonical recommendation bytes with the approved service identity."""
    _verify_private_key_identity(private_key_path, config)
    with tempfile.TemporaryDirectory(prefix="phase3b-recommendation-sign-") as name:
        directory = Path(name)
        message = directory / "recommendation.json"
        message.write_bytes(canonical_json_bytes(payload))
        result = subprocess.run(
            [
                "ssh-keygen",
                "-Y",
                "sign",
                "-f",
                str(private_key_path),
                "-n",
                RECOMMENDATION_SIGNATURE_NAMESPACE,
                str(message),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise GovernanceError("recommendation signing failed")
        return Path(f"{message}.sig").read_text(encoding="utf-8")


def verify_recommendation_signature(
    *, payload: dict[str, Any], signature: str, config: ExecutionConfig
) -> None:
    """Verify a recommendation against the frozen service public key."""
    public_key = config.signing.public_key_path.read_text(encoding="utf-8").strip()
    with tempfile.TemporaryDirectory(prefix="phase3b-recommendation-verify-") as name:
        directory = Path(name)
        allowed = directory / "allowed_signers"
        signature_path = directory / "recommendation.sig"
        allowed.write_text(
            f"{config.signing.principal} {public_key}\n", encoding="utf-8"
        )
        signature_path.write_text(signature, encoding="utf-8")
        result = subprocess.run(
            [
                "ssh-keygen",
                "-Y",
                "verify",
                "-f",
                str(allowed),
                "-I",
                config.signing.principal,
                "-n",
                RECOMMENDATION_SIGNATURE_NAMESPACE,
                "-s",
                str(signature_path),
            ],
            input=canonical_json_bytes(payload),
            check=False,
            capture_output=True,
        )
    if result.returncode != 0:
        raise GovernanceError("recommendation signature verification failed")


def _market_features(
    frame: pd.DataFrame, *, decision_date: date, feature_version: str
) -> tuple[float, ...]:
    columns = [f"obs_market_{index:03d}" for index in range(302)]
    expected = ["date", "feature_version", *columns]
    if list(frame.columns) != expected or len(frame) != 1:
        raise GovernanceError("feature payload schema mismatch")
    if pd.Timestamp(frame.iloc[0]["date"]).date() != decision_date:
        raise GovernanceError("feature payload date mismatch")
    if str(frame.iloc[0]["feature_version"]) != feature_version:
        raise GovernanceError("feature payload version mismatch")
    values = frame.loc[:, columns].to_numpy(dtype=np.float32)[0]
    if not np.isfinite(values).all():
        raise GovernanceError("feature payload contains nonfinite values")
    return tuple(float(value) for value in values)


def _trailing_returns(
    frame: pd.DataFrame, *, decision_date: date, asset_order: tuple[str, ...]
) -> tuple[tuple[date, ...], tuple[tuple[float, ...], ...]]:
    columns = [f"return_{ticker.lower()}_1d" for ticker in asset_order]
    if list(frame.columns) != ["date", *columns] or len(frame) != 63:
        raise GovernanceError("trailing-return payload schema mismatch")
    dates = tuple(pd.Timestamp(value).date() for value in frame["date"])
    if list(dates) != sorted(set(dates)) or any(value > decision_date for value in dates):
        raise GovernanceError(
            "trailing returns must be unique, ordered, and no later than the decision close"
        )
    values = frame.loc[:, columns].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise GovernanceError("trailing returns contain nonfinite values")
    return dates, tuple(tuple(float(value) for value in row) for row in values)


def _live_weights(
    frame: pd.DataFrame, asset_order: tuple[str, ...]
) -> dict[str, tuple[float, ...]]:
    if list(frame.columns) != ["strategy", "ticker", "current_weight"]:
        raise GovernanceError("live weights schema mismatch")
    if len(frame) != len(STRATEGIES) * len(asset_order):
        raise GovernanceError("live weights row count mismatch")
    if set(frame["strategy"].astype(str)) != set(STRATEGIES):
        raise GovernanceError("live weights strategy set mismatch")
    result = {}
    for strategy in STRATEGIES:
        rows = frame.loc[frame["strategy"] == strategy]
        if list(rows["ticker"].astype(str)) != list(asset_order):
            raise GovernanceError(f"live weights asset order mismatch: {strategy}")
        values = rows["current_weight"].to_numpy(dtype=np.float64)
        if (
            not np.isfinite(values).all()
            or (values < 0).any()
            or not np.isclose(values.sum(), 1.0)
        ):
            raise GovernanceError(f"invalid live weights: {strategy}")
        result[strategy] = tuple(float(value) for value in values)
    return result


def _validate_source_inventory(
    inventory: Any, *, decision_date: date, snapshot_generated_at: datetime
) -> None:
    if not isinstance(inventory, list) or not inventory:
        raise GovernanceError("snapshot source inventory must be nonempty")
    observed = set()
    for record in inventory:
        _keys(
            record,
            {"source", "max_observation_date", "available_at", "vintage_id"},
            "snapshot source",
        )
        source = _identifier(record["source"], "source")
        if source in observed:
            raise GovernanceError("snapshot sources must be unique")
        observed.add(source)
        if (
            _date(record["max_observation_date"], "max_observation_date")
            > decision_date
        ):
            raise GovernanceError("snapshot source contains future observations")
        if (
            _datetime(record["available_at"], "source available_at")
            > snapshot_generated_at
        ):
            raise GovernanceError("snapshot source was unavailable when generated")
        _identifier(record["vintage_id"], "vintage_id")


def _verified_file(root: Path, record: Any, label: str) -> Path:
    _keys(record, {"path", "sha256", "logical_sha256"}, label)
    path = resolve_path(root, record["path"])
    if sha256_file(path) != _sha(record["sha256"], label):
        raise GovernanceError(f"{label} file hash mismatch")
    _sha(record["logical_sha256"], f"{label} logical")
    return path


def _verify_payload_hash(payload: dict[str, Any], field: str, label: str) -> str:
    recorded = _sha(payload[field], label)
    content = dict(payload)
    del content[field]
    if logical_json_sha256(content) != recorded:
        raise GovernanceError(f"{label} payload hash mismatch")
    return recorded


def _verify_private_key_identity(path: Path, config: ExecutionConfig) -> None:
    result = subprocess.run(
        ["ssh-keygen", "-y", "-f", str(path)],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise GovernanceError("cannot read recommendation private key")
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8") as public:
        public.write(result.stdout)
        public.flush()
        if (
            ssh_public_key_fingerprint(Path(public.name))
            != config.signing.public_key_fingerprint
        ):
            raise GovernanceError(
                "recommendation private key does not match approved identity"
            )


def _json_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, float) and not np.isfinite(value):
        raise GovernanceError("cannot hash nonfinite tabular value")
    return value


def _keys(payload: Any, expected: set[str], label: str) -> None:
    if not isinstance(payload, dict) or set(payload) != expected:
        raise GovernanceError(f"{label} keys mismatch")


def _identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise GovernanceError(f"{label} is unresolved")
    return value.strip()


def _date(value: Any, label: str) -> date:
    if not isinstance(value, str):
        raise GovernanceError(f"{label} must be an ISO date")
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise GovernanceError(f"{label} must be an ISO date") from exc


def _datetime(value: Any, label: str) -> datetime:
    if not isinstance(value, str):
        raise GovernanceError(f"{label} must be an ISO timestamp")
    try:
        result = datetime.fromisoformat(value)
    except ValueError as exc:
        raise GovernanceError(f"{label} must be an ISO timestamp") from exc
    if result.tzinfo is None:
        raise GovernanceError(f"{label} must be timezone-aware")
    return result.astimezone(UTC)


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise GovernanceError(f"{label} SHA-256 is invalid")
    try:
        int(value, 16)
    except ValueError as exc:
        raise GovernanceError(f"{label} SHA-256 is invalid") from exc
    return value

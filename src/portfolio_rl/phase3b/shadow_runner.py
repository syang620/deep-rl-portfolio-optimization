"""Deterministic Phase 3B candidate and baseline recommendation runner."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from portfolio_rl.phase3b.baseline_runner import generate_baseline_targets
from portfolio_rl.phase3b.execution import (
    ExecutionConfig,
    ExecutionInstruction,
    build_execution_instructions,
    require_before_recommendation_cutoff,
)
from portfolio_rl.phase3b.frozen_candidate_loader import (
    FrozenCandidateRuntime,
    runtime_identity_payload,
)
from portfolio_rl.phase3b.governance import (
    GovernanceError,
    canonical_json_bytes,
    logical_json_sha256,
    read_json,
    relative_path,
    resolve_path,
    sha256_file,
    validate_container_identity,
)
from portfolio_rl.phase3b.holdout_registry import verify_holdout_registration
from portfolio_rl.phase3b.snapshot_chain import (
    GENESIS_CHAIN_HASH,
    LivePortfolioState,
    PointInTimeSnapshot,
    dataframe_logical_sha256,
    recommendation_chain_hash,
    sign_recommendation_manifest,
    verify_recommendation_signature,
)

TARGET_FILES = (
    "member_targets.parquet",
    "ensemble_target.parquet",
    "executed_target.parquet",
    "baseline_targets.parquet",
    "current_weights.parquet",
    "execution_instructions.json",
    "incident_record.json",
)
COPIED_INPUT_FILES = (
    "input_snapshot_manifest.json",
    "input_live_state_manifest.json",
    "feature_payload.parquet",
    "trailing_log_returns.parquet",
)


@dataclass(frozen=True)
class MemberTarget:
    """One ordered frozen-member recommendation."""

    seed: int
    weights: tuple[float, ...]


@dataclass(frozen=True)
class ShadowDecision:
    """All recommendation outputs for one snapshot and live-state bundle."""

    decision_date: date
    generated_at: datetime
    snapshot: PointInTimeSnapshot
    live_state: LivePortfolioState
    member_targets: tuple[MemberTarget, ...]
    ensemble_target: tuple[float, ...]
    executed_target: tuple[float, ...]
    baseline_targets: dict[str, tuple[float, ...]]
    execution_instructions: tuple[ExecutionInstruction, ...]


def generate_shadow_decision(
    *,
    runtime: FrozenCandidateRuntime,
    snapshot: PointInTimeSnapshot,
    live_state: LivePortfolioState,
    generated_at: datetime,
    execution_config: ExecutionConfig,
) -> ShadowDecision:
    """Generate candidate and baseline targets without performance computation."""
    if generated_at.tzinfo is None:
        raise GovernanceError("recommendation timestamp must be timezone-aware")
    generated_utc = generated_at.astimezone(UTC)
    if generated_utc < snapshot.generated_at:
        raise GovernanceError("recommendation precedes snapshot generation")
    require_before_recommendation_cutoff(
        generated_at=generated_utc,
        execution_date=snapshot.next_trading_date,
        config=execution_config,
    )
    if snapshot.decision_date != live_state.as_of_date:
        raise GovernanceError("snapshot and live-state dates differ")
    if snapshot.asset_order != live_state.asset_order:
        raise GovernanceError("snapshot and live-state asset orders differ")
    if runtime.candidate_manifest_sha256 != execution_config.candidate_manifest_sha256:
        raise GovernanceError("runtime and execution config candidate hashes differ")

    candidate_current = np.asarray(live_state.weights["candidate"], dtype=np.float64)
    observation = np.concatenate(
        [np.asarray(snapshot.market_features, dtype=np.float32), candidate_current]
    ).astype(np.float32)
    if observation.shape != (runtime.feature_spec.observation_dim,):
        raise GovernanceError("candidate observation dimension mismatch")
    member_rows = []
    arrays = []
    for seed, policy in runtime.member_policies:
        info = {
            "date": pd.Timestamp(snapshot.decision_date),
            "current_weights": candidate_current.copy(),
            "asset_order": list(snapshot.asset_order),
            "trailing_log_returns": np.asarray(
                snapshot.trailing_log_returns, dtype=np.float64
            ),
        }
        target = np.asarray(
            policy.target_weights(observation.copy(), info), dtype=np.float64
        )
        _weights(target, len(snapshot.asset_order), f"member seed {seed}")
        arrays.append(target)
        member_rows.append(
            MemberTarget(seed=seed, weights=tuple(float(value) for value in target))
        )
    if tuple(row.seed for row in member_rows) != runtime.candidate.member_seeds:
        raise GovernanceError("member target order differs from frozen seed order")
    ensemble = np.mean(np.stack(arrays), axis=0)
    _weights(ensemble, len(snapshot.asset_order), "ensemble")
    executed = candidate_current + runtime.candidate.partial_rebalance_alpha * (
        ensemble - candidate_current
    )
    _weights(executed, len(snapshot.asset_order), "executed candidate")

    baseline_targets = generate_baseline_targets(
        snapshot=snapshot,
        live_state=live_state,
        baseline_definitions=runtime.baseline_definitions,
    )
    all_targets = {
        "candidate": tuple(float(value) for value in executed),
        **baseline_targets,
    }
    instructions = build_execution_instructions(
        decision_date=snapshot.decision_date,
        execution_date=snapshot.next_trading_date,
        current_weights=live_state.weights,
        targets=all_targets,
        config=execution_config,
    )
    return ShadowDecision(
        decision_date=snapshot.decision_date,
        generated_at=generated_utc,
        snapshot=snapshot,
        live_state=live_state,
        member_targets=tuple(member_rows),
        ensemble_target=tuple(float(value) for value in ensemble),
        executed_target=tuple(float(value) for value in executed),
        baseline_targets=baseline_targets,
        execution_instructions=instructions,
    )


def write_shadow_decision(
    *,
    decision: ShadowDecision,
    runtime: FrozenCandidateRuntime,
    execution_config: ExecutionConfig,
    registration_dir: Path,
    runtime_identity_path: Path,
    signing_key_path: Path,
    repository_root: Path,
) -> Path:
    """Atomically publish one signed recommendation under registered governance."""
    root = repository_root.resolve()
    registration_path = resolve_path(root, registration_dir)
    registered = verify_holdout_registration(registration_path, repository_root=root)
    registration_payload = read_json(registration_path / "holdout_registration.json")
    if registered.candidate_manifest_sha256 != runtime.candidate_manifest_sha256:
        raise GovernanceError("registered and runtime candidate hashes differ")
    verify_registered_execution_config(registration_path, execution_config)
    runtime_identity_resolved = resolve_path(root, runtime_identity_path)
    runtime_identity = read_json(runtime_identity_resolved)
    validate_container_identity(runtime_identity, registered.git_commit)
    if runtime_identity != read_json(registration_path / "container_identity.json"):
        raise GovernanceError("runtime container identity differs from registration")
    if not (
        registered.start_decision_date
        <= decision.decision_date
        <= registered.end_decision_date
    ):
        raise GovernanceError("decision date is outside the registered holdout")
    schedule_dates = read_json(
        resolve_path(
            root,
            _registered_challenge(registration_path)["inputs"][
                "trading_session_schedule"
            ]["path"],
        )
    )["holdout_decision_dates"]
    if decision.decision_date.isoformat() not in schedule_dates:
        raise GovernanceError("decision date is not in the registered schedule")

    holdout_root = execution_config.output_root / registered.holdout_id
    decision_root = holdout_root / "decisions"
    destination = decision_root / decision.decision_date.isoformat()
    decision_root.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        _require_equivalent_retry(
            destination=destination,
            decision=decision,
            runtime=runtime,
            execution_config=execution_config,
            registration_dir=registration_path,
            repository_root=root,
        )
        return destination

    lock = holdout_root / ".decision.lock"
    lock_fd = _acquire_lock(lock)
    try:
        if destination.exists():
            raise GovernanceError("decision appeared while acquiring the lock")
        _verify_previous_chain(decision_root, decision.live_state.previous_chain_hash)
        temporary = Path(
            tempfile.mkdtemp(prefix=f".{destination.name}.", dir=decision_root)
        )
        try:
            _write_decision_files(temporary, decision, execution_config)
            manifest = _recommendation_manifest(
                directory=temporary,
                decision=decision,
                runtime=runtime,
                execution_config=execution_config,
                registration_payload=registration_payload,
                runtime_identity_path=runtime_identity_resolved,
                runtime_identity=runtime_identity,
                repository_root=root,
            )
            signature = sign_recommendation_manifest(
                payload=manifest,
                private_key_path=resolve_path(root, signing_key_path),
                config=execution_config,
            )
            (temporary / "recommendation_manifest.json").write_bytes(
                canonical_json_bytes(manifest)
            )
            (temporary / "recommendation_manifest.sig").write_text(
                signature, encoding="utf-8"
            )
            temporary.replace(destination)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
    finally:
        os.close(lock_fd)
        lock.unlink(missing_ok=True)
    verify_shadow_decision(
        decision_dir=destination,
        execution_config=execution_config,
        registration_dir=registration_path,
        repository_root=root,
    )
    return destination


def verify_shadow_decision(
    *,
    decision_dir: Path,
    execution_config: ExecutionConfig,
    registration_dir: Path,
    repository_root: Path,
) -> dict[str, Any]:
    """Verify every decision artifact, signature, source, and chain reference."""
    root = repository_root.resolve()
    directory = resolve_path(root, decision_dir)
    registration_path = resolve_path(root, registration_dir)
    registered = verify_holdout_registration(registration_path, repository_root=root)
    verify_registered_execution_config(registration_path, execution_config)
    manifest = read_json(directory / "recommendation_manifest.json")
    signature = (directory / "recommendation_manifest.sig").read_text(encoding="utf-8")
    verify_recommendation_signature(
        payload=manifest,
        signature=signature,
        config=execution_config,
    )
    manifest_payload = dict(manifest)
    recorded_manifest_hash = manifest_payload.pop("manifest_payload_sha256", None)
    if logical_json_sha256(manifest_payload) != recorded_manifest_hash:
        raise GovernanceError("recommendation manifest payload hash mismatch")
    content = manifest.get("recommendation_content")
    if not isinstance(content, dict) or not isinstance(content.get("candidate"), dict):
        raise GovernanceError("recommendation candidate identity is missing")
    if content.get("registration_id") != registered.holdout_id:
        raise GovernanceError("recommendation holdout identity mismatch")
    if content.get("execution_config_sha256") != execution_config.config_sha256:
        raise GovernanceError("recommendation execution config mismatch")
    if (
        content["candidate"].get("candidate_manifest_sha256")
        != registered.candidate_manifest_sha256
    ):
        raise GovernanceError("recommendation candidate differs from registration")
    try:
        decision_date = date.fromisoformat(str(manifest.get("decision_date")))
    except ValueError as exc:
        raise GovernanceError("recommendation decision date is invalid") from exc
    if directory.name != decision_date.isoformat() or not (
        registered.start_decision_date <= decision_date <= registered.end_decision_date
    ):
        raise GovernanceError("recommendation decision date is not registered")
    records = manifest.get("files")
    if not isinstance(records, list):
        raise GovernanceError("recommendation file inventory is missing")
    expected = set(TARGET_FILES) | set(COPIED_INPUT_FILES)
    if {record.get("path") for record in records} != expected:
        raise GovernanceError("recommendation file inventory mismatch")
    for record in records:
        path = directory / record["path"]
        if sha256_file(path) != record["sha256"]:
            raise GovernanceError(f"recommendation artifact hash mismatch: {path.name}")
        if (
            path.suffix == ".parquet"
            and dataframe_logical_sha256(pd.read_parquet(path))
            != record["logical_sha256"]
        ):
            raise GovernanceError(
                f"recommendation artifact logical hash mismatch: {path.name}"
            )
    snapshot_input = read_json(directory / "input_snapshot_manifest.json")
    state_input = read_json(directory / "input_live_state_manifest.json")
    if snapshot_input.get("snapshot_payload_sha256") != manifest["snapshot_sha256"]:
        raise GovernanceError("copied snapshot identity mismatch")
    if state_input.get("state_payload_sha256") != manifest["state_sha256"]:
        raise GovernanceError("copied live-state identity mismatch")
    content_base = dict(content)
    content_hash = logical_json_sha256(content_base)
    if content_hash != manifest["recommendation_content_sha256"]:
        raise GovernanceError("recommendation content hash mismatch")
    expected_chain = recommendation_chain_hash(
        previous_chain_hash=manifest["previous_chain_hash"],
        snapshot_sha256=manifest["snapshot_sha256"],
        state_sha256=manifest["state_sha256"],
        recommendation_content_sha256=content_hash,
    )
    if expected_chain != manifest["chain_hash"]:
        raise GovernanceError("recommendation chain hash mismatch")
    _verify_previous_chain(directory.parent, manifest["previous_chain_hash"], directory)
    if manifest.get("performance_computed") is not False:
        raise GovernanceError("shadow decision must not compute performance")
    if manifest.get("test_accessed") is not False:
        raise GovernanceError("shadow decision indicates test access")
    return manifest


def _write_decision_files(
    directory: Path,
    decision: ShadowDecision,
    execution_config: ExecutionConfig,
) -> None:
    shutil.copy2(
        decision.snapshot.manifest_path,
        directory / "input_snapshot_manifest.json",
    )
    shutil.copy2(
        decision.live_state.manifest_path,
        directory / "input_live_state_manifest.json",
    )
    shutil.copy2(
        decision.snapshot.feature_payload_path, directory / "feature_payload.parquet"
    )
    shutil.copy2(
        decision.snapshot.trailing_returns_path,
        directory / "trailing_log_returns.parquet",
    )
    asset_order = decision.snapshot.asset_order
    member_rows = [
        {
            "decision_date": decision.decision_date,
            "snapshot_sha256": decision.snapshot.snapshot_sha256,
            "seed": member.seed,
            "ticker": ticker,
            "target_weight": weight,
        }
        for member in decision.member_targets
        for ticker, weight in zip(asset_order, member.weights, strict=True)
    ]
    _write_parquet(pd.DataFrame(member_rows), directory / "member_targets.parquet")
    _write_weight_table(
        directory / "ensemble_target.parquet",
        decision,
        "ensemble_target_weight",
        decision.ensemble_target,
    )
    _write_weight_table(
        directory / "executed_target.parquet",
        decision,
        "executed_target_weight",
        decision.executed_target,
    )
    baseline_rows = [
        {
            "decision_date": decision.decision_date,
            "snapshot_sha256": decision.snapshot.snapshot_sha256,
            "strategy": strategy,
            "ticker": ticker,
            "target_weight": weight,
        }
        for strategy in sorted(decision.baseline_targets)
        for ticker, weight in zip(
            asset_order, decision.baseline_targets[strategy], strict=True
        )
    ]
    _write_parquet(pd.DataFrame(baseline_rows), directory / "baseline_targets.parquet")
    current_rows = [
        {"strategy": strategy, "ticker": ticker, "current_weight": weight}
        for strategy in sorted(decision.live_state.weights)
        for ticker, weight in zip(
            asset_order, decision.live_state.weights[strategy], strict=True
        )
    ]
    _write_parquet(pd.DataFrame(current_rows), directory / "current_weights.parquet")
    _write_json(
        directory / "execution_instructions.json",
        _execution_payload(decision, execution_config),
    )
    _write_json(
        directory / "incident_record.json",
        {
            "schema_version": 1,
            "decision_date": decision.decision_date.isoformat(),
            "incidents": [],
            "performance_computed": False,
        },
    )


def _recommendation_manifest(
    *,
    directory: Path,
    decision: ShadowDecision,
    runtime: FrozenCandidateRuntime,
    execution_config: ExecutionConfig,
    registration_payload: dict[str, Any],
    runtime_identity_path: Path,
    runtime_identity: dict[str, Any],
    repository_root: Path,
) -> dict[str, Any]:
    files = []
    for name in (*COPIED_INPUT_FILES, *TARGET_FILES):
        path = directory / name
        record = {
            "path": name,
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        if path.suffix == ".parquet":
            record["logical_sha256"] = dataframe_logical_sha256(pd.read_parquet(path))
        files.append(record)
    content = {
        "decision_date": decision.decision_date.isoformat(),
        "generated_at": decision.generated_at.isoformat(),
        "execution_date": decision.snapshot.next_trading_date.isoformat(),
        "candidate": runtime_identity_payload(runtime, repository_root),
        "execution_config_path": relative_path(
            repository_root, execution_config.config_path
        ),
        "execution_config_sha256": execution_config.config_sha256,
        "registration_id": registration_payload["holdout_id"],
        "runtime_identity_path": relative_path(repository_root, runtime_identity_path),
        "runtime_identity_sha256": logical_json_sha256(runtime_identity),
        "files": files,
    }
    content_hash = logical_json_sha256(content)
    manifest = {
        "schema_version": 1,
        "manifest_type": "phase3b_shadow_recommendation",
        "decision_date": decision.decision_date.isoformat(),
        "generated_at": decision.generated_at.isoformat(),
        "snapshot_sha256": decision.snapshot.snapshot_sha256,
        "state_sha256": decision.live_state.state_sha256,
        "previous_chain_hash": decision.live_state.previous_chain_hash,
        "recommendation_content": content,
        "recommendation_content_sha256": content_hash,
        "chain_hash": recommendation_chain_hash(
            previous_chain_hash=decision.live_state.previous_chain_hash,
            snapshot_sha256=decision.snapshot.snapshot_sha256,
            state_sha256=decision.live_state.state_sha256,
            recommendation_content_sha256=content_hash,
        ),
        "files": files,
        "signature_namespace": execution_config.signing.namespace,
        "signature_principal": execution_config.signing.principal,
        "performance_computed": False,
        "test_accessed": False,
        "self_hash_contract": "sha256_of_canonical_json_without_manifest_payload_sha256",
    }
    manifest["manifest_payload_sha256"] = logical_json_sha256(manifest)
    return manifest


def _require_equivalent_retry(
    *,
    destination: Path,
    decision: ShadowDecision,
    runtime: FrozenCandidateRuntime,
    execution_config: ExecutionConfig,
    registration_dir: Path,
    repository_root: Path,
) -> None:
    manifest = verify_shadow_decision(
        decision_dir=destination,
        execution_config=execution_config,
        registration_dir=registration_dir,
        repository_root=repository_root,
    )
    if manifest["snapshot_sha256"] != decision.snapshot.snapshot_sha256:
        raise GovernanceError("same-date retry uses a different snapshot")
    if manifest["state_sha256"] != decision.live_state.state_sha256:
        raise GovernanceError("same-date retry uses a different live state")
    candidate_identity = manifest["recommendation_content"]["candidate"]
    if candidate_identity != runtime_identity_payload(runtime, repository_root):
        raise GovernanceError("same-date retry uses a different candidate")
    expected_frames = _decision_frames(decision)
    for filename, frame in expected_frames.items():
        if dataframe_logical_sha256(frame) != dataframe_logical_sha256(
            pd.read_parquet(destination / filename)
        ):
            raise GovernanceError(f"same-date retry output differs: {filename}")
    if read_json(destination / "execution_instructions.json") != _execution_payload(
        decision, execution_config
    ):
        raise GovernanceError("same-date retry output differs: execution instructions")


def _decision_frames(decision: ShadowDecision) -> dict[str, pd.DataFrame]:
    asset_order = decision.snapshot.asset_order
    return {
        "member_targets.parquet": pd.DataFrame(
            [
                {
                    "decision_date": decision.decision_date,
                    "snapshot_sha256": decision.snapshot.snapshot_sha256,
                    "seed": member.seed,
                    "ticker": ticker,
                    "target_weight": weight,
                }
                for member in decision.member_targets
                for ticker, weight in zip(asset_order, member.weights, strict=True)
            ]
        ),
        "ensemble_target.parquet": _weight_frame(
            decision, "ensemble_target_weight", decision.ensemble_target
        ),
        "executed_target.parquet": _weight_frame(
            decision, "executed_target_weight", decision.executed_target
        ),
        "baseline_targets.parquet": pd.DataFrame(
            [
                {
                    "decision_date": decision.decision_date,
                    "snapshot_sha256": decision.snapshot.snapshot_sha256,
                    "strategy": strategy,
                    "ticker": ticker,
                    "target_weight": weight,
                }
                for strategy in sorted(decision.baseline_targets)
                for ticker, weight in zip(
                    asset_order, decision.baseline_targets[strategy], strict=True
                )
            ]
        ),
        "current_weights.parquet": pd.DataFrame(
            [
                {"strategy": strategy, "ticker": ticker, "current_weight": weight}
                for strategy in sorted(decision.live_state.weights)
                for ticker, weight in zip(
                    asset_order, decision.live_state.weights[strategy], strict=True
                )
            ]
        ),
    }


def verify_registered_execution_config(
    registration_dir: Path, execution_config: ExecutionConfig
) -> None:
    """Require the execution config to match the signed registration hash."""
    payload = read_json(registration_dir / "config_hashes.json")
    records = payload.get("configs")
    if not isinstance(records, list) or not all(
        isinstance(record, dict) for record in records
    ):
        raise GovernanceError("registered config inventory is malformed")
    execution = next(
        (record for record in records if record.get("name") == "execution"), None
    )
    if execution is None or execution.get("sha256") != execution_config.config_sha256:
        raise GovernanceError("execution config differs from registered hash")


def _registered_challenge(registration_dir: Path) -> dict[str, Any]:
    approval = read_json(registration_dir / "approval_record.json")
    challenge = approval.get("registration_challenge")
    if not isinstance(challenge, dict):
        raise GovernanceError("registration challenge is unavailable")
    return challenge


def _verify_previous_chain(
    decisions_root: Path,
    previous_hash: str,
    current_directory: Path | None = None,
) -> None:
    directories = (
        sorted(
            path
            for path in decisions_root.iterdir()
            if path.is_dir()
            and not path.name.startswith(".")
            and path != current_directory
        )
        if decisions_root.exists()
        else []
    )
    if previous_hash == GENESIS_CHAIN_HASH:
        if directories:
            raise GovernanceError(
                "genesis chain hash is invalid after the first decision"
            )
        return
    matching = []
    for directory in directories:
        path = directory / "recommendation_manifest.json"
        if path.exists() and read_json(path).get("chain_hash") == previous_hash:
            matching.append(directory)
    if len(matching) != 1:
        raise GovernanceError(
            "previous recommendation chain hash is missing or ambiguous"
        )


def _write_weight_table(
    path: Path,
    decision: ShadowDecision,
    weight_column: str,
    weights: tuple[float, ...],
) -> None:
    _write_parquet(_weight_frame(decision, weight_column, weights), path)


def _weight_frame(
    decision: ShadowDecision, weight_column: str, weights: tuple[float, ...]
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "decision_date": decision.decision_date,
                "snapshot_sha256": decision.snapshot.snapshot_sha256,
                "ticker": ticker,
                weight_column: weight,
            }
            for ticker, weight in zip(
                decision.snapshot.asset_order, weights, strict=True
            )
        ]
    )


def _instruction_payload(instruction: ExecutionInstruction) -> dict[str, Any]:
    payload = asdict(instruction)
    payload["decision_date"] = instruction.decision_date.isoformat()
    payload["execution_date"] = instruction.execution_date.isoformat()
    payload["current_decision_weights"] = list(instruction.current_decision_weights)
    payload["target_weights"] = list(instruction.target_weights)
    return payload


def _execution_payload(
    decision: ShadowDecision, execution_config: ExecutionConfig
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "execution_contract": {
            "delay_closes": 1,
            "turnover_definition": "half_l1_one_way",
            "turnover_basis": "live_drifted_weights_at_execution_close",
            "primary_flat_cost_bps": 10.0,
            "asset_tier_cost_role": "stress_advisory",
            "asset_cost_bps": dict(execution_config.asset_cost_bps),
        },
        "instructions": [
            _instruction_payload(row) for row in decision.execution_instructions
        ],
    }


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    frame.to_parquet(path, index=False)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _weights(values: np.ndarray, n_assets: int, label: str) -> None:
    if values.shape != (n_assets,):
        raise GovernanceError(f"{label} weights have the wrong shape")
    if not np.isfinite(values).all() or (values < 0).any():
        raise GovernanceError(f"{label} weights are invalid")
    if not np.isclose(values.sum(), 1.0):
        raise GovernanceError(f"{label} weights do not sum to one")


def _acquire_lock(path: Path) -> int:
    try:
        return os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise GovernanceError("another Phase 3B decision is in progress") from exc

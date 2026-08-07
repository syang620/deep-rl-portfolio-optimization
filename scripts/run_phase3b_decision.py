"""Generate one signed Phase 3B shadow decision without reading performance."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

from portfolio_rl.phase3b.execution import load_execution_config
from portfolio_rl.phase3b.frozen_candidate_loader import (
    load_frozen_candidate_runtime,
)
from portfolio_rl.phase3b.governance import (
    GovernanceError,
    read_json,
    resolve_path,
    validate_container_identity,
)
from portfolio_rl.phase3b.holdout_registry import verify_holdout_registration
from portfolio_rl.phase3b.shadow_runner import (
    generate_shadow_decision,
    verify_registered_execution_config,
    write_shadow_decision,
)
from portfolio_rl.phase3b.snapshot_chain import (
    load_live_portfolio_state,
    load_point_in_time_snapshot,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate a registered, signed Phase 3B shadow recommendation."
    )
    parser.add_argument("--root", default=".")
    parser.add_argument("--registration-dir", required=True)
    parser.add_argument("--frozen-candidate", required=True)
    parser.add_argument("--execution-config", required=True)
    parser.add_argument("--snapshot-manifest", required=True)
    parser.add_argument("--live-state-manifest", required=True)
    parser.add_argument("--runtime-identity", required=True)
    parser.add_argument("--signing-key", required=True)
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    registration_dir = Path(args.registration_dir)
    registered = verify_holdout_registration(registration_dir, repository_root=root)
    registration_path = resolve_path(root, registration_dir)
    runtime_identity_path = resolve_path(root, Path(args.runtime_identity))
    runtime_identity = read_json(runtime_identity_path)
    validate_container_identity(runtime_identity, registered.git_commit)
    if runtime_identity != read_json(registration_path / "container_identity.json"):
        raise GovernanceError("runtime container identity differs from registration")
    execution_config = load_execution_config(
        Path(args.execution_config), repository_root=root
    )
    verify_registered_execution_config(
        registration_path,
        execution_config,
    )
    runtime = load_frozen_candidate_runtime(
        repository_root=root,
        frozen_candidate_path=Path(args.frozen_candidate),
        device="cpu",
    )
    if runtime.candidate_manifest_sha256 != registered.candidate_manifest_sha256:
        raise GovernanceError("runtime candidate differs from registered candidate")
    snapshot = load_point_in_time_snapshot(
        manifest_path=Path(args.snapshot_manifest),
        repository_root=root,
        config=execution_config,
        expected_asset_order=runtime.candidate.asset_order,
        expected_feature_version=runtime.candidate.feature_version,
        expected_feature_spec_sha256=runtime.candidate.feature_spec_hash,
    )
    live_state = load_live_portfolio_state(
        manifest_path=Path(args.live_state_manifest),
        repository_root=root,
        config=execution_config,
        expected_asset_order=runtime.candidate.asset_order,
        decision_date=snapshot.decision_date,
    )
    decision = generate_shadow_decision(
        runtime=runtime,
        snapshot=snapshot,
        live_state=live_state,
        generated_at=datetime.now(UTC),
        execution_config=execution_config,
    )
    output = write_shadow_decision(
        decision=decision,
        runtime=runtime,
        execution_config=execution_config,
        registration_dir=registration_dir,
        runtime_identity_path=runtime_identity_path,
        signing_key_path=Path(args.signing_key),
        repository_root=root,
    )
    print(f"shadow_decision: {output}")
    print("performance_computed: false")
    print("test_accessed: false")


if __name__ == "__main__":
    main()

"""Record one evidence-bound Phase 3B operational-certification cycle."""

from __future__ import annotations

import argparse
from pathlib import Path

from portfolio_rl.phase3b.certification import (
    verify_certification_authorization,
)
from portfolio_rl.phase3b.certification_runner import write_cycle_manifest
from portfolio_rl.phase3b.execution import load_execution_config
from portfolio_rl.phase3b.governance import GovernanceError, read_json
from portfolio_rl.phase3b.operational_metrics import load_operations_config


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--certification-id", required=True)
    parser.add_argument("--cycle-number", required=True, type=int)
    parser.add_argument("--decision-date", required=True)
    parser.add_argument("--execution-date", required=True)
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--official", action="store_true")
    parser.add_argument("--authorization")
    parser.add_argument("--execution-config")
    parser.add_argument("--operations-config")
    parser.add_argument(
        "--approver-public-key",
        action="append",
        default=[],
        metavar="ROLE=PATH",
    )
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    evidence = read_json(Path(args.evidence))
    identity_sha = str(evidence.get("identity_sha256"))
    if args.official:
        if not all(
            (args.authorization, args.execution_config, args.operations_config)
        ):
            raise GovernanceError(
                "official certification requires authorization and approved configs"
            )
        authorization = read_json(Path(args.authorization))
        if set(authorization) != {"authorization", "approval_records"}:
            raise GovernanceError("certification authorization envelope mismatch")
        keys = _role_paths(args.approver_public_key)
        execution = load_execution_config(
            Path(args.execution_config), repository_root=root, require_approved=True
        )
        operations = load_operations_config(
            Path(args.operations_config), repository_root=root, require_approved=True
        )
        identity = verify_certification_authorization(
            payload=authorization["authorization"],
            approval_records=authorization["approval_records"],
            approver_public_keys=keys,
            execution_config=execution,
            operations_config=operations,
        )
        if identity.identity_sha256 != identity_sha:
            raise GovernanceError("cycle evidence uses a different approved identity")
    output = write_cycle_manifest(
        path=Path(args.output),
        certification_id=args.certification_id,
        cycle_number=args.cycle_number,
        identity_sha256=identity_sha,
        decision_date=args.decision_date,
        execution_date=args.execution_date,
        artifact_bindings=evidence["artifact_bindings"],
        checks=evidence["checks"],
        official=args.official,
        scheduled_decision_missed=bool(evidence["scheduled_decision_missed"]),
    )
    print(f"certification_cycle: {output}")
    print(f"official: {str(args.official).lower()}")
    print("canonical_holdout_registered: false")


def _role_paths(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        role, separator, path = value.partition("=")
        if not separator or not role or not path or role in result:
            raise GovernanceError("approver key must use unique ROLE=PATH syntax")
        result[role] = Path(path)
    return result


if __name__ == "__main__":
    main()

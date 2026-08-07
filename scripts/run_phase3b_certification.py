"""Record one evidence-bound Phase 3B operational-certification cycle."""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from portfolio_rl.phase3b.certification_readiness import require_certification_readiness
from portfolio_rl.phase3b.certification_runner import write_cycle_manifest
from portfolio_rl.phase3b.governance import (
    GovernanceError,
    read_json,
)


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
    parser.add_argument("--identity-approval-package")
    parser.add_argument("--container-identity")
    parser.add_argument("--embedded-runtime-identity")
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    evidence = read_json(Path(args.evidence))
    identity_sha = str(evidence.get("identity_sha256"))
    if args.official:
        if not all(
            (
                args.authorization,
                args.identity_approval_package,
                args.container_identity,
                args.embedded_runtime_identity,
            )
        ):
            raise GovernanceError(
                "official certification requires finalized identity, authorization, and runtime identities"
            )
        authorization = require_certification_readiness(
            repository_root=root,
            identity_package_path=Path(args.identity_approval_package),
            authorization_package_path=Path(args.authorization),
            container_identity_path=Path(args.container_identity),
            embedded_identity_path=Path(args.embedded_runtime_identity),
            certification_id=args.certification_id,
            decision_date=date.fromisoformat(args.decision_date),
            cycle_number=args.cycle_number,
        )
        if authorization.identity.identity_sha256 != identity_sha:
            raise GovernanceError("cycle evidence uses a different approved identity")
    output_path = Path(args.output).resolve()
    if args.official:
        expected_root = (
            root / "artifacts/phase3b/certification" / args.certification_id
        ).resolve()
        try:
            output_path.relative_to(expected_root)
        except ValueError as exc:
            raise GovernanceError(
                "official certification output must be inside its immutable registry"
            ) from exc
    output = write_cycle_manifest(
        path=output_path,
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


if __name__ == "__main__":
    main()

"""Print READY or precise blockers without starting Phase 3B certification."""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from portfolio_rl.phase3b.certification_readiness import check_certification_readiness


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--identity-package", required=True)
    parser.add_argument("--certification-authorization", required=True)
    parser.add_argument("--container-identity", required=True)
    parser.add_argument("--embedded-runtime-identity", required=True)
    parser.add_argument("--certification-id", required=True)
    parser.add_argument("--decision-date")
    parser.add_argument("--cycle-number", type=int)
    args = parser.parse_args(argv)
    blockers = check_certification_readiness(
        repository_root=Path(args.root),
        identity_package_path=Path(args.identity_package),
        authorization_package_path=Path(args.certification_authorization),
        container_identity_path=Path(args.container_identity),
        embedded_identity_path=Path(args.embedded_runtime_identity),
        certification_id=args.certification_id,
        decision_date=date.fromisoformat(args.decision_date)
        if args.decision_date
        else None,
        cycle_number=args.cycle_number,
    )
    if blockers:
        print("BLOCKED")
        for reason in blockers:
            print(f"- {reason}")
        raise SystemExit(1)
    print("READY")


if __name__ == "__main__":
    main()

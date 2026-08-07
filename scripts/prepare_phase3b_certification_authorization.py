"""Prepare a separate unsigned authorization to begin official certification."""

from __future__ import annotations

import argparse
from datetime import UTC, date, datetime
from pathlib import Path

from portfolio_rl.phase3b.certification_authorization import (
    prepare_certification_authorization,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--identity-package", required=True)
    parser.add_argument("--certification-id", required=True)
    parser.add_argument("--approved-start-date", required=True)
    parser.add_argument("--approved-end-date", required=True)
    parser.add_argument(
        "--output-root", default="artifacts/phase3b/certification_authorization"
    )
    args = parser.parse_args(argv)
    output = prepare_certification_authorization(
        repository_root=Path(args.root),
        identity_package_path=Path(args.identity_package),
        certification_id=args.certification_id,
        approved_start_date=date.fromisoformat(args.approved_start_date),
        approved_end_date=date.fromisoformat(args.approved_end_date),
        output_root=Path(args.output_root),
        created_at=datetime.now(UTC),
    )
    print(f"certification_authorization: {output}")
    print("status: awaiting_three_external_signatures")
    print("official_certification_started: false")


if __name__ == "__main__":
    main()

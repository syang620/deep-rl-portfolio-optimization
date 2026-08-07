"""Finalize a unanimous Phase 3B certification authorization package."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

from portfolio_rl.phase3b.certification_authorization import (
    finalize_certification_authorization,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--package", required=True)
    args = parser.parse_args(argv)
    output = finalize_certification_authorization(
        repository_root=Path(args.root),
        package_path=Path(args.package),
        finalized_at=datetime.now(UTC),
    )
    print(f"finalized_certification_authorization: {output}")
    print("official_certification_started: false")


if __name__ == "__main__":
    main()

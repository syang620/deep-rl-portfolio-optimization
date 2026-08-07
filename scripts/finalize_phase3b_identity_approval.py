"""Finalize a unanimous Phase 3B runtime identity approval package."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

from portfolio_rl.phase3b.identity_approval import finalize_identity_approval


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--package", required=True)
    args = parser.parse_args(argv)
    output = finalize_identity_approval(
        repository_root=Path(args.root),
        package_path=Path(args.package),
        finalized_at=datetime.now(UTC),
    )
    print(f"finalized_identity_package: {output}")
    print("official_certification_started: false")
    print("canonical_holdout_registered: false")


if __name__ == "__main__":
    main()

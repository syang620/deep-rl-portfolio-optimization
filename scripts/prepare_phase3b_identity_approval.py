"""Prepare an unsigned, evidence-backed Phase 3B runtime identity challenge."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

from portfolio_rl.phase3b.identity_approval import prepare_identity_approval


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    output = prepare_identity_approval(
        repository_root=Path(args.root),
        config_path=Path(args.config),
        created_at=datetime.now(UTC),
    )
    print(f"identity_approval_package: {output}")
    print("status: awaiting_three_external_signatures")
    print("official_certification_started: false")
    print("canonical_holdout_registered: false")


if __name__ == "__main__":
    main()

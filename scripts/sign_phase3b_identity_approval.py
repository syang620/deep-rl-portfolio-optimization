"""Sign one Phase 3B identity challenge with an external approver key."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

from portfolio_rl.phase3b.identity_approval import sign_identity_approval


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", required=True)
    parser.add_argument("--role", required=True)
    parser.add_argument("--private-key", required=True)
    args = parser.parse_args(argv)
    output = sign_identity_approval(
        package_path=Path(args.package),
        role=args.role,
        private_key_path=Path(args.private_key),
        signed_at=datetime.now(UTC),
    )
    print(f"identity_approval_signature: {output}")


if __name__ == "__main__":
    main()

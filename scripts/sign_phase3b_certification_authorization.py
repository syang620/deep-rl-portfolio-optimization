"""Sign one certification authorization role using an external private key."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

from portfolio_rl.phase3b.certification_authorization import (
    sign_certification_authorization,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--package", required=True)
    parser.add_argument("--role", required=True)
    parser.add_argument("--private-key", required=True)
    args = parser.parse_args(argv)
    output = sign_certification_authorization(
        repository_root=Path(args.root),
        package_path=Path(args.package),
        role=args.role,
        private_key_path=Path(args.private_key),
        signed_at=datetime.now(UTC),
    )
    print(f"certification_authorization_signature: {output}")


if __name__ == "__main__":
    main()

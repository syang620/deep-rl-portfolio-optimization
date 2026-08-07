"""Fail closed unless repository, image, and approved runtime Git identities match."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from portfolio_rl.phase3b.runtime_identity import verify_runtime_identity


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument(
        "--embedded-identity",
        default=os.environ.get(
            "PHASE3B_RUNTIME_IDENTITY_PATH", "/opt/portfolio-rl/runtime_identity.json"
        ),
    )
    parser.add_argument("--container-identity", required=True)
    parser.add_argument("--identity-package")
    args = parser.parse_args(argv)
    sha = verify_runtime_identity(
        repository_root=Path(args.root),
        embedded_identity_path=Path(args.embedded_identity),
        container_identity_path=Path(args.container_identity),
        identity_package_path=(
            Path(args.identity_package) if args.identity_package else None
        ),
    )
    print(f"VERIFIED {sha}")


if __name__ == "__main__":
    main()

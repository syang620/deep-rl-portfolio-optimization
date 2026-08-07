"""Independently verify a finalized Phase 3B certification authorization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from portfolio_rl.phase3b.certification_authorization import (
    verify_finalized_certification_authorization,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--package", required=True)
    args = parser.parse_args(argv)
    approved = verify_finalized_certification_authorization(
        repository_root=Path(args.root), package_path=Path(args.package)
    )
    print(
        json.dumps(
            {
                "certification_id": approved.certification_id,
                "identity_sha256": approved.identity.identity_sha256,
                "verified": True,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

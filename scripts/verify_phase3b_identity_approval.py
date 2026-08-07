"""Verify a finalized Phase 3B runtime identity approval package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from portfolio_rl.phase3b.identity_approval import verify_identity_approval


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--package", required=True)
    parser.add_argument(
        "--historical",
        action="store_true",
        help="Verify the frozen package after certification has legitimately started.",
    )
    args = parser.parse_args(argv)
    identity = verify_identity_approval(
        repository_root=Path(args.root),
        package_path=Path(args.package),
        require_current_evidence=not args.historical,
    )
    print(
        json.dumps(
            {
                "approval_id": identity.approval_id,
                "pr22_merge_sha": identity.pr22_merge_sha,
                "identity_tooling_merge_sha": identity.identity_tooling_merge_sha,
                "container_digest": identity.container_digest,
                "verified": True,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

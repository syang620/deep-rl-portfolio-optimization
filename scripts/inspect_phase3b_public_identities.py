"""Inspect externally generated Phase 3B public identities without copying them."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from portfolio_rl.phase3b.public_identities import inspect_public_identities


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--service-signing-key", required=True)
    parser.add_argument("--performance-sealing-key", required=True)
    parser.add_argument("--portfolio-manager-key", required=True)
    parser.add_argument("--independent-reviewer-key", required=True)
    parser.add_argument("--data-operations-custodian-key", required=True)
    args = parser.parse_args(argv)
    result = inspect_public_identities(
        service_signing_key=Path(args.service_signing_key),
        performance_sealing_key=Path(args.performance_sealing_key),
        approver_keys={
            "portfolio_manager": Path(args.portfolio_manager_key),
            "independent_reviewer": Path(args.independent_reviewer_key),
            "data_operations_custodian": Path(args.data_operations_custodian_key),
        },
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

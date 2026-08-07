"""Verify an encrypted Phase 3B ledger without decrypting performance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from portfolio_rl.phase3b.sealed_ledger import verify_sealed_ledger


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger-root", required=True)
    parser.add_argument("--service-public-key", required=True)
    parser.add_argument("--custodian-public-key")
    args = parser.parse_args(argv)
    result = verify_sealed_ledger(
        ledger_root=Path(args.ledger_root),
        service_public_key_path=Path(args.service_public_key),
        custodian_public_key_path=(
            Path(args.custodian_public_key) if args.custodian_public_key else None
        ),
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()

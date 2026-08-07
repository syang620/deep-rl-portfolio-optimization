"""Display only allowlisted Phase 3B operational output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from portfolio_rl.phase3b.governance import read_json
from portfolio_rl.phase3b.operational_metrics import (
    assert_operationally_safe,
    load_operations_config,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--operations-config", required=True)
    parser.add_argument("--operational-artifact", required=True)
    parser.add_argument("--allow-draft", action="store_true")
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    config = load_operations_config(
        Path(args.operations_config),
        repository_root=root,
        require_approved=not args.allow_draft,
    )
    payload = read_json(Path(args.operational_artifact))
    assert_operationally_safe(payload, config.forbidden_field_tokens)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

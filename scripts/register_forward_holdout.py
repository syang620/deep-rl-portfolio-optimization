"""Prepare, register, or verify an independent Phase 3B forward holdout."""

from __future__ import annotations

import argparse
from pathlib import Path

from portfolio_rl.phase3b.holdout_registry import (
    prepare_registration_challenge,
    register_forward_holdout,
    verify_holdout_registration,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Govern Phase 3B registration without running the holdout."
    )
    parser.add_argument("--root", default=".")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--config", required=True)
    prepare.add_argument("--challenge-output", required=True)

    register = subparsers.add_parser("register")
    register.add_argument("--challenge", required=True)
    register.add_argument("--approvals-dir", required=True)
    register.add_argument("--output-root")

    verify = subparsers.add_parser("verify")
    verify.add_argument("--registration-dir", required=True)

    args = parser.parse_args(argv)
    root = Path(args.root)
    if args.command == "prepare":
        output = prepare_registration_challenge(
            repository_root=root,
            config_path=Path(args.config),
            challenge_output_path=Path(args.challenge_output),
        )
        print(f"registration_challenge: {output}")
        print("holdout_registered: false")
        print("test_accessed: false")
        return
    if args.command == "register":
        output = register_forward_holdout(
            repository_root=root,
            challenge_path=Path(args.challenge),
            approvals_dir=Path(args.approvals_dir),
            output_root=Path(args.output_root) if args.output_root else None,
        )
        print(f"holdout_registration: {output}")
        print("test_accessed: false")
        print("performance_data_accessed: false")
        return
    registered = verify_holdout_registration(
        Path(args.registration_dir), repository_root=root
    )
    print(f"verified_holdout_id: {registered.holdout_id}")
    print("test_accessed: false")
    print("performance_data_accessed: false")


if __name__ == "__main__":
    main()

"""Create or verify the immutable PR 19 pre-test candidate freeze."""

from __future__ import annotations

import argparse
from pathlib import Path

from portfolio_rl.training.pretest_freeze import (
    freeze_final_candidate,
    verify_frozen_candidate,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Freeze one final candidate without accessing a final holdout."
    )
    parser.add_argument("--root", default=".")
    parser.add_argument("--verify", help="Verify an existing frozen_candidate.json.")
    parser.add_argument("--research-freeze")
    parser.add_argument("--ensemble-manifest")
    parser.add_argument("--walk-forward-results")
    parser.add_argument("--execution-stress")
    parser.add_argument("--regime-attribution")
    parser.add_argument(
        "--acceptance-config",
        default="configs/final_candidate_acceptance.yaml",
    )
    parser.add_argument("--model-version")
    parser.add_argument("--output-root", default="artifacts/pretest_freeze")
    args = parser.parse_args(argv)

    if args.verify:
        candidate = verify_frozen_candidate(Path(args.verify))
        print(f"verified_model_version: {candidate.model_version}")
        print("test_accessed: false")
        print("phase3b_authorized: false")
        return

    required = {
        "research_freeze": args.research_freeze,
        "ensemble_manifest": args.ensemble_manifest,
        "walk_forward_results": args.walk_forward_results,
        "execution_stress": args.execution_stress,
        "regime_attribution": args.regime_attribution,
        "model_version": args.model_version,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        parser.error("missing required arguments: " + ", ".join(missing))

    output = freeze_final_candidate(
        repository_root=Path(args.root),
        research_freeze_path=Path(args.research_freeze),
        ensemble_manifest_path=Path(args.ensemble_manifest),
        walk_forward_results_path=Path(args.walk_forward_results),
        execution_stress_path=Path(args.execution_stress),
        regime_attribution_path=Path(args.regime_attribution),
        acceptance_config_path=Path(args.acceptance_config),
        model_version=str(args.model_version),
        output_root=Path(args.output_root),
    )
    print(f"pretest_freeze: {output}")
    print("final_test_status: not_run_blocked_pending_new_independent_holdout")
    print("test_accessed: false")
    print("phase3b_authorized: false")


if __name__ == "__main__":
    main()

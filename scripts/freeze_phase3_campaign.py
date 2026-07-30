"""CLI entrypoint for freezing the Phase 3 turnover-v2 research campaign."""

from __future__ import annotations

import argparse
from pathlib import Path

from portfolio_rl.training.provenance import (
    ProvenanceAuditError,
    freeze_phase3_campaign,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Audit and freeze the five-seed Phase 3 research campaign.",
    )
    parser.add_argument(
        "--config",
        default="configs/research/phase3_candidate_qualification.yaml",
        help="Path to the tracked campaign qualification config.",
    )
    parser.add_argument(
        "--output-root",
        default="artifacts/research_freeze",
        help="Parent directory for immutable research-freeze artifacts.",
    )
    args = parser.parse_args(argv)

    try:
        result = freeze_phase3_campaign(
            config_path=Path(args.config),
            output_root=Path(args.output_root),
        )
    except ProvenanceAuditError as exc:
        parser.exit(1, f"research freeze failed: {exc}\n")

    print(f"research_freeze: {result.output_dir}")
    print(f"campaign_id: {result.campaign_id}")
    print(f"campaign_test_free: {str(result.campaign_test_free).lower()}")
    print(f"phase3b_authorized: {str(result.phase3b_authorized).lower()}")


if __name__ == "__main__":
    main()

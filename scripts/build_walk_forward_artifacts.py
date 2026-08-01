"""Build deterministic PR 16 walk-forward data artifacts."""

from __future__ import annotations

import argparse

from portfolio_rl.features.fold_pipeline import build_walk_forward_artifacts


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Build leakage-safe walk-forward fold artifacts.",
    )
    parser.add_argument("--config", default="configs/walk_forward.yaml")
    parser.add_argument("--root", default=".")
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args(argv)
    result = build_walk_forward_artifacts(
        config_path=args.config,
        root=args.root,
        output_root=args.output_root,
    )
    print(f"campaign_manifest: {result.campaign_manifest}")
    for fold_id, fold_dir in result.fold_directories.items():
        print(f"{fold_id}: {fold_dir}")


if __name__ == "__main__":
    main()

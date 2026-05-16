"""CLI entrypoint for the Phase 2 PortfolioEnv dummy-agent smoke test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from portfolio_rl.config.loader import load_env_config
from portfolio_rl.data.dataset import load_portfolio_dataset
from portfolio_rl.data.feature_store import PortfolioFeatureStore
from portfolio_rl.env.episode_sampler import RandomWindowEpisodeSampler
from portfolio_rl.env.portfolio_env import PortfolioEnv
from portfolio_rl.env.smoke import run_dummy_random_agent_episode


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run a random-agent smoke episode against PortfolioEnv.",
    )
    parser.add_argument(
        "--root",
        default=".",
        help="Repository root containing Phase 1 data/ and artifacts/ outputs.",
    )
    parser.add_argument(
        "--env-config",
        default="configs/env.yaml",
        help="Path to the Phase 2 environment config.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    root = Path(args.root)
    try:
        dataset = load_portfolio_dataset(root)
    except FileNotFoundError as exc:
        raise SystemExit(
            "Missing Phase 1 artifacts. Run Phase 1 first or pass --root to a "
            f"repo/artifact root containing data/ and artifacts/: {exc}"
        ) from exc

    env = PortfolioEnv(
        feature_store=PortfolioFeatureStore(dataset, split="train"),
        env_config=load_env_config(Path(args.env_config)),
        episode_sampler=RandomWindowEpisodeSampler(),
        seed=args.seed,
    )
    result = run_dummy_random_agent_episode(env, seed=args.seed)
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()

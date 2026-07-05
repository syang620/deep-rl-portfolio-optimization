# Deep RL Portfolio Optimization

Research codebase for dynamic portfolio allocation with reinforcement learning.
The project builds a reproducible pipeline from market data ingestion through
feature engineering, Gymnasium environment simulation, PPO training, and
baseline comparison.

The current focus is Phase 2: a tested RL research environment where learned
policies are evaluated against deterministic portfolio baselines under the same
transaction-cost and weight-drift mechanics.

## Architecture

```text
configs/                  YAML configuration for data, features, env, and PPO
scripts/                  Thin CLI entrypoints for pipeline and experiments
src/portfolio_rl/data/    ETL, storage, split assignment, feature-store access
src/portfolio_rl/features/ Feature engineering, normalization, model matrix
src/portfolio_rl/env/     Gymnasium portfolio environment and mechanics
src/portfolio_rl/policies/ Baseline policies and SB3 policy adapter
src/portfolio_rl/evaluation/ Backtests, metrics, and validation reports
src/portfolio_rl/training/ PPO training and validation checkpoint callbacks
tests/                    Unit and integration coverage
docs/                     Planning documents and technical references
```

For a fuller technical explanation, see
[`docs/technical_overview.md`](docs/technical_overview.md).

## Core Workflow

```bash
python scripts/run_etl.py
python scripts/build_features.py
python scripts/validate_phase1.py
python scripts/smoke_test_env.py
python scripts/run_baselines.py
python scripts/train_ppo.py
python scripts/evaluate_policy.py --model-path artifacts/experiments/<run_id>/model.zip
python scripts/compare_validation.py
```

Generated data and experiment outputs are written under `data/` and
`artifacts/`, which are intentionally git-ignored.

## Local Setup

```bash
conda create -n drl-portfolio-opt python=3.11
conda activate drl-portfolio-opt
python -m pip install -e ".[dev]"
```

Run tests:

```bash
python -m pytest
```

## Current Splits

The default data split contract is defined in `configs/data.yaml`:

```text
train:       2010-01-01 through 2023-12-31
validation: 2024-01-01 through 2024-12-31
test:       2025-01-01 onward
```

PPO trains on random one-year windows from the train split and is evaluated
deterministically on validation before final test evaluation.

## Documentation

- [`docs/technical_overview.md`](docs/technical_overview.md): project goal,
  architecture, data flow, and experiment design.
- [`docs/planning/phase_1.md`](docs/planning/phase_1.md): Phase 1 data pipeline
  plan.
- [`docs/planning/phase_2.md`](docs/planning/phase_2.md): active Phase 2 RL
  environment, training, and evaluation blueprint.

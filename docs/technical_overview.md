# Technical Overview

## Goal

This project builds an institutional-style research platform for dynamic
portfolio allocation with reinforcement learning. The objective is not only to
train a PPO model, but to make policy learning comparable against deterministic
baselines through one shared simulation and evaluation stack.

The system emphasizes:

- strict data timing to avoid lookahead leakage,
- train-only feature normalization,
- explicit portfolio mechanics such as transaction costs and weight drift,
- reproducible experiment artifacts,
- common backtesting for learned and rule-based policies.

## Data And Split Contract

Raw market and macro data are transformed into model-ready daily features. The
Phase 2 environment consumes those artifacts through a split-aware feature store.

Default split dates:

```text
train:       2010-01-01 through 2023-12-31
validation: 2024-01-01 through 2024-12-31
test:       2025-01-01 through latest ingested trading date
```

For a decision at row `i`, the agent observes information known at the close of
row `i`, chooses portfolio weights, and receives reward from future return rows
`i+1:i+6` for weekly rebalancing. Same-row returns are never used for the
decision reward.

## System Architecture

The project is organized around thin scripts and testable package modules:

```text
configs/
  data.yaml              data ranges, sources, and storage paths
  env.yaml               portfolio environment mechanics
  features.yaml          feature engineering and normalization settings
  train_ppo.yaml         PPO training and evaluation settings

scripts/
  run_etl.py             fetch and persist raw/interim/processed data
  build_features.py      build normalized features and model matrix
  validate_phase1.py     validate Phase 1 artifacts
  smoke_test_env.py      run one random-agent environment episode
  run_baselines.py       run deterministic baseline backtests
  train_ppo.py           train PPO and write experiment artifacts
  evaluate_policy.py     backtest a saved PPO model
  compare_validation.py  compare baselines and PPO validation metrics

src/portfolio_rl/
  data/                  ETL, storage, splits, and feature-store APIs
  features/              returns, technicals, macro joins, normalization
  env/                   Gymnasium environment and portfolio mechanics
  policies/              deterministic baselines and SB3 policy wrapper
  evaluation/            backtest engine, metrics, reports
  training/              PPO harness and validation checkpoint callback
```

## Portfolio Environment

The Gymnasium environment models weekly rebalancing:

- action space is a normalized vector in `[-1, 1]`,
- actions are converted to long-only weights with temperature-scaled softmax,
- weights drift during the five-day holding period,
- turnover is measured against drifted pre-trade weights,
- transaction costs reduce both NAV and the learning reward,
- observations append live current weights rather than static Phase 1 weights.

Current important defaults in `configs/env.yaml`:

```yaml
rebalance_frequency_trading_days: 5
episode_length_trading_days: 260
max_episode_steps: 52
action_transform: softmax
action_temperature: 0.5
transaction_cost_bps: 10.0
reward_type: log_growth
reward_scale: 100.0
```

`action_temperature` controls how concentrated the softmax weights become.
Higher values create sharper allocations and can increase turnover. The current
default is `0.5` after validation diagnostics showed that higher temperatures
caused excessive turnover and transaction-cost drag.

## Training And Evaluation

PPO training uses Stable-Baselines3. The default full budget is
`total_timesteps: 500000`, where one timestep is one weekly rebalance decision.
Short pilot runs can override this from the CLI, for example `25000` timesteps.

Training samples random one-year windows from the train split. Validation is
deterministic on the validation split. During training, a validation callback
periodically backtests the in-memory model and saves:

- `best_model.zip`,
- `best_metrics_validation.json`,
- final `model.zip`,
- final validation NAV, weights, trades, costs, and metrics.

The best checkpoint is selected by `evaluation.metric_for_best_model`, currently
`sharpe_ratio`.

## Baselines And Reports

Deterministic baselines use the same backtest mechanics as PPO evaluation. The
current baseline set includes equal-weight weekly, buy-and-hold equal-weight,
SPY-only, SHY-only, and inverse-volatility policies.

Backtest outputs include:

- daily NAV,
- target weights,
- drifted weights,
- trades,
- turnover and transaction costs,
- summary metrics such as total return, CAGR, volatility, Sharpe, drawdown, and
  transaction-cost drag.

Validation comparison reports highlight best strategies and warn when PPO
underperforms equal-weight on return or Sharpe.

## Reproducibility

Experiment bundles under `artifacts/experiments/` include copied configs,
feature specs, data quality reports when available, validation outputs, and a
manifest with hashes. These artifacts are generated outputs and are not tracked
by Git.

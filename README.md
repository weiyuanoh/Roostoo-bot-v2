# Roostoo Bot v2

Research and live-trading environment for a long-only crypto spot bot that:

- executes through the Roostoo mock exchange API
- collects live market data from Binance public spot klines
- centralizes feature, scoring, risk, and execution infrastructure so notebooks, backtests, and live trading do not drift

## Current Deployment Baseline

Research is paused for deployment. The current deployment baseline is the
fixed top-30 USD universe with the `momentum_only` ridge score and TP/SL exits
only. Rank-decay exits and Roll-impact regime throttles did not beat the plain
TP/SL baseline after costs, so they should remain disabled unless a future
research pass shows a clear out-of-sample improvement.

```text
universe = fixed_top30_median_dollar_volume
model = momentum_only
score = beta_momentum * z_momentum
horizon = 24
top_k = 1
max_new_entries = 1
max_positions = 3
position_fraction = 0.25
take_profit = 0.03
stop_loss = 0.015
exits = TP/SL only
```

The fixed deployment universe is:

```text
BTC/USD, ETH/USD, SOL/USD, XRP/USD, ZEC/USD, BNB/USD, DOGE/USD, SUI/USD,
TRX/USD, ADA/USD, PEPE/USD, PAXG/USD, LINK/USD, TAO/USD, AVAX/USD,
NEAR/USD, LTC/USD, ENA/USD, UNI/USD, WLD/USD, AAVE/USD, HBAR/USD,
FET/USD, FIL/USD, TRUMP/USD, TON/USD, DOT/USD, ICP/USD, APT/USD,
VIRTUAL/USD
```

Latest relevant backtest artifacts:

- `reports/backtests/rank_decay_exit_top3_top30.md`
- `reports/backtests/rank_decay_exit_top3_top30_baseline_slip0_summary.csv`
- `reports/backtests/rank_decay_exit_top3_top30_baseline_slip5_summary.csv`
- `reports/backtests/rank_decay_exit_top3_top30_candidate_slip0_summary.csv`
- `reports/backtests/rank_decay_exit_top3_top30_candidate_slip5_summary.csv`

Result summary:

```text
TP/SL baseline, 0 bps slippage:   +15.18%, rank_decay_exits=0
rank-decay candidate, 0 bps:      -12.59%, rank_decay_exits=414
TP/SL baseline, 5 bps slippage:    +0.14%, rank_decay_exits=0
rank-decay candidate, 5 bps:      -30.07%, rank_decay_exits=414
```

## Research Log

The original hypothesis was:

> Market microstructure statistics may directly predict cross-sectional rankings of forward crypto returns.

The evidence from initial 1h-candle IC checks was weak. Microstructure measures, including Roll measure, Roll impact, VPIN-like measures, and related liquidity proxies, did not show strong standalone predictive power for 1h, 6h, or 24h forward returns.

A live-style portfolio backtest on the 8-pair live universe also failed to validate microstructure as a linear alpha overlay. With 4 months IS / 4 months OS, 1h candles, 24h forward-return training target, `top_k=1`, `max_new_entries=1`, `max_positions=3`, `position_fraction=0.25`, `tp=0.03`, and `sl=0.015`, `momentum_only` returned `+4.94%` OS while `momentum_plus_roll_plus_interaction` returned `-4.04%` OS. The combined model improved in-sample IC, but did not improve traded OS performance.

The working hypothesis has therefore evolved:

> Recent momentum is the primary alpha. Microstructure, especially Roll impact / low-liquidity stress, may be useful as a regime, risk, or exposure modifier rather than as a direct alpha.

The research baseline became:

```text
baseline = momentum_only
score = beta_momentum * z_momentum
```

Any microstructure layer must improve this baseline out-of-sample after fees and realistic slippage, preferably by reducing drawdown, stop-hit rate, bad pair exposure, or left-tail losses without destroying return.

The live strategy can run either:

- `momentum_only`
- `momentum_plus_roll`
- `momentum_roll_interaction`
- `momentum_plus_roll_plus_interaction`

The deployment command should explicitly use:

```text
momentum_only
```

The historical default in code may still be configured differently by
environment variables. Do not rely on implicit defaults for deployment.

The combined score uses:

```text
z_momentum
z_low_roll_impact
z_momentum_x_low_roll_impact
```

## What Has Been Built

### Data

- Binance public kline client in `bot/binance_data.py`
- Roostoo pair to Binance symbol mapping in `bot/config.py`
- Optional CSV candle persistence for backtesting / research only
- Live trading fetches Binance data directly

### Microstructure And Features

- Microstructure measures in `bot/microstructure.py`
- Alpha features in `bot/features.py`
- Forward return and IC helpers in `bot/forward_ic.py`
- Shared ridge score implementation in `bot/strategy/ridge.py`

### Research Notebooks

Research artifacts are grouped under:

```text
notebooks/feature/
notebooks/microstructure/
```

Existing checks include:

- microstructure IC checks on 1h candles
- feature IC checks on 1h candles
- momentum + Roll impact risk-filter checks
- ridge score walk-forward checks
- portfolio backtest outputs

### Live Trading

Live trading is implemented in:

```text
bot/live_trader.py
bot/main.py
bot/live_state.py
bot/executor.py
```

Live behavior:

- dry-run by default
- `--execute` required for real Roostoo orders
- waits until next UTC hour + delay in continuous mode
- uses market orders
- long-only spot
- local state file stores entry price, entry score, quantity, and order id
- exits happen only through TP/SL unless manual liquidation is requested

Local live state:

```text
data/live_state.json
```

This file is intentionally ignored by git.

### Backtest Infrastructure

The backtest was refactored to better mimic live trading:

- shared cycle intent builder in `bot/strategy/ridge.py`
- Roostoo-like simulated executor in `bot/backtest/simulated_executor.py`
- portfolio accounting in `bot/backtest/portfolio.py`
- portfolio backtest in `bot/backtest/ridge_score_portfolio.py`

Backtest validity improvements:

- live and backtest use the same entry / exit intent logic
- supports asymmetric `--tp` and `--sl`
- supports `--top-k`, `--max-new-entries`, and `--max-positions`
- simulated executor applies amount precision
- simulated executor enforces minimum order notional
- simulated executor supports configurable slippage
- exits and entries are decided before portfolio mutation, matching live behavior

Known remaining mismatch:

- backtest fills against candle prices
- live fills against Roostoo market liquidity at runtime
- spread and realized market-order slippage are approximated only with `--slippage-bps`

## Setup

```bash
cp .env.example .env
poetry install
```

Fill in `.env`:

```text
ROOSTOO_API_KEY=...
ROOSTOO_API_SECRET=...
ROOSTOO_BASE_URL=https://mock-api.roostoo.com
```

Do not commit `.env`.

## Basic Commands

Check public Roostoo and Binance connectivity:

```bash
.venv/bin/python -m bot.main smoke
```

Check signed Roostoo balance:

```bash
.venv/bin/python -m bot.main balance
```

Collect candles to local CSV for backtesting:

```bash
.venv/bin/python -m bot.main collect \
  --pairs BTC/USD,ETH/USD,SOL/USD \
  --interval 1h \
  --limit 1000
```

## Live Trading Commands

Deployment pair universe:

```bash
PAIRS=BTC/USD,ETH/USD,SOL/USD,XRP/USD,ZEC/USD,BNB/USD,DOGE/USD,SUI/USD,TRX/USD,ADA/USD,PEPE/USD,PAXG/USD,LINK/USD,TAO/USD,AVAX/USD,NEAR/USD,LTC/USD,ENA/USD,UNI/USD,WLD/USD,AAVE/USD,HBAR/USD,FET/USD,FIL/USD,TRUMP/USD,TON/USD,DOT/USD,ICP/USD,APT/USD,VIRTUAL/USD
```

Dry-run one live cycle:

```bash
.venv/bin/python -m bot.main live-once \
  --pairs "$PAIRS" \
  --model momentum_only \
  --top-k 1 \
  --max-new-entries 1 \
  --max-positions 3 \
  --position-fraction 0.25 \
  --tp 0.03 \
  --sl 0.015
```

Execute one live cycle:

```bash
.venv/bin/python -m bot.main live-once \
  --pairs "$PAIRS" \
  --model momentum_only \
  --top-k 1 \
  --max-new-entries 1 \
  --max-positions 3 \
  --position-fraction 0.25 \
  --tp 0.03 \
  --sl 0.015 \
  --execute
```

Run continuously on the hour:

```bash
.venv/bin/python -m bot.main live \
  --pairs "$PAIRS" \
  --model momentum_only \
  --top-k 1 \
  --max-new-entries 1 \
  --max-positions 3 \
  --position-fraction 0.25 \
  --tp 0.03 \
  --sl 0.015 \
  --execute
```

Meaning of the main risk knobs:

```text
--top-k 1             only the highest-ranked candidate is eligible
--max-new-entries 1   open at most one new position per cycle
--max-positions 3     hold at most three total positions
--position-fraction   fraction of portfolio value allocated per new trade
--tp                  take-profit return threshold
--sl                  stop-loss return threshold
```

Example risk math:

```text
position_fraction = 0.25
sl = 0.015
portfolio loss at stop ~= 0.25 * 0.015 = 0.375% per stopped position
```

Fees, slippage, and hourly stop checks can make realized loss worse.

## Liquidation

Dry-run liquidation:

```bash
.venv/bin/python -m bot.main liquidate \
  --pairs "$PAIRS"
```

Execute liquidation:

```bash
.venv/bin/python -m bot.main liquidate \
  --pairs "$PAIRS" \
  --cancel-pending \
  --execute
```

Liquidation sells only free non-USD balances. Locked balances are reserved by pending orders unless pending orders are cancelled first.

## Backtesting

Run the current deployment baseline backtest:

```bash
.venv/bin/python -m bot.backtest.rank_decay_exit_experiment
```

This writes the fixed top-30 TP/SL baseline plus the rejected rank-decay
candidate for comparison.

Run the generic live-like portfolio backtest manually:

```bash
.venv/bin/python -m bot.backtest.ridge_score_portfolio \
  --pairs "$PAIRS" \
  --model momentum_only \
  --top-k 1 \
  --max-new-entries 1 \
  --max-positions 3 \
  --position-fraction 0.25 \
  --tp 0.03 \
  --sl 0.015 \
  --fee-bps 10 \
  --slippage-bps 5
```

## Operational Notes

For live testing, use `tmux` so the bot keeps running if the terminal disconnects:

```bash
tmux new -s roostoo-bot
```

Monitor logs:

```bash
tail -f logs/live_cycles.jsonl
tail -f logs/trades.jsonl
```

Check wallet:

```bash
.venv/bin/python -m bot.main balance
```

Check local state:

```bash
cat data/live_state.json
```

## Tests

Run the test suite:

```bash
.venv/bin/python -m pytest -q
.venv/bin/ruff check .
```

Current coverage includes:

- Binance client behavior
- Roostoo signing helpers
- microstructure measures
- feature engineering
- forward return / IC helpers
- live state reconciliation
- scheduler behavior
- shared strategy intent generation
- simulated executor behavior
- portfolio backtest behavior
- liquidation planning

## Next Work

Research is paused. Deployment and operations are now the priority:

1. Run the deployment baseline in dry-run mode and inspect `logs/live_cycles.jsonl`.
2. Confirm Roostoo balances, minimum order handling, and local `data/live_state.json`.
3. Execute one live cycle, then verify orders and local state reconciliation.
4. Move production operation from raw terminal / tmux to `systemd` or another process manager.
5. Add a `positions` command that merges Roostoo wallet data with local entry metadata.
6. Add live-vs-backtest monitoring for fills, slippage, stop hits, and pair attribution.
7. Resume research only after live operation is stable.

The core principle going forward is:

> Backtest and live trading should differ only by data source and executor implementation.

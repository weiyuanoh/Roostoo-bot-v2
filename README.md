# Roostoo Bot v2

Research and live-trading environment for a long-only crypto spot bot that:

- executes through the Roostoo mock exchange API
- collects live market data from Binance public spot klines
- centralizes feature, scoring, risk, and execution infrastructure so notebooks, backtests, and live trading do not drift

## Current Hypothesis

The original hypothesis was:

> Market microstructure statistics may directly predict cross-sectional rankings of forward crypto returns.

The evidence from initial 1h-candle IC checks was weak. Microstructure measures, including Roll measure, Roll impact, VPIN-like measures, and related liquidity proxies, did not show strong standalone predictive power for 1h, 6h, or 24h forward returns.

The working hypothesis has therefore evolved:

> Recent momentum is the primary alpha. Microstructure, especially Roll impact / low-liquidity stress, may be useful as a regime, risk, or exposure modifier rather than as a direct alpha.

The current live strategy can run either:

- `momentum_only`
- `momentum_plus_roll`
- `momentum_roll_interaction`
- `momentum_plus_roll_plus_interaction`

The default is currently:

```text
momentum_plus_roll_plus_interaction
```

This score uses:

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

Dry-run one live cycle:

```bash
.venv/bin/python -m bot.main live-once \
  --pairs BTC/USD,ETH/USD,SOL/USD,BNB/USD,XRP/USD \
  --model momentum_only \
  --top-k 1 \
  --max-new-entries 1 \
  --max-positions 1 \
  --position-fraction 0.10 \
  --tp 0.05 \
  --sl 0.025
```

Execute one live cycle:

```bash
.venv/bin/python -m bot.main live-once \
  --pairs BTC/USD,ETH/USD,SOL/USD,BNB/USD,XRP/USD \
  --model momentum_only \
  --top-k 1 \
  --max-new-entries 1 \
  --max-positions 1 \
  --position-fraction 0.10 \
  --tp 0.05 \
  --sl 0.025 \
  --execute
```

Run continuously on the hour:

```bash
.venv/bin/python -m bot.main live \
  --pairs BTC/USD,ETH/USD,SOL/USD,BNB/USD,XRP/USD \
  --model momentum_only \
  --top-k 1 \
  --max-new-entries 1 \
  --max-positions 1 \
  --position-fraction 0.10 \
  --tp 0.05 \
  --sl 0.025 \
  --execute
```

Meaning of the main risk knobs:

```text
--top-k 1             only the highest-ranked candidate is eligible
--max-new-entries 1   open at most one new position per cycle
--max-positions 1     hold at most one total position
--position-fraction   fraction of portfolio value allocated per new trade
--tp                  take-profit return threshold
--sl                  stop-loss return threshold
```

Example risk math:

```text
position_fraction = 0.10
sl = 0.025
portfolio loss at stop ~= 0.10 * 0.025 = 0.25%
```

Fees, slippage, and hourly stop checks can make realized loss worse.

## Liquidation

Dry-run liquidation:

```bash
.venv/bin/python -m bot.main liquidate \
  --pairs BTC/USD,ETH/USD,SOL/USD,BNB/USD,XRP/USD
```

Execute liquidation:

```bash
.venv/bin/python -m bot.main liquidate \
  --pairs BTC/USD,ETH/USD,SOL/USD,BNB/USD,XRP/USD \
  --cancel-pending \
  --execute
```

Liquidation sells only free non-USD balances. Locked balances are reserved by pending orders unless pending orders are cancelled first.

## Backtesting

Run the live-like portfolio backtest:

```bash
.venv/bin/python -m bot.backtest.ridge_score_portfolio \
  --model momentum_only \
  --top-k 1 \
  --max-new-entries 1 \
  --max-positions 1 \
  --position-fraction 0.10 \
  --tp 0.05 \
  --sl 0.025 \
  --fee-bps 10 \
  --slippage-bps 5
```

Compare with the combined score:

```bash
.venv/bin/python -m bot.backtest.ridge_score_portfolio \
  --model momentum_plus_roll_plus_interaction \
  --top-k 1 \
  --max-new-entries 1 \
  --max-positions 1 \
  --position-fraction 0.10 \
  --tp 0.05 \
  --sl 0.025 \
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

Priority items:

1. Run controlled live-like backtests comparing:
   - `momentum_only`
   - `momentum_plus_roll_plus_interaction`
   - momentum with Roll impact as a hard risk filter
2. Evaluate TP/SL regimes:
   - fast: `tp=0.02`, `sl=0.01`
   - balanced: `tp=0.03`, `sl=0.015`
   - slower swing: `tp=0.05`, `sl=0.025`
3. Add a formal strategy interface so future strategies can be swapped without editing live/backtest orchestration.
4. Add a `positions` command that merges Roostoo wallet data with local entry metadata.
5. Move production operation from raw terminal / tmux to `systemd` or another process manager.
6. Add periodic research reports that compare live fills and outcomes against the backtest assumptions.

The core principle going forward is:

> Backtest and live trading should differ only by data source and executor implementation.


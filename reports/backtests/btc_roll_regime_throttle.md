# Backtest: BTC roll-impact regime throttle

Run date: 2026-06-17

## Hypothesis

High BTC `roll_impact` is a liquidity-stress regime. Momentum entries opened during this regime should have worse realised trade outcomes. Blocking new entries during this regime should improve the momentum-only baseline.

## Baseline score math

```text
score[i,t] = beta_momentum * z_momentum[i,t]
```

## Regime math

```text
btc_roll_impact[t] = roll_impact[BTC/USD,t]

threshold[t] = 80th percentile of prior 720 hourly BTC roll_impact values

is_stressed[t] = btc_roll_impact[t] >= threshold[t]
```

The threshold uses only prior bars. If history is insufficient, `is_stressed = False`.

## Trading rule

```text
if is_stressed[t]:
    no new entries
else:
    trade normally
```

Existing positions still exit normally through TP/SL.

## Parameters

- Pairs: `BNB/USD, BTC/USD, ETH/USD, SOL/USD, TAO/USD, TRX/USD, XRP/USD, ZEC/USD`
- Model: `momentum_only`
- Horizon: `24`
- IS/OS: `4m/4m`
- Top K: `1`
- Max new entries: `1`
- Max positions: `3`
- Position fraction: `0.25`
- TP/SL: `0.03` / `0.015`
- Fee bps: `10.0`
- Slippage bps: `0,5`

## Comparison

| Run | Slippage | Return | Max DD | Ret/DD | Sharpe | Sortino | Win | Stop | TP | Avg stopped loss | Median hold | Trades | Stressed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 0 | 4.94% | 14.66% | 0.337 | 0.421 | 0.873 | 40.23% | 59.48% | 39.65% | -2.21% | 7.00h | 686 | 0.00% |
| candidate | 0 | -6.83% | 13.24% | -0.516 | -0.867 | -1.645 | 37.46% | 61.92% | 37.15% | -2.20% | 7.00h | 646 | 20.28% |
| baseline | 5 | -4.14% | 15.28% | -0.271 | -0.576 | -1.185 | 39.26% | 60.46% | 38.68% | -2.24% | 6.50h | 698 | 0.00% |
| candidate | 5 | -14.26% | 16.70% | -0.853 | -1.862 | -3.333 | 36.86% | 62.54% | 36.56% | -2.25% | 6.00h | 662 | 20.28% |

## Pair attribution

### baseline slippage=0 bps

| Pair | Closed | Win | Avg return | Median return | PnL |
| --- | --- | --- | --- | --- | --- |
| ZEC/USD | 111 | 44.14% | 0.54% | -1.72% | 154,536 |
| TRX/USD | 23 | 56.52% | 0.94% | 2.90% | 53,190 |
| TAO/USD | 77 | 41.56% | 0.15% | -1.71% | 21,496 |
| BTC/USD | 27 | 44.44% | 0.35% | -1.74% | 17,581 |
| SOL/USD | 21 | 38.10% | 0.15% | -1.64% | 6,865 |
| BNB/USD | 33 | 39.39% | 0.08% | -1.68% | 5,134 |
| ETH/USD | 28 | 25.00% | -0.82% | -1.85% | -58,997 |
| XRP/USD | 23 | 17.39% | -1.13% | -1.80% | -64,207 |

### candidate slippage=0 bps

| Pair | Closed | Win | Avg return | Median return | PnL |
| --- | --- | --- | --- | --- | --- |
| ZEC/USD | 99 | 41.41% | 0.39% | -1.72% | 91,434 |
| TRX/USD | 24 | 50.00% | 0.77% | 1.26% | 44,379 |
| SOL/USD | 21 | 38.10% | 0.12% | -1.65% | 4,929 |
| BTC/USD | 27 | 37.04% | -0.03% | -1.74% | -6,322 |
| TAO/USD | 73 | 38.36% | -0.05% | -1.71% | -11,714 |
| BNB/USD | 33 | 36.36% | -0.14% | -1.72% | -12,122 |
| ETH/USD | 26 | 26.92% | -0.68% | -1.75% | -42,424 |
| XRP/USD | 20 | 15.00% | -1.23% | -1.78% | -59,440 |

### baseline slippage=5 bps

| Pair | Closed | Win | Avg return | Median return | PnL |
| --- | --- | --- | --- | --- | --- |
| ZEC/USD | 111 | 43.24% | 0.42% | -1.71% | 110,221 |
| TRX/USD | 25 | 52.00% | 0.67% | 0.82% | 39,959 |
| SOL/USD | 21 | 42.86% | 0.31% | -1.72% | 14,872 |
| BTC/USD | 27 | 44.44% | 0.28% | -1.71% | 14,591 |
| BNB/USD | 35 | 37.14% | -0.08% | -1.75% | -8,383 |
| TAO/USD | 78 | 39.74% | -0.06% | -1.76% | -14,232 |
| XRP/USD | 25 | 20.00% | -0.82% | -1.81% | -49,836 |
| ETH/USD | 27 | 22.22% | -1.00% | -1.76% | -64,948 |

### candidate slippage=5 bps

| Pair | Closed | Win | Avg return | Median return | PnL |
| --- | --- | --- | --- | --- | --- |
| ZEC/USD | 100 | 40.00% | 0.23% | -1.75% | 47,431 |
| TRX/USD | 25 | 48.00% | 0.57% | -0.47% | 33,169 |
| SOL/USD | 21 | 38.10% | 0.06% | -1.73% | 2,243 |
| BTC/USD | 26 | 38.46% | -0.01% | -1.81% | -3,216 |
| BNB/USD | 33 | 36.36% | -0.17% | -1.75% | -14,361 |
| ETH/USD | 26 | 26.92% | -0.72% | -1.75% | -42,266 |
| XRP/USD | 23 | 21.74% | -0.77% | -1.82% | -42,625 |
| TAO/USD | 77 | 36.36% | -0.25% | -1.80% | -47,172 |

## Artifacts

- `reports/backtests/btc_roll_regime_throttle_baseline_slip0_summary.csv`
- `reports/backtests/btc_roll_regime_throttle_baseline_slip0_equity.csv`
- `reports/backtests/btc_roll_regime_throttle_baseline_slip0_trades.csv`
- `reports/backtests/btc_roll_regime_throttle_baseline_slip0_metadata.json`
- `reports/backtests/btc_roll_regime_throttle_candidate_slip0_summary.csv`
- `reports/backtests/btc_roll_regime_throttle_candidate_slip0_equity.csv`
- `reports/backtests/btc_roll_regime_throttle_candidate_slip0_trades.csv`
- `reports/backtests/btc_roll_regime_throttle_candidate_slip0_metadata.json`
- `reports/backtests/btc_roll_regime_throttle_baseline_slip5_summary.csv`
- `reports/backtests/btc_roll_regime_throttle_baseline_slip5_equity.csv`
- `reports/backtests/btc_roll_regime_throttle_baseline_slip5_trades.csv`
- `reports/backtests/btc_roll_regime_throttle_baseline_slip5_metadata.json`
- `reports/backtests/btc_roll_regime_throttle_candidate_slip5_summary.csv`
- `reports/backtests/btc_roll_regime_throttle_candidate_slip5_equity.csv`
- `reports/backtests/btc_roll_regime_throttle_candidate_slip5_trades.csv`
- `reports/backtests/btc_roll_regime_throttle_candidate_slip5_metadata.json`

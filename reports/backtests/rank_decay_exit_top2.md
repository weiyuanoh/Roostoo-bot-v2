# Backtest: rank-decay exit for momentum-only

Run date: 2026-06-17

## Experiment

Baseline uses the current momentum-only score with TP/SL exits. Candidate uses the same score, sizing, entries, fees, and slippage, but also exits a held pair when its current score rank is worse than the configured threshold.

```text
score[i,t] = beta_momentum * z_momentum[i,t]

if held_pair_rank[t] > 2:
    exit position
```

The rank-decay exit is evaluated before TP/SL. Same-cycle re-entry of the exiting pair is blocked by the existing held-pair entry logic.

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
- Rank exit threshold: `2`

## Comparison

| Run | Slippage | Return | Max DD | Ret/DD | Sharpe | Sortino | Win | Stop | TP | Rank exits | Median hold | Trades |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 0 | 4.94% | 14.66% | 0.337 | 0.421 | 0.873 | 40.23% | 59.48% | 39.65% | 0 | 7.00h | 686 |
| candidate | 0 | -13.79% | 16.50% | -0.836 | -2.078 | -3.976 | 38.62% | 19.67% | 16.21% | 350 | 4.00h | 1098 |
| baseline | 5 | -4.14% | 15.28% | -0.271 | -0.576 | -1.185 | 39.26% | 60.46% | 38.68% | 0 | 6.50h | 698 |
| candidate | 5 | -26.14% | 28.04% | -0.932 | -4.267 | -7.879 | 35.51% | 21.01% | 15.76% | 347 | 4.00h | 1104 |

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
| ZEC/USD | 144 | 40.28% | 0.21% | -0.76% | 64,910 |
| TAO/USD | 104 | 41.35% | 0.11% | -0.71% | 25,022 |
| SOL/USD | 38 | 34.21% | -0.02% | -0.32% | -1,683 |
| XRP/USD | 40 | 40.00% | -0.07% | -0.55% | -8,687 |
| BTC/USD | 47 | 38.30% | -0.14% | -0.31% | -14,734 |
| TRX/USD | 82 | 40.24% | -0.08% | -0.21% | -15,811 |
| BNB/USD | 54 | 37.04% | -0.23% | -0.45% | -29,547 |
| ETH/USD | 40 | 27.50% | -0.34% | -0.52% | -31,777 |

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
| ZEC/USD | 145 | 40.00% | 0.11% | -0.88% | 26,208 |
| TAO/USD | 106 | 38.68% | -0.05% | -0.91% | -8,565 |
| SOL/USD | 38 | 31.58% | -0.14% | -0.43% | -10,997 |
| XRP/USD | 39 | 33.33% | -0.17% | -0.66% | -18,824 |
| BTC/USD | 47 | 34.04% | -0.25% | -0.41% | -22,372 |
| TRX/USD | 82 | 34.15% | -0.18% | -0.31% | -32,278 |
| ETH/USD | 40 | 22.50% | -0.44% | -0.62% | -38,238 |
| BNB/USD | 55 | 34.55% | -0.32% | -0.55% | -39,553 |

## Artifacts

- `reports/backtests/rank_decay_exit_top2_baseline_slip0_summary.csv`
- `reports/backtests/rank_decay_exit_top2_baseline_slip0_equity.csv`
- `reports/backtests/rank_decay_exit_top2_baseline_slip0_trades.csv`
- `reports/backtests/rank_decay_exit_top2_baseline_slip0_metadata.json`
- `reports/backtests/rank_decay_exit_top2_candidate_slip0_summary.csv`
- `reports/backtests/rank_decay_exit_top2_candidate_slip0_equity.csv`
- `reports/backtests/rank_decay_exit_top2_candidate_slip0_trades.csv`
- `reports/backtests/rank_decay_exit_top2_candidate_slip0_metadata.json`
- `reports/backtests/rank_decay_exit_top2_baseline_slip5_summary.csv`
- `reports/backtests/rank_decay_exit_top2_baseline_slip5_equity.csv`
- `reports/backtests/rank_decay_exit_top2_baseline_slip5_trades.csv`
- `reports/backtests/rank_decay_exit_top2_baseline_slip5_metadata.json`
- `reports/backtests/rank_decay_exit_top2_candidate_slip5_summary.csv`
- `reports/backtests/rank_decay_exit_top2_candidate_slip5_equity.csv`
- `reports/backtests/rank_decay_exit_top2_candidate_slip5_trades.csv`
- `reports/backtests/rank_decay_exit_top2_candidate_slip5_metadata.json`

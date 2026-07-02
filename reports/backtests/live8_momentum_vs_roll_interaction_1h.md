# Backtest: momentum-only vs momentum + roll-impact interaction on 1h candles

Run date: 2026-06-17

This report compares the current live-style portfolio rules on the same 8-pair universe from the recent live command.

## Data and split

- Feature source: `notebooks/microstructure/momentum_roll_impact_is_os_1h_features_8m.csv`
- Candle frequency: 1h
- Feature rows after pair filter: 46,664
- Data range: 2025-10-08 17:00 UTC to 2026-06-08 17:00 UTC
- IS window: 4 months
- OS window: 4 months
- Step: 1 month
- Available fold count for this data and split: 1
- IS: 2025-10-08 17:00 UTC to 2026-02-08 17:00 UTC
- OS: 2026-02-08 17:00 UTC to 2026-06-08 17:00 UTC
- Forward-return training target: 24 bars, i.e. 24 hours

## Pair universe

`BTC/USD`, `ETH/USD`, `SOL/USD`, `BNB/USD`, `XRP/USD`, `TRX/USD`, `ZEC/USD`, `TAO/USD`

## Portfolio rules

- Initial cash: 1,000,000 USD
- Long-only spot
- Rebalance/check frequency: each 1h bar
- Ranking rule: trade only top 1 score per cycle
- Max new entries per cycle: 1
- Max open positions: 3
- Position size: 25% of current portfolio value per new entry
- Take profit: 3.0% trade return
- Stop loss: 1.5% trade return
- Fees: 10 bps
- Slippage: 0 bps
- Exits are evaluated before entries, but same-cycle entry capacity uses pre-exit holdings to mimic the current live intent construction.

## Score definitions

At each 1h timestamp, the model ranks the pair universe cross-sectionally.

For pair \(i\) at hour \(t\):

```text
r_1h[i,t] = log(close[i,t] / close[i,t-1])

mom_24[i,t] = log(close[i,t] / close[i,t-24])

vol_24[i,t] = rolling_std_24(r_1h[i,t])

vol_adj_momentum_24[i,t] = mom_24[i,t] / vol_24[i,t]
```

Then the raw feature is cross-sectionally z-scored across the 8-pair universe at the same timestamp:

```text
z_momentum[i,t] =
    (vol_adj_momentum_24[i,t] - mean_j(vol_adj_momentum_24[j,t]))
    / std_j(vol_adj_momentum_24[j,t])
```

For the microstructure overlay:

```text
raw_low_roll_impact[i,t] = -roll_impact[i,t]

z_low_roll_impact[i,t] =
    (raw_low_roll_impact[i,t] - mean_j(raw_low_roll_impact[j,t]))
    / std_j(raw_low_roll_impact[j,t])

raw_interaction[i,t] = z_momentum[i,t] * z_low_roll_impact[i,t]

z_momentum_x_low_roll_impact[i,t] =
    (raw_interaction[i,t] - mean_j(raw_interaction[j,t]))
    / std_j(raw_interaction[j,t])
```

The training target is the cross-sectional z-score of the 24h forward return:

```text
forward_return_24[i,t] = close[i,t+24] / close[i,t] - 1

target_z[i,t] =
    (forward_return_24[i,t] - mean_j(forward_return_24[j,t]))
    / std_j(forward_return_24[j,t])
```

Ridge beta is fitted on the IS window against `target_z`. For a feature matrix `X`, target vector `y`, and ridge penalty `alpha`, the fitted coefficients are:

```text
beta = inverse(X'X + alpha * I) X'y
```

For `momentum_only`, `X` has only one column, `z_momentum`, so the ridge beta is just one scalar. The score is therefore:

```text
score[i,t] = beta_momentum * z_momentum[i,t]
```

For `momentum_plus_roll_plus_interaction`, `X` has three columns:

```text
score[i,t] =
    beta_momentum * z_momentum[i,t]
    + beta_low_roll * z_low_roll_impact[i,t]
    + beta_interaction * z_momentum_x_low_roll_impact[i,t]
```

The alpha is selected from `0.1, 1.0, 10.0, 100.0` by in-sample mean Spearman IC, with hit-rate as the secondary criterion.

## Model formulas selected

| Model | Alpha | Score |
|---|---:|---|
| `momentum_only` | 0.1 | `0.0689930359086 * z_momentum` |
| `momentum_plus_roll_plus_interaction` | 100.0 | `0.0668847817356 * z_momentum - 0.0255792904001 * z_low_roll_impact + 0.0122314458089 * z_momentum_x_low_roll_impact` |

The combined model slightly improved in-sample IC, but its roll-impact coefficient is negative. Under the current feature convention, this means it penalized lower roll impact and favored higher roll impact after controlling for momentum and interaction.

## Portfolio results

| Model | OS return | Max drawdown | Sharpe | Sortino | Trades | Closed trades | Win rate | IS mean Spearman | IS Spearman hit rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `momentum_only` | 4.94% | 14.66% | 0.421 | 0.873 | 686 | 343 | 40.23% | 0.01955 | 50.65% |
| `momentum_plus_roll_plus_interaction` | -4.04% | 13.06% | -0.629 | -1.044 | 618 | 309 | 38.83% | 0.02564 | 51.29% |

## Exit mix

| Model | Stop exits | TP exits | Fold-end exits |
|---|---:|---:|---:|
| `momentum_only` | 204 | 136 | 3 |
| `momentum_plus_roll_plus_interaction` | 188 | 118 | 3 |

## Holding time, excluding fold-end exits

| Model | Mean hours | Median hours | 25% | 75% | 90% | Max |
|---|---:|---:|---:|---:|---:|---:|
| `momentum_only` | 18.36 | 7.00 | 3.00 | 15.00 | 45.00 | 424.00 |
| `momentum_plus_roll_plus_interaction` | 21.10 | 7.00 | 3.00 | 17.00 | 54.50 | 424.00 |

TP trades lasted longer than SL trades in both models. Momentum-only TP exits averaged 27.65h, while SL exits averaged 12.17h. The combined model TP exits averaged 31.31h, while SL exits averaged 14.70h.

## Pair attribution

### Momentum only

| Pair | Closed | Win rate | Avg return | Median return | PnL |
|---|---:|---:|---:|---:|---:|
| ZEC/USD | 111 | 44.14% | 0.54% | -1.72% | 154,536 |
| TRX/USD | 23 | 56.52% | 0.94% | 2.90% | 53,190 |
| TAO/USD | 77 | 41.56% | 0.15% | -1.71% | 21,496 |
| BTC/USD | 27 | 44.44% | 0.35% | -1.74% | 17,581 |
| SOL/USD | 21 | 38.10% | 0.15% | -1.64% | 6,865 |
| BNB/USD | 33 | 39.39% | 0.08% | -1.68% | 5,134 |
| ETH/USD | 28 | 25.00% | -0.82% | -1.85% | -58,997 |
| XRP/USD | 23 | 17.39% | -1.13% | -1.80% | -64,207 |

### Momentum + roll-impact interaction

| Pair | Closed | Win rate | Avg return | Median return | PnL |
|---|---:|---:|---:|---:|---:|
| ZEC/USD | 91 | 45.05% | 0.63% | -1.62% | 139,313 |
| TRX/USD | 23 | 52.17% | 0.75% | 0.92% | 41,718 |
| SOL/USD | 11 | 63.64% | 1.40% | 3.08% | 38,222 |
| BNB/USD | 22 | 40.91% | 0.14% | -1.64% | 6,260 |
| ETH/USD | 11 | 27.27% | -0.72% | -1.87% | -19,863 |
| XRP/USD | 10 | 20.00% | -1.05% | -1.76% | -25,922 |
| BTC/USD | 69 | 31.88% | -0.33% | -1.72% | -59,900 |
| TAO/USD | 72 | 33.33% | -0.46% | -1.77% | -85,434 |

## Trade logs

Full generated artifacts:

- `reports/backtests/live8_1h_momentum_only_tp3_sl15_summary.csv`
- `reports/backtests/live8_1h_momentum_only_tp3_sl15_equity.csv`
- `reports/backtests/live8_1h_momentum_only_tp3_sl15_trades.csv`
- `reports/backtests/live8_1h_momentum_only_tp3_sl15_metadata.json`
- `reports/backtests/live8_1h_momentum_roll_interaction_tp3_sl15_summary.csv`
- `reports/backtests/live8_1h_momentum_roll_interaction_tp3_sl15_equity.csv`
- `reports/backtests/live8_1h_momentum_roll_interaction_tp3_sl15_trades.csv`
- `reports/backtests/live8_1h_momentum_roll_interaction_tp3_sl15_metadata.json`

### Momentum-only trade excerpts

First closed exits:

| Timestamp | Pair | Reason | Entry price | Exit price | Return | PnL | Score |
|---|---|---|---:|---:|---:|---:|---:|
| 1770624000000 | BTC/USD | loss_threshold | 71147.2500 | 69873.9900 | -1.89% | -4,714 | 0.173255 |
| 1770681600000 | ZEC/USD | loss_threshold | 242.8200 | 237.6000 | -2.25% | -5,605 | 0.099327 |
| 1770717600000 | ZEC/USD | gain_threshold | 236.2800 | 243.8800 | 3.11% | 7,668 | 0.181336 |
| 1770721200000 | XRP/USD | loss_threshold | 1.4309 | 1.4081 | -1.69% | -4,200 | 0.098216 |
| 1770732000000 | ZEC/USD | loss_threshold | 238.6800 | 233.9600 | -2.08% | -5,136 | 0.142157 |

Last closed exits:

| Timestamp | Pair | Reason | Entry price | Exit price | Return | PnL | Score |
|---|---|---|---:|---:|---:|---:|---:|
| 1780873200000 | ETH/USD | gain_threshold | 1631.3500 | 1690.5100 | 3.52% | 9,037 | 0.103900 |
| 1780891200000 | ETH/USD | loss_threshold | 1697.5900 | 1657.0500 | -2.49% | -6,480 | 0.179197 |
| 1780930800000 | ZEC/USD | gain_threshold | 427.9800 | 445.5500 | 4.00% | 10,367 | 0.130525 |
| fold_end | ETH/USD | fold_end | 1653.6800 | 1682.7000 | 1.65% | 4,274 | 0.101710 |
| fold_end | ZEC/USD | fold_end | 447.7800 | 447.7800 | -0.10% | -262 | 0.096272 |

### Momentum + roll-impact interaction trade excerpts

First closed exits:

| Timestamp | Pair | Reason | Entry price | Exit price | Return | PnL | Score |
|---|---|---|---:|---:|---:|---:|---:|
| 1770627600000 | BTC/USD | loss_threshold | 70785.9900 | 69630.3500 | -1.73% | -4,325 | 0.236137 |
| 1770703200000 | XRP/USD | loss_threshold | 1.4462 | 1.4202 | -1.90% | -4,715 | 0.114713 |
| 1770717600000 | ZEC/USD | gain_threshold | 236.2800 | 243.8800 | 3.11% | 7,681 | 0.177953 |
| 1770732000000 | ZEC/USD | loss_threshold | 238.6800 | 233.9600 | -2.08% | -5,151 | 0.139713 |
| 1770778800000 | BTC/USD | loss_threshold | 69272.0000 | 68187.9900 | -1.66% | -4,136 | 0.109297 |

Last closed exits:

| Timestamp | Pair | Reason | Entry price | Exit price | Return | PnL | Score |
|---|---|---|---:|---:|---:|---:|---:|
| 1780873200000 | ETH/USD | gain_threshold | 1631.3500 | 1690.5100 | 3.52% | 8,274 | 0.097412 |
| 1780891200000 | ETH/USD | loss_threshold | 1697.5900 | 1657.0500 | -2.49% | -5,933 | 0.180002 |
| 1780930800000 | ZEC/USD | gain_threshold | 427.9800 | 445.5500 | 4.00% | 9,493 | 0.130246 |
| fold_end | BTC/USD | fold_end | 62764.0000 | 63532.0100 | 1.12% | 2,656 | 0.127372 |
| fold_end | ZEC/USD | fold_end | 447.7800 | 447.7800 | -0.10% | -240 | 0.093514 |

## Readout

For this specific live-style test, momentum-only is the stronger candidate. It produced better OS return, better Sharpe/Sortino, higher win rate, and less pair-level damage from BTC/TAO than the combined score.

The combined model did not validate the hypothesis that adding roll-impact improves the traded portfolio under these rules. It reduced trade count and max drawdown slightly, but the improvement in drawdown was not enough to compensate for the lower returns. The negative fitted coefficient on `z_low_roll_impact` is also a warning sign because it runs against the intended "low impact is safer" interpretation.

This is only one 4m/4m fold, so it should not be treated as final. The next validity check should repeat this exact comparison across rolling folds or longer data, then include slippage of 5-10 bps before using the result as a live default.

## Updated baseline

The current research baseline is:

```text
model = momentum_only
score[i,t] = beta_momentum * z_momentum[i,t]
```

On this test, microstructure has been checked in two ways:

- Direct IC alpha: weak evidence on 1h, 6h, and 24h forward returns.
- Linear portfolio alpha overlay: improved in-sample IC, but worse OS traded portfolio return.

The next tests should therefore treat microstructure as a risk or regime layer. It should not be considered a proven additive alpha until it improves the momentum-only portfolio baseline out-of-sample.

## Proposed microstructure risk/regime tests

### 1. Hard entry filter

Trade the same `momentum_only` score, but only allow entries when the candidate passes a microstructure condition:

```text
eligible[i,t] =
    rank(score[i,t]) <= top_k
    and z_low_roll_impact[i,t] >= threshold
```

Thresholds to test:

```text
z_low_roll_impact >= -0.5
z_low_roll_impact >= 0.0
z_low_roll_impact >= 0.5
z_low_roll_impact >= 1.0
```

This directly tests whether low roll-impact names produce cleaner momentum trades.

### 2. Exposure scaler

Keep the same entry ranking, but vary position size by microstructure quality:

```text
base_notional = portfolio_value * position_fraction

scale[i,t] = clip(0.5 + 0.25 * z_low_roll_impact[i,t], 0.25, 1.25)

trade_notional[i,t] = base_notional * scale[i,t]
```

This tests whether microstructure is useful without turning trades fully on/off.

### 3. Regime-level throttle

Build a market-wide liquidity stress measure from the universe:

```text
market_roll_stress[t] = median_i(roll_impact[i,t])
```

Then reduce or disable entries when market-wide stress is high:

```text
if percentile_rank(market_roll_stress[t], rolling_window=30d) >= 80%:
    max_new_entries = 0
or:
    position_fraction = 0.5 * normal_position_fraction
```

This tests whether microstructure detects broad bad trading regimes rather than pair-specific alpha.

### 4. Exit/risk modifier

Keep entries as momentum-only, but tighten exits when microstructure worsens after entry:

```text
if z_low_roll_impact[i,t] drops below threshold:
    effective_sl = min(base_sl, tighter_sl)
```

or:

```text
if z_low_roll_impact[i,t] drops below threshold and trade_return < 0:
    exit position
```

This tests whether roll impact helps avoid bad left-tail continuations.

### Acceptance criteria

The microstructure layer should be compared against the exact momentum-only baseline using the same folds, pairs, costs, and portfolio rules. It should pass at least one of these tests out-of-sample:

- Higher total return with similar or lower drawdown.
- Similar total return with clearly lower drawdown.
- Lower stop-hit rate.
- Lower average loss per stopped trade.
- Better return / max-drawdown ratio.
- More stable pair attribution, with fewer large losses concentrated in one or two pairs.

It should also be tested with 5-10 bps slippage. A risk layer that only works at zero slippage is not robust enough for live deployment.

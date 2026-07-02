"""Market-regime throttle helpers shared by live and backtest trading."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class RegimeThrottleConfig:
    metric_col: str = "roll_impact"
    aggregation: str = "median"
    lookback_bars: int = 720
    percentile: float = 0.80
    min_history_bars: int = 168
    time_col: str = "open_time"
    market_col: str = "market_roll_impact"
    threshold_col: str = "market_roll_impact_threshold"
    stressed_col: str = "market_roll_stressed"
    history_col: str = "market_roll_history_bars"

    def __post_init__(self) -> None:
        if self.lookback_bars <= 0:
            raise ValueError("lookback_bars must be positive")
        if not 0 <= self.percentile <= 1:
            raise ValueError("percentile must be in [0, 1]")
        if self.min_history_bars <= 0:
            raise ValueError("min_history_bars must be positive")
        if self.aggregation not in {"median", "mean"}:
            raise ValueError("aggregation must be 'median' or 'mean'")


@dataclass(frozen=True)
class RegimeDecision:
    is_stressed: bool
    market_roll_impact: float | None
    threshold: float | None
    history_bars: int
    entries_blocked: bool
    reason: str


def add_roll_impact_regime(
    frame: pd.DataFrame,
    config: RegimeThrottleConfig,
) -> pd.DataFrame:
    """Append universe-level roll-impact regime columns without lookahead."""
    missing = {config.time_col, config.metric_col} - set(frame.columns)
    if missing:
        raise ValueError(f"missing required regime columns: {', '.join(sorted(missing))}")

    out = frame.copy()
    grouped = out.groupby(config.time_col, observed=True)[config.metric_col]
    if config.aggregation == "mean":
        market = grouped.mean().sort_index()
    else:
        market = grouped.median().sort_index()
    prior_market = market.shift(1)
    history = prior_market.rolling(config.lookback_bars, min_periods=1).count()
    threshold = prior_market.rolling(
        config.lookback_bars,
        min_periods=config.min_history_bars,
    ).quantile(config.percentile)
    stressed = market.ge(threshold) & threshold.notna() & history.ge(config.min_history_bars)

    out[config.market_col] = out[config.time_col].map(market)
    out[config.threshold_col] = out[config.time_col].map(threshold)
    out[config.history_col] = out[config.time_col].map(history).fillna(0).astype(int)
    out[config.stressed_col] = out[config.time_col].map(stressed).fillna(False).astype(bool)
    return out


def decide_regime_throttle(
    scores_at_timestamp: pd.DataFrame,
    config: RegimeThrottleConfig,
) -> RegimeDecision:
    """Return the entry-throttle decision for one scored timestamp."""
    if scores_at_timestamp.empty:
        return RegimeDecision(
            is_stressed=False,
            market_roll_impact=None,
            threshold=None,
            history_bars=0,
            entries_blocked=False,
            reason="empty_scores",
        )

    required = {
        config.market_col,
        config.threshold_col,
        config.stressed_col,
        config.history_col,
    }
    missing = required - set(scores_at_timestamp.columns)
    if missing:
        raise ValueError(f"missing regime decision columns: {', '.join(sorted(missing))}")

    row = scores_at_timestamp.iloc[0]
    market_value = _none_if_na(row[config.market_col])
    threshold = _none_if_na(row[config.threshold_col])
    history_bars = int(row[config.history_col]) if pd.notna(row[config.history_col]) else 0
    is_stressed = bool(row[config.stressed_col])

    if history_bars < config.min_history_bars:
        reason = "insufficient_history"
        is_stressed = False
    elif threshold is None:
        reason = "missing_threshold"
        is_stressed = False
    elif is_stressed:
        reason = "market_roll_impact_stressed"
    else:
        reason = "market_roll_impact_normal"

    return RegimeDecision(
        is_stressed=is_stressed,
        market_roll_impact=market_value,
        threshold=threshold,
        history_bars=history_bars,
        entries_blocked=is_stressed,
        reason=reason,
    )


def _none_if_na(value: object) -> float | None:
    if pd.isna(value):
        return None
    return float(value)

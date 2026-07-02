import pandas as pd

from bot.strategy.regime import (
    RegimeThrottleConfig,
    add_roll_impact_regime,
    decide_regime_throttle,
)


def test_roll_impact_regime_uses_cross_sectional_median_per_timestamp():
    frame = pd.DataFrame(
        [
            {"open_time": 1, "pair": "BTC/USD", "roll_impact": 1.0},
            {"open_time": 1, "pair": "ETH/USD", "roll_impact": 3.0},
            {"open_time": 2, "pair": "BTC/USD", "roll_impact": 10.0},
            {"open_time": 2, "pair": "ETH/USD", "roll_impact": 30.0},
        ]
    )

    out = add_roll_impact_regime(frame, RegimeThrottleConfig(lookback_bars=2, min_history_bars=1))

    assert out.loc[out["open_time"].eq(1), "market_roll_impact"].unique().tolist() == [2.0]
    assert out.loc[out["open_time"].eq(2), "market_roll_impact"].unique().tolist() == [20.0]


def test_roll_impact_regime_can_use_cross_sectional_mean_per_timestamp():
    frame = pd.DataFrame(
        [
            {"open_time": 1, "pair": "BTC/USD", "roll_impact": 1.0},
            {"open_time": 1, "pair": "ETH/USD", "roll_impact": 5.0},
            {"open_time": 1, "pair": "SOL/USD", "roll_impact": 99.0},
            {"open_time": 2, "pair": "BTC/USD", "roll_impact": 10.0},
            {"open_time": 2, "pair": "ETH/USD", "roll_impact": 20.0},
            {"open_time": 2, "pair": "SOL/USD", "roll_impact": 30.0},
        ]
    )

    out = add_roll_impact_regime(
        frame,
        RegimeThrottleConfig(
            aggregation="mean",
            lookback_bars=2,
            min_history_bars=1,
        ),
    )

    assert out.loc[out["open_time"].eq(1), "market_roll_impact"].unique().tolist() == [35.0]
    assert out.loc[out["open_time"].eq(2), "market_roll_impact"].unique().tolist() == [20.0]


def test_roll_impact_regime_threshold_uses_only_prior_bars():
    frame = pd.DataFrame(
        [
            {"open_time": 1, "pair": "BTC/USD", "roll_impact": 1.0},
            {"open_time": 1, "pair": "ETH/USD", "roll_impact": 1.0},
            {"open_time": 2, "pair": "BTC/USD", "roll_impact": 2.0},
            {"open_time": 2, "pair": "ETH/USD", "roll_impact": 2.0},
            {"open_time": 3, "pair": "BTC/USD", "roll_impact": 100.0},
            {"open_time": 3, "pair": "ETH/USD", "roll_impact": 100.0},
        ]
    )
    config = RegimeThrottleConfig(lookback_bars=2, percentile=0.5, min_history_bars=2)

    out = add_roll_impact_regime(frame, config)
    at_t3 = out[out["open_time"].eq(3)].iloc[0]

    assert at_t3["market_roll_impact_threshold"] == 1.5
    assert bool(at_t3["market_roll_stressed"])


def test_roll_impact_regime_insufficient_history_does_not_trigger_stress():
    frame = pd.DataFrame(
        [
            {"open_time": 1, "pair": "BTC/USD", "roll_impact": 1.0},
            {"open_time": 2, "pair": "BTC/USD", "roll_impact": 100.0},
        ]
    )
    config = RegimeThrottleConfig(lookback_bars=3, percentile=0.8, min_history_bars=2)

    out = add_roll_impact_regime(frame, config)
    decision = decide_regime_throttle(out[out["open_time"].eq(2)], config)

    assert not decision.is_stressed
    assert not decision.entries_blocked
    assert decision.reason == "insufficient_history"


def test_roll_impact_regime_stress_triggers_at_threshold():
    frame = pd.DataFrame(
        [
            {"open_time": 1, "pair": "BTC/USD", "roll_impact": 1.0},
            {"open_time": 2, "pair": "BTC/USD", "roll_impact": 2.0},
            {"open_time": 3, "pair": "BTC/USD", "roll_impact": 2.0},
        ]
    )
    config = RegimeThrottleConfig(lookback_bars=2, percentile=1.0, min_history_bars=2)

    out = add_roll_impact_regime(frame, config)
    decision = decide_regime_throttle(out[out["open_time"].eq(3)], config)

    assert decision.is_stressed
    assert decision.entries_blocked
    assert decision.market_roll_impact == 2.0
    assert decision.threshold == 2.0

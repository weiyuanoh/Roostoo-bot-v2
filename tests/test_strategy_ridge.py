import pandas as pd
import pytest

from bot.backtest.portfolio import Position
from bot.strategy.ridge import (
    add_training_target,
    build_cycle_intents,
    entry_intents,
    exit_intent,
    score_ranks,
)
from bot.strategy.regime import RegimeThrottleConfig


def test_add_training_target_adds_forward_return_and_target_z():
    rows = []
    timestamps = pd.date_range("2026-01-01", periods=4, freq="h", tz="UTC")
    for p_idx, pair in enumerate(["BTC/USD", "ETH/USD"]):
        for t_idx, timestamp in enumerate(timestamps):
            rows.append(
                {
                    "pair": pair,
                    "open_time": int(timestamp.timestamp() * 1000),
                    "close": 100 + p_idx * 10 + t_idx,
                }
            )
    frame = pd.DataFrame(rows)

    out = add_training_target(frame, horizon=1)

    assert "forward_return_1" in out.columns
    assert "target_z" in out.columns
    terminal_targets = out.sort_values("open_time").groupby("pair")["forward_return_1"].tail(1)
    assert terminal_targets.isna().all()


def test_entry_intents_selects_top_scores_with_cash_and_slots():
    scores = pd.DataFrame(
        [
            {"pair": "BTC/USD", "ridge_score": 0.1, "close": 100.0},
            {"pair": "ETH/USD", "ridge_score": 0.5, "close": 50.0},
            {"pair": "SOL/USD", "ridge_score": 0.3, "close": 10.0},
        ]
    )

    intents = entry_intents(
        scores,
        {"BTC/USD"},
        portfolio_value=1000.0,
        available_cash=200.0,
        position_fraction=0.1,
        max_positions=2,
    )

    assert len(intents) == 1
    assert intents[0].pair == "ETH/USD"
    assert intents[0].notional_usd == pytest.approx(100.0)


def test_entry_intents_respects_top_k_before_skipping_held_pairs():
    scores = pd.DataFrame(
        [
            {"pair": "BTC/USD", "ridge_score": 1.0, "close": 100.0},
            {"pair": "ETH/USD", "ridge_score": 0.9, "close": 50.0},
            {"pair": "SOL/USD", "ridge_score": 0.8, "close": 10.0},
        ]
    )

    intents = entry_intents(
        scores,
        {"BTC/USD"},
        portfolio_value=1000.0,
        available_cash=1000.0,
        position_fraction=0.1,
        max_positions=3,
        top_k=1,
    )

    assert intents == []


def test_entry_intents_respects_max_new_entries():
    scores = pd.DataFrame(
        [
            {"pair": "BTC/USD", "ridge_score": 1.0, "close": 100.0},
            {"pair": "ETH/USD", "ridge_score": 0.9, "close": 50.0},
            {"pair": "SOL/USD", "ridge_score": 0.8, "close": 10.0},
        ]
    )

    intents = entry_intents(
        scores,
        set(),
        portfolio_value=1000.0,
        available_cash=1000.0,
        position_fraction=0.1,
        max_positions=3,
        max_new_entries=1,
    )

    assert [intent.pair for intent in intents] == ["BTC/USD"]


def test_exit_intent_fires_only_at_thresholds():
    assert exit_intent("BTC/USD", 1.0, 100.0, 149.0, threshold=0.5) is None

    gain = exit_intent("BTC/USD", 1.0, 100.0, 150.0, threshold=0.5)
    loss = exit_intent("BTC/USD", 1.0, 100.0, 50.0, threshold=0.5)

    assert gain is not None
    assert gain.reason == "gain_threshold"
    assert loss is not None
    assert loss.reason == "loss_threshold"


def test_exit_intent_supports_split_take_profit_and_stop_loss():
    hold = exit_intent(
        "BTC/USD",
        1.0,
        100.0,
        125.0,
        take_profit=0.5,
        stop_loss=0.2,
    )
    stop = exit_intent(
        "BTC/USD",
        1.0,
        100.0,
        80.0,
        take_profit=0.5,
        stop_loss=0.2,
    )
    take = exit_intent(
        "BTC/USD",
        1.0,
        100.0,
        150.0,
        take_profit=0.5,
        stop_loss=0.2,
    )

    assert hold is None
    assert stop is not None
    assert stop.reason == "loss_threshold"
    assert take is not None
    assert take.reason == "gain_threshold"


def test_build_cycle_intents_keeps_exiting_pair_held_for_same_cycle_entries():
    scores = pd.DataFrame(
        [
            {"pair": "BTC/USD", "ridge_score": 10.0, "close": 150.0},
            {"pair": "ETH/USD", "ridge_score": 9.0, "close": 50.0},
        ]
    )
    positions = {
        "BTC/USD": Position(
            pair="BTC/USD",
            quantity=1.0,
            entry_price=100.0,
            entry_time=1,
            entry_value=100.0,
            entry_score=1.0,
        )
    }

    cycle = build_cycle_intents(
        scores,
        positions,
        {"BTC/USD": 150.0, "ETH/USD": 50.0},
        portfolio_value=1_000.0,
        available_cash=1_000.0,
        position_fraction=0.5,
        max_positions=2,
        take_profit=0.5,
        stop_loss=0.2,
    )

    assert [intent.pair for intent in cycle.exits] == ["BTC/USD"]
    assert [intent.pair for intent in cycle.entries] == ["ETH/USD"]


def test_score_ranks_orders_scores_descending():
    scores = pd.DataFrame(
        [
            {"pair": "BTC/USD", "ridge_score": 0.1},
            {"pair": "ETH/USD", "ridge_score": 0.5},
            {"pair": "SOL/USD", "ridge_score": 0.3},
        ]
    )

    assert score_ranks(scores) == {"ETH/USD": 1, "SOL/USD": 2, "BTC/USD": 3}


def test_build_cycle_intents_exits_when_held_pair_rank_decays_below_threshold():
    scores = pd.DataFrame(
        [
            {"pair": "BTC/USD", "ridge_score": 1.0, "close": 100.0},
            {"pair": "ETH/USD", "ridge_score": 0.9, "close": 50.0},
            {"pair": "SOL/USD", "ridge_score": 0.8, "close": 10.0},
        ]
    )
    positions = {
        "SOL/USD": Position(
            pair="SOL/USD",
            quantity=10.0,
            entry_price=10.0,
            entry_time=1,
            entry_value=100.0,
            entry_score=1.0,
        )
    }

    cycle = build_cycle_intents(
        scores,
        positions,
        {"BTC/USD": 100.0, "ETH/USD": 50.0, "SOL/USD": 10.0},
        portfolio_value=1_000.0,
        available_cash=1_000.0,
        position_fraction=0.5,
        max_positions=2,
        rank_exit_threshold=2,
        take_profit=10.0,
        stop_loss=10.0,
    )

    assert [intent.pair for intent in cycle.exits] == ["SOL/USD"]
    assert cycle.exits[0].reason == "rank_decay"
    assert [intent.pair for intent in cycle.entries] == ["BTC/USD"]


def test_build_cycle_intents_blocks_entries_but_keeps_exits_during_stress():
    scores = pd.DataFrame(
        [
            {
                "pair": "BTC/USD",
                "ridge_score": 10.0,
                "close": 150.0,
                "market_roll_impact": 5.0,
                "market_roll_impact_threshold": 4.0,
                "market_roll_history_bars": 10,
                "market_roll_stressed": True,
            },
            {
                "pair": "ETH/USD",
                "ridge_score": 9.0,
                "close": 50.0,
                "market_roll_impact": 5.0,
                "market_roll_impact_threshold": 4.0,
                "market_roll_history_bars": 10,
                "market_roll_stressed": True,
            },
        ]
    )
    positions = {
        "BTC/USD": Position(
            pair="BTC/USD",
            quantity=1.0,
            entry_price=100.0,
            entry_time=1,
            entry_value=100.0,
            entry_score=1.0,
        )
    }

    cycle = build_cycle_intents(
        scores,
        positions,
        {"BTC/USD": 150.0, "ETH/USD": 50.0},
        portfolio_value=1_000.0,
        available_cash=1_000.0,
        position_fraction=0.5,
        max_positions=2,
        regime_config=RegimeThrottleConfig(lookback_bars=10, min_history_bars=10),
        take_profit=0.5,
        stop_loss=0.2,
    )

    assert [intent.pair for intent in cycle.exits] == ["BTC/USD"]
    assert cycle.entries == []
    assert cycle.regime is not None
    assert cycle.regime.entries_blocked


def test_build_cycle_intents_preserves_entries_when_regime_is_not_stressed():
    scores = pd.DataFrame(
        [
            {
                "pair": "ETH/USD",
                "ridge_score": 9.0,
                "close": 50.0,
                "market_roll_impact": 3.0,
                "market_roll_impact_threshold": 4.0,
                "market_roll_history_bars": 10,
                "market_roll_stressed": False,
            },
        ]
    )

    cycle = build_cycle_intents(
        scores,
        {},
        {"ETH/USD": 50.0},
        portfolio_value=1_000.0,
        available_cash=1_000.0,
        position_fraction=0.5,
        max_positions=2,
        regime_config=RegimeThrottleConfig(lookback_bars=10, min_history_bars=10),
        take_profit=0.5,
        stop_loss=0.2,
    )

    assert [intent.pair for intent in cycle.entries] == ["ETH/USD"]
    assert cycle.regime is not None
    assert not cycle.regime.is_stressed

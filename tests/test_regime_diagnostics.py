import pandas as pd
import pytest

from bot.backtest.regime_diagnostics import (
    add_decision_time_regime_features,
    build_reports,
    entry_regime_frame,
    pair_trade_entries,
    post_exit_returns,
)


def synthetic_features():
    rows = []
    for idx, open_time in enumerate([0, 3_600_000, 7_200_000, 10_800_000]):
        for pair, score, close in [
            ("AAA/USD", 3.0 + idx, 100.0 + idx * 2),
            ("BBB/USD", 2.0, 50.0 + idx),
            ("CCC/USD", 1.0, 25.0),
        ]:
            rows.append(
                {
                    "open_time": open_time,
                    "timestamp": pd.Timestamp(open_time, unit="ms", tz="UTC"),
                    "pair": pair,
                    "close": close,
                    "ridge_score": score,
                    "fold": 0,
                }
            )
    return pd.DataFrame(rows)


def synthetic_trades():
    return pd.DataFrame(
        [
            {
                "timestamp": 0,
                "pair": "AAA/USD",
                "side": "BUY",
                "price": 100.0,
                "quantity": 1.0,
                "notional": 100.0,
                "fee": 0.1,
                "score": 3.0,
                "reason": "ridge_entry",
            },
            {
                "timestamp": 7_200_000,
                "pair": "AAA/USD",
                "side": "SELL",
                "price": 104.0,
                "quantity": 1.0,
                "notional": 104.0,
                "fee": 0.1,
                "score": 3.0,
                "reason": "gain_threshold",
                "entry_time": 0,
                "entry_price": 100.0,
                "entry_value": 100.0,
                "pnl": 3.9,
                "return_pct": 0.039,
            },
            {
                "timestamp": 10_800_000,
                "pair": "AAA/USD",
                "side": "BUY",
                "price": 106.0,
                "quantity": 1.0,
                "notional": 106.0,
                "fee": 0.1,
                "score": 6.0,
                "reason": "ridge_entry",
            },
        ]
    )


def test_decision_time_features_use_current_and_prior_only():
    regime = add_decision_time_regime_features(synthetic_features())
    entry = regime[(regime["pair"].eq("AAA/USD")) & (regime["open_time"].eq(0))].iloc[0]
    later = regime[(regime["pair"].eq("AAA/USD")) & (regime["open_time"].eq(7_200_000))].iloc[0]

    assert entry["entry_rank"] == 1
    assert entry["pair_top3_count_3h"] == 1
    assert later["pair_top3_count_3h"] == 3
    assert pd.isna(entry["score_gap_rank1_rank2_trailing_median"])


def test_pair_trade_entries_and_reentry_detection():
    entries = pair_trade_entries(
        synthetic_trades(),
        pd.DataFrame({"open_time": [0, 10_800_000], "positions": [1, 1]}),
    )
    regime = add_decision_time_regime_features(synthetic_features())
    joined = entry_regime_frame(entries, regime)

    first = joined[joined["entry_time"].eq(0)].iloc[0]
    second = joined[joined["entry_time"].eq(10_800_000)].iloc[0]
    assert first["exit_reason"] == "gain_threshold"
    assert bool(first["tp_hit"]) is True
    assert bool(second["is_same_pair_reentry"]) is True
    assert second["prior_exit_reason"] == "gain_threshold"
    assert second["hours_since_prior_exit"] == pytest.approx(1.0)


def test_post_exit_returns_are_labels_only():
    features = add_decision_time_regime_features(synthetic_features())
    sells = synthetic_trades()[lambda df: df["side"].eq("SELL")]

    report = post_exit_returns(sells, features, horizons=(1,))

    assert report["post_exit_return"].iloc[0] == pytest.approx((106.0 / 104.0) - 1.0)


def test_build_reports_produces_expected_tables():
    reports = build_reports(
        features=synthetic_features(),
        trades=synthetic_trades(),
        equity=pd.DataFrame({"open_time": [0, 10_800_000], "positions": [1, 1]}),
        horizons=(1,),
    )

    assert set(reports) == {
        "entry_regime",
        "trade_outcome_attribution",
        "rank_persistence",
        "score_gap_report",
        "same_pair_reentry",
        "post_exit_returns",
        "shadow_regime_filters",
    }
    assert not reports["entry_regime"].empty
    assert not reports["shadow_regime_filters"].empty

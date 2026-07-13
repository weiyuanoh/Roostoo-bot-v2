import pandas as pd
import pytest

from bot.backtest.refit_policy_experiment import (
    OosFold,
    build_refit_segments,
    build_oos_folds,
    filter_common_history_pairs,
    latest_oos_window,
    ridge_training_slice,
    run_refit_policy_backtest,
)
from bot.strategy.regime import ClusterRegimeGateConfig


def _synthetic_refit_frame(*, days: int = 105) -> pd.DataFrame:
    rows = []
    timestamps = pd.date_range("2025-01-01", periods=24 * days, freq="h", tz="UTC")
    pairs = [f"P{idx}/USD" for idx in range(4)]
    for t_idx, timestamp in enumerate(timestamps):
        for p_idx, pair in enumerate(pairs):
            close = 100.0 + p_idx + t_idx * (0.01 if p_idx == 3 else 0.001)
            rows.append(
                {
                    "open_time": int(timestamp.timestamp() * 1000),
                    "timestamp": timestamp,
                    "pair": pair,
                    "close": close,
                    "target_z": float(p_idx),
                    "z_momentum": float(p_idx),
                    "z_low_roll_impact": 0.0,
                    "z_momentum_x_low_roll_impact": 0.0,
                    "forward_return_1": float(p_idx),
                }
            )
    return pd.DataFrame(rows)


def test_refit_schedule_splits_weekly_and_monthly_boundaries():
    os_start = pd.Timestamp("2025-05-01", tz="UTC")
    os_end = pd.Timestamp("2025-07-01", tz="UTC")

    segments = build_refit_segments(
        os_start,
        os_end,
        ridge_refit_days=7,
        regime_refit_months=1,
        scheduled=True,
    )

    ridge_starts = [segment.start for segment in segments if segment.ridge_refit]
    regime_starts = [segment.start for segment in segments if segment.regime_refit]
    assert ridge_starts[:3] == [
        pd.Timestamp("2025-05-01", tz="UTC"),
        pd.Timestamp("2025-05-08", tz="UTC"),
        pd.Timestamp("2025-05-15", tz="UTC"),
    ]
    assert pd.Timestamp("2025-06-01", tz="UTC") in regime_starts
    assert segments[-1].end == os_end


def test_ridge_training_slice_excludes_labels_that_need_future_data():
    timestamps = pd.date_range("2025-01-01", periods=10, freq="h", tz="UTC")
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open_time": [int(ts.timestamp() * 1000) for ts in timestamps],
        }
    )

    train = ridge_training_slice(
        frame,
        train_start=timestamps[0],
        train_end=timestamps[8],
        horizon=3,
    )

    assert train["timestamp"].max() == timestamps[5]


def test_latest_oos_window_fails_clearly_without_regime_history():
    frame = _synthetic_refit_frame(days=90)

    with pytest.raises(ValueError, match="not enough history.*24-month regime window"):
        latest_oos_window(frame, os_months=2, required_train_months=24)


def test_build_oos_folds_steps_backward_and_returns_chronological_folds():
    frame = _synthetic_refit_frame(days=150)

    folds = build_oos_folds(
        frame,
        os_months=1,
        required_train_months=1,
        folds=3,
        fold_step_days=7,
    )

    assert [fold.fold_id for fold in folds] == [0, 1, 2]
    assert folds[0].os_start < folds[1].os_start < folds[2].os_start
    assert folds[1].os_start - folds[0].os_start == pd.Timedelta(days=7)


def test_common_history_filter_drops_pairs_without_full_fold_coverage():
    frame = _synthetic_refit_frame(days=120)
    stale = frame[frame["pair"].eq("P0/USD")].copy()
    stale["pair"] = "STALE/USD"
    stale = stale.iloc[24 * 10 : -24 * 10]
    frame = pd.concat([frame, stale], ignore_index=True)
    folds = [
        OosFold(
            fold_id=0,
            os_start=pd.Timestamp("2025-03-15", tz="UTC"),
            os_end=pd.Timestamp("2025-04-15", tz="UTC"),
        ),
        OosFold(
            fold_id=1,
            os_start=pd.Timestamp("2025-03-22", tz="UTC"),
            os_end=pd.Timestamp("2025-04-22", tz="UTC"),
        ),
    ]

    filtered, pairs = filter_common_history_pairs(frame, os_folds=folds, required_train_months=2)

    assert "STALE/USD" not in pairs
    assert set(filtered["pair"].unique()) == set(pairs)


def test_fixed_policy_uses_one_ridge_and_regime_artifact():
    frame = _synthetic_refit_frame()

    summary, _, _, segments = run_refit_policy_backtest(
        frame,
        policy="fixed",
        horizon=1,
        os_months=2,
        ridge_train_months=1,
        regime_train_months=1,
        ridge_alphas=(1.0,),
        initial_cash=10_000.0,
        position_fraction=0.25,
        take_profit=10.0,
        stop_loss=10.0,
        fee_bps=0.0,
        slippage_bps=0.0,
        max_positions=1,
        top_k=1,
        max_new_entries=1,
        cluster_gate_config=ClusterRegimeGateConfig(n_clusters=2, min_cluster_trades=10_000),
    )

    assert summary["ridge_refits"].iloc[0] == 1
    assert summary["regime_refits"].iloc[0] == 1
    assert len(segments) == 1
    assert segments["ridge_refit"].sum() == 1
    assert segments["regime_refit"].sum() == 1


def test_scheduled_policy_refits_ridge_weekly_and_regime_monthly():
    frame = _synthetic_refit_frame()

    summary, _, _, segments = run_refit_policy_backtest(
        frame,
        policy="scheduled",
        horizon=1,
        os_months=2,
        ridge_train_months=1,
        regime_train_months=1,
        ridge_refit_days=7,
        regime_refit_months=1,
        ridge_alphas=(1.0,),
        initial_cash=10_000.0,
        position_fraction=0.25,
        take_profit=10.0,
        stop_loss=10.0,
        fee_bps=0.0,
        slippage_bps=0.0,
        max_positions=1,
        top_k=1,
        max_new_entries=1,
        cluster_gate_config=ClusterRegimeGateConfig(n_clusters=2, min_cluster_trades=10_000),
    )

    assert summary["ridge_refits"].iloc[0] > 1
    assert summary["regime_refits"].iloc[0] == 2
    assert segments["ridge_refit"].sum() > segments["regime_refit"].sum()
    assert segments["regime_train_end"].max() < summary["os_end"].iloc[0]


def test_scheduled_policy_carries_positions_until_final_oos_liquidation():
    frame = _synthetic_refit_frame()

    _, _, trades, segments = run_refit_policy_backtest(
        frame,
        policy="scheduled",
        horizon=1,
        os_months=2,
        ridge_train_months=1,
        regime_train_months=1,
        ridge_refit_days=7,
        regime_refit_months=1,
        ridge_alphas=(1.0,),
        initial_cash=10_000.0,
        position_fraction=0.25,
        take_profit=10.0,
        stop_loss=10.0,
        fee_bps=0.0,
        slippage_bps=0.0,
        max_positions=1,
        top_k=1,
        max_new_entries=1,
        cluster_gate_config=None,
    )

    buys = trades[trades["side"].eq("BUY")]
    sells = trades[trades["side"].eq("SELL")]
    refit_boundary_starts = set(segments.loc[segments["refit_segment_id"] > 0, "segment_start"])

    assert len(buys) == 1
    assert len(sells) == 1
    assert sells["reason"].iloc[0] == "fold_end"
    assert sells["timestamp"].iloc[0] == "oos_end"
    assert not refit_boundary_starts.intersection(set(pd.to_datetime(sells["timestamp"], errors="coerce")))

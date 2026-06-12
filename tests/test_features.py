import math

import numpy as np
import pandas as pd
import pytest

from bot.features import (
    add_alpha_features,
    alpha_feature_columns,
    breakout_quality,
    log_return,
    realized_volatility,
    trend_persistence,
    vol_adjusted_momentum,
)


def test_log_return_matches_hand_computed_values():
    price = pd.Series([100.0, 110.0, 121.0])

    out = log_return(price, horizon=2)

    assert out.iloc[0] != out.iloc[0]
    assert out.iloc[1] != out.iloc[1]
    assert out.iloc[2] == pytest.approx(math.log(1.21))


def test_log_return_rejects_bad_horizon():
    with pytest.raises(ValueError, match="horizon"):
        log_return(pd.Series([1.0, 2.0]), horizon=0)


def test_realized_volatility_requires_window_of_at_least_two():
    with pytest.raises(ValueError, match="window"):
        realized_volatility(pd.Series([1.0, 2.0]), window=1)


def test_vol_adjusted_momentum_returns_nan_for_flat_prices():
    price = pd.Series([100.0] * 8)

    out = vol_adjusted_momentum(price, window=4)

    assert out.dropna().empty


def test_trend_persistence_is_centered_fraction_of_positive_bars():
    price = pd.Series([100.0, 101.0, 102.0, 101.0, 103.0])

    out = trend_persistence(price, window=4)

    assert out.iloc[-1] == pytest.approx(0.5)


def test_breakout_quality_uses_prior_window_high_and_volume():
    close = pd.Series([10.0, 10.0, 10.0, 10.0, 12.0])
    high = pd.Series([10.0, 10.0, 10.0, 10.0, 12.0])
    volume = pd.Series([100.0, 100.0, 100.0, 100.0, 200.0])

    out = breakout_quality(close, high, volume, window=3)

    assert out.iloc[-1] == pytest.approx(0.4)


def test_breakout_quality_does_not_treat_current_high_as_prior_high():
    close = pd.Series([10.0, 10.0, 10.0, 10.0, 11.0])
    high = pd.Series([10.0, 10.0, 10.0, 10.0, 99.0])
    volume = pd.Series([100.0] * 5)

    out = breakout_quality(close, high, volume, window=3)

    assert out.iloc[-1] == pytest.approx(0.1)


def test_add_alpha_features_sorts_by_pair_and_time_without_mutating_input():
    frame = pd.DataFrame(
        {
            "pair": ["B", "A", "A", "B", "A", "B"],
            "open_time": [1, 2, 1, 2, 3, 3],
            "close": [20.0, 12.0, 10.0, 22.0, 15.0, 25.0],
            "high": [20.0, 12.0, 10.0, 22.0, 15.0, 25.0],
            "volume": [100.0] * 6,
        }
    )
    original_columns = frame.columns.tolist()

    out = add_alpha_features(frame, windows=(2,))

    assert frame.columns.tolist() == original_columns
    assert out[["pair", "open_time"]].values.tolist() == [
        ["A", 1],
        ["A", 2],
        ["A", 3],
        ["B", 1],
        ["B", 2],
        ["B", 3],
    ]
    a_last = out[out["pair"].eq("A")].iloc[-1]
    assert a_last["log_return_2"] == pytest.approx(math.log(15.0 / 10.0))


def test_relative_strength_and_residual_momentum_are_cross_sectional():
    frame = pd.DataFrame(
        {
            "pair": ["A", "A", "A", "B", "B", "B"],
            "open_time": [1, 2, 3, 1, 2, 3],
            "close": [10.0, 11.0, 13.0, 10.0, 10.5, 10.6],
            "high": [10.0, 11.0, 13.0, 10.0, 10.5, 10.6],
            "volume": [100.0] * 6,
        }
    )

    out = add_alpha_features(frame, windows=(2,))
    at_t3 = out[out["open_time"].eq(3)].sort_values("pair")

    assert at_t3.loc[at_t3["pair"].eq("A"), "relative_strength_2"].iloc[0] == pytest.approx(1.0)
    assert at_t3.loc[at_t3["pair"].eq("B"), "relative_strength_2"].iloc[0] == pytest.approx(0.5)
    assert at_t3["residual_momentum_2"].sum() == pytest.approx(0.0)


def test_microstructure_adjusted_momentum_rewards_lower_risk_rank():
    frame = pd.DataFrame(
        {
            "pair": ["A", "A", "A", "B", "B", "B"],
            "open_time": [1, 2, 3, 1, 2, 3],
            "close": [10.0, 11.0, 13.0, 10.0, 11.0, 13.0],
            "high": [10.0, 11.0, 13.0, 10.0, 11.0, 13.0],
            "volume": [100.0] * 6,
            "vpin": [0.1, 0.1, 0.1, 0.9, 0.9, 0.9],
            "roll_impact": [0.1, 0.1, 0.1, 0.9, 0.9, 0.9],
        }
    )

    out = add_alpha_features(frame, windows=(2,))
    at_t3 = out[out["open_time"].eq(3)].sort_values("pair")

    assert at_t3["vol_adj_momentum_2"].iloc[0] == pytest.approx(
        at_t3["vol_adj_momentum_2"].iloc[1]
    )
    assert at_t3["momentum_low_vpin_2"].iloc[0] > at_t3["momentum_low_vpin_2"].iloc[1]
    assert (
        at_t3["momentum_low_roll_impact_2"].iloc[0]
        > at_t3["momentum_low_roll_impact_2"].iloc[1]
    )


def test_missing_microstructure_columns_create_nan_interaction_features():
    frame = pd.DataFrame(
        {
            "pair": ["A", "A", "A"],
            "open_time": [1, 2, 3],
            "close": [10.0, 11.0, 13.0],
            "high": [10.0, 11.0, 13.0],
            "volume": [100.0] * 3,
        }
    )

    out = add_alpha_features(frame, windows=(2,))

    assert out["momentum_low_vpin_2"].isna().all()
    assert out["momentum_low_roll_impact_2"].isna().all()


def test_alpha_feature_columns_lists_expected_names():
    columns = alpha_feature_columns((6,))

    assert columns == [
        "log_return_6",
        "realized_vol_6",
        "vol_adj_momentum_6",
        "trend_persistence_6",
        "breakout_quality_6",
        "relative_strength_6",
        "residual_momentum_6",
        "momentum_low_vpin_6",
        "momentum_low_roll_impact_6",
    ]


def test_add_alpha_features_rejects_missing_required_columns():
    with pytest.raises(ValueError, match="missing required columns"):
        add_alpha_features(pd.DataFrame({"close": [1.0, 2.0]}), windows=(2,))


def test_add_alpha_features_rejects_empty_or_too_short_windows():
    frame = pd.DataFrame(
        {
            "pair": ["A", "A"],
            "open_time": [1, 2],
            "close": [1.0, 2.0],
            "high": [1.0, 2.0],
            "volume": [1.0, 1.0],
        }
    )

    with pytest.raises(ValueError, match="windows"):
        add_alpha_features(frame, windows=())

    with pytest.raises(ValueError, match="windows"):
        add_alpha_features(frame, windows=(1,))


def test_feature_outputs_do_not_contain_infinite_values():
    frame = pd.DataFrame(
        {
            "pair": ["A"] * 5,
            "open_time": [1, 2, 3, 4, 5],
            "close": [10.0, 0.0, 11.0, 12.0, 13.0],
            "high": [10.0, 0.0, 11.0, 12.0, 13.0],
            "volume": [100.0, 0.0, 100.0, 100.0, 100.0],
        }
    )

    out = add_alpha_features(frame, windows=(2,))
    numeric = out.select_dtypes(include=[np.number])

    assert not np.isinf(numeric.to_numpy()).any()


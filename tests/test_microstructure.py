import math

import numpy as np
import pandas as pd
import pytest

from bot.microstructure import (
    MICROSTRUCTURE_COLUMNS,
    amihud_measure,
    compute_microstructure_measures,
    kyle_lambda,
    latest_microstructure_snapshot,
    roll_impact,
    roll_measure,
    vpin,
)


def test_roll_measure_matches_rolling_covariance_formula():
    price = pd.Series([100.0, 101.0, 100.0, 101.0, 100.0, 99.0])
    window = 3

    expected = 2.0 * np.sqrt(price.diff().rolling(window).cov(price.diff().shift(1)).abs())

    pd.testing.assert_series_equal(roll_measure(price, window), expected)


def test_roll_impact_uses_current_bar_dollar_volume():
    price = pd.Series([100.0, 101.0, 100.0, 101.0, 100.0, 99.0])
    volume = pd.Series([10.0, 11.0, 12.0, 13.0, 14.0, 15.0])
    window = 3

    expected = roll_measure(price, window) / (price * volume)

    pd.testing.assert_series_equal(roll_impact(price, volume, window), expected)


def test_amihud_is_rolling_abs_return_over_dollar_volume():
    price = pd.Series([100.0, 102.0, 101.0, 103.0, 104.0])
    volume = pd.Series([10.0, 10.0, 20.0, 20.0, 40.0])
    window = 2

    per_bar = price.pct_change().abs() / (price * volume)
    expected = per_bar.rolling(window).mean()

    pd.testing.assert_series_equal(amihud_measure(price, volume, window), expected)


def test_kyle_lambda_uses_price_change_over_signed_volume():
    price = pd.Series([100.0, 101.0, 102.0, 101.0, 103.0])
    volume = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])
    window = 2

    signed_volume = np.sign(price.diff()) * volume
    expected = (price - price.shift(window)) / signed_volume.rolling(window + 1).sum()

    pd.testing.assert_series_equal(kyle_lambda(price, volume, window), expected)


def test_vpin_stays_between_zero_and_one_when_defined():
    price = pd.Series([100, 101, 99, 102, 98, 103, 97, 104, 96, 105], dtype=float)
    volume = pd.Series([10] * len(price), dtype=float)

    values = vpin(price, volume, window=3).dropna()

    assert not values.empty
    assert ((values >= 0) & (values <= 1)).all()


def test_compute_microstructure_measures_appends_expected_columns():
    candles = pd.DataFrame(
        {
            "close": [100, 101, 99, 102, 98, 103, 97, 104, 96, 105],
            "volume": [10] * 10,
        }
    )

    out = compute_microstructure_measures(candles, window=3)

    assert list(MICROSTRUCTURE_COLUMNS) == [
        "roll_measure",
        "roll_impact",
        "amihud",
        "kyle_lambda",
        "vpin",
    ]
    assert set(MICROSTRUCTURE_COLUMNS).issubset(out.columns)
    assert out["roll_measure"].notna().any()


def test_latest_microstructure_snapshot_returns_plain_floats():
    candles = pd.DataFrame(
        {
            "close": [100, 101, 99, 102, 98, 103, 97, 104, 96, 105],
            "volume": [10] * 10,
        }
    )

    snapshot = latest_microstructure_snapshot(candles, window=3)

    assert set(snapshot) == set(MICROSTRUCTURE_COLUMNS)
    assert all(isinstance(value, float) for value in snapshot.values())
    assert not math.isnan(snapshot["roll_measure"])


def test_compute_microstructure_requires_price_and_volume_columns():
    with pytest.raises(ValueError, match="missing required columns"):
        compute_microstructure_measures(pd.DataFrame({"close": [1.0]}))


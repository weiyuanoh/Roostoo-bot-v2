import math

import numpy as np
import pandas as pd
import pytest

from bot.forward_ic import (
    add_forward_return,
    compute_horizon_ics,
    compute_univariate_ics,
    forward_return,
    information_coefficient,
    iter_horizon_ics,
    pearson_ic,
    spearman_ic,
)


def test_forward_return_pct_log_and_diff():
    price = pd.Series([100.0, 110.0, 121.0, 133.1])

    pct = forward_return(price, horizon=2, method="pct")
    log_ret = forward_return(price, horizon=2, method="log")
    diff = forward_return(price, horizon=2, method="diff")

    assert pct.iloc[0] == pytest.approx(0.21)
    assert log_ret.iloc[0] == pytest.approx(math.log(1.21))
    assert diff.iloc[0] == pytest.approx(21.0)
    assert pct.tail(2).isna().all()


def test_forward_return_rejects_bad_horizon_and_method():
    with pytest.raises(ValueError, match="horizon"):
        forward_return([1, 2, 3], horizon=0)

    with pytest.raises(ValueError, match="unsupported return method"):
        forward_return([1, 2, 3], horizon=1, method="bad")


def test_add_forward_return_groups_and_sorts_before_shift():
    frame = pd.DataFrame(
        {
            "pair": ["B", "A", "A", "B"],
            "t": [1, 2, 1, 2],
            "close": [20.0, 12.0, 10.0, 25.0],
        }
    )

    out = add_forward_return(
        frame,
        horizon=1,
        group_col="pair",
        sort_col="t",
        target_col="fwd_1",
    )

    a = out[out["pair"] == "A"].sort_values("t")
    b = out[out["pair"] == "B"].sort_values("t")
    assert a["fwd_1"].iloc[0] == pytest.approx(0.2)
    assert b["fwd_1"].iloc[0] == pytest.approx(0.25)
    assert a["fwd_1"].iloc[1] != a["fwd_1"].iloc[1]


def test_pearson_and_spearman_ic():
    feature = pd.Series([1.0, 2.0, 3.0, 4.0])
    target = pd.Series([2.0, 4.0, 6.0, 8.0])

    pearson, n_pearson = pearson_ic(feature, target)
    spearman, n_spearman = spearman_ic(feature, target)

    assert pearson == pytest.approx(1.0)
    assert spearman == pytest.approx(1.0)
    assert n_pearson == 4
    assert n_spearman == 4


def test_spearman_rank_method_is_configurable_for_ties():
    feature = pd.Series([1.0, 1.0, 2.0, 3.0])
    target = pd.Series([1.0, 2.0, 3.0, 4.0])

    average, _ = spearman_ic(feature, target, rank_method="average")
    dense, _ = spearman_ic(feature, target, rank_method="dense")

    assert average != dense


def test_information_coefficient_rejects_unknown_method():
    with pytest.raises(ValueError, match="unsupported IC method"):
        information_coefficient([1, 2, 3], [1, 2, 3], method="kendall")


def test_compute_univariate_ics_returns_feature_method_rows():
    frame = pd.DataFrame(
        {
            "roll_measure": [1.0, 2.0, 3.0, 4.0],
            "vpin": [4.0, 3.0, 2.0, 1.0],
            "forward_return_1": [0.1, 0.2, 0.3, 0.4],
        }
    )

    out = compute_univariate_ics(
        frame,
        feature_cols=["roll_measure", "vpin"],
        target_col="forward_return_1",
        methods=["pearson", "spearman"],
        horizon=1,
    )

    assert len(out) == 4
    assert set(out["feature"]) == {"roll_measure", "vpin"}
    assert set(out["method"]) == {"pearson", "spearman"}
    assert out.loc[out["feature"].eq("roll_measure"), "ic"].max() == pytest.approx(1.0)


def test_compute_univariate_ics_can_group_by_symbol():
    frame = pd.DataFrame(
        {
            "pair": ["A", "A", "A", "B", "B", "B"],
            "feature": [1.0, 2.0, 3.0, 3.0, 2.0, 1.0],
            "target": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
        }
    )

    out = compute_univariate_ics(
        frame,
        feature_cols=["feature"],
        target_col="target",
        methods=["pearson"],
        group_col="pair",
    )

    assert len(out) == 2
    assert out.loc[out["group"].eq("A"), "ic"].iloc[0] == pytest.approx(1.0)
    assert out.loc[out["group"].eq("B"), "ic"].iloc[0] == pytest.approx(-1.0)


def test_compute_horizon_ics_evaluates_multiple_horizons():
    frame = pd.DataFrame(
        {
            "close": [100.0, 101.0, 103.0, 106.0, 110.0],
            "roll_measure": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )

    out = compute_horizon_ics(
        frame,
        horizons=[1, 2],
        feature_cols=["roll_measure"],
        ic_methods=["pearson"],
        min_periods=3,
    )

    assert list(out["horizon"]) == [1, 2]
    assert list(out["method"]) == ["pearson", "pearson"]
    assert out["n"].tolist() == [4, 3]


def test_iter_horizon_ics_is_lazy_generator():
    frame = pd.DataFrame(
        {
            "close": [100.0, 101.0, 103.0, 106.0],
            "feature": [1.0, 2.0, 3.0, 4.0],
        }
    )

    generated = iter_horizon_ics(
        frame,
        horizons=[1],
        feature_cols=["feature"],
        ic_methods=["pearson"],
    )

    assert not isinstance(generated, list)
    assert next(generated).feature == "feature"


def test_constant_series_ic_is_nan_with_valid_n():
    ic, n = pearson_ic([1, 1, 1], [1, 2, 3])

    assert np.isnan(ic)
    assert n == 3


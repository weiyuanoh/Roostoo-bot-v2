"""Alpha feature computation for live trading and research.

The features here are deliberately simple and auditable. They are computed from
closed bars only, use grouped pair-wise rolling windows, and avoid future data.
Research notebooks can combine these features with forward-return targets; live
trading can compute the same columns on an in-memory candle buffer.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


BASE_FEATURE_PATTERNS = (
    "log_return",
    "realized_vol",
    "vol_adj_momentum",
    "trend_persistence",
    "breakout_quality",
    "relative_strength",
    "residual_momentum",
    "momentum_low_vpin",
    "momentum_low_roll_impact",
)


def log_return(price: pd.Series, horizon: int = 1) -> pd.Series:
    """Log return over `horizon` previous bars."""
    if horizon < 1:
        raise ValueError("horizon must be at least 1")
    price = _series(price, "price")
    ratio = _safe_divide(price, price.shift(horizon)).replace(0, np.nan)
    return np.log(ratio).replace([np.inf, -np.inf], np.nan)


def realized_volatility(price: pd.Series, window: int) -> pd.Series:
    """Rolling standard deviation of one-bar log returns."""
    if window < 2:
        raise ValueError("window must be at least 2")
    returns = log_return(price, 1)
    return returns.rolling(window=window, min_periods=window).std()


def vol_adjusted_momentum(price: pd.Series, window: int) -> pd.Series:
    """Window log return divided by window realized volatility."""
    return _safe_divide(log_return(price, window), realized_volatility(price, window))


def trend_persistence(price: pd.Series, window: int) -> pd.Series:
    """Rolling directional persistence in [-1, 1]."""
    returns = log_return(price, 1)
    positive_fraction = (returns > 0).astype(float).rolling(
        window=window,
        min_periods=window,
    ).mean()
    return (2.0 * positive_fraction) - 1.0


def breakout_quality(
    close: pd.Series,
    high: pd.Series,
    volume: pd.Series,
    window: int,
) -> pd.Series:
    """Breakout above prior rolling high, scaled by volume confirmation.

    The rolling high and median volume are shifted by one bar, so the feature
    measures the current close/volume against the prior window only.
    """
    if window < 2:
        raise ValueError("window must be at least 2")
    close = _series(close, "close")
    high = _series(high, "high")
    volume = _series(volume, "volume")
    prior_high = high.rolling(window=window, min_periods=window).max().shift(1)
    prior_median_volume = volume.rolling(window=window, min_periods=window).median().shift(1)
    breakout = (_safe_divide(close, prior_high) - 1.0).clip(lower=0.0)
    volume_ratio = _safe_divide(volume, prior_median_volume)
    return breakout * volume_ratio


def alpha_feature_columns(windows: Iterable[int]) -> list[str]:
    """Return expected alpha feature columns for `windows`."""
    columns: list[str] = []
    for window in windows:
        for pattern in BASE_FEATURE_PATTERNS:
            columns.append(f"{pattern}_{window}")
    return columns


def add_alpha_features(
    frame: pd.DataFrame,
    windows: Iterable[int] = (6, 12, 24),
    pair_col: str = "pair",
    time_col: str = "open_time",
    price_col: str = "close",
    high_col: str = "high",
    volume_col: str = "volume",
    vpin_col: str = "vpin",
    roll_impact_col: str = "roll_impact",
) -> pd.DataFrame:
    """Return `frame` with candidate alpha feature columns appended.

    Features added per window:
    - `log_return_W`
    - `realized_vol_W`
    - `vol_adj_momentum_W`
    - `trend_persistence_W`
    - `breakout_quality_W`
    - `relative_strength_W`: cross-sectional percentile rank of `log_return_W`
    - `residual_momentum_W`: `log_return_W` minus timestamp mean `log_return_W`
    - `momentum_low_vpin_W`: vol-adjusted momentum scaled by low VPIN rank
    - `momentum_low_roll_impact_W`: vol-adjusted momentum scaled by low Roll impact rank
    """
    windows = tuple(windows)
    if not windows:
        raise ValueError("windows must not be empty")
    if any(window < 2 for window in windows):
        raise ValueError("all windows must be at least 2")
    _require_columns(frame, [pair_col, time_col, price_col, high_col, volume_col])

    out = frame.copy()
    out = out.sort_values([pair_col, time_col]).reset_index(drop=True)
    grouped = out.groupby(pair_col, sort=False, group_keys=False)

    for window in windows:
        out[f"log_return_{window}"] = grouped[price_col].transform(
            lambda values, w=window: log_return(values, w)
        )
        out[f"realized_vol_{window}"] = grouped[price_col].transform(
            lambda values, w=window: realized_volatility(values, w)
        )
        out[f"vol_adj_momentum_{window}"] = grouped[price_col].transform(
            lambda values, w=window: vol_adjusted_momentum(values, w)
        )
        out[f"trend_persistence_{window}"] = grouped[price_col].transform(
            lambda values, w=window: trend_persistence(values, w)
        )
        out[f"breakout_quality_{window}"] = _grouped_breakout_quality(
            grouped,
            price_col=price_col,
            high_col=high_col,
            volume_col=volume_col,
            window=window,
        )

        momentum_col = f"log_return_{window}"
        out[f"relative_strength_{window}"] = _cross_sectional_rank(
            out,
            value_col=momentum_col,
            time_col=time_col,
        )
        out[f"residual_momentum_{window}"] = (
            out[momentum_col] - out.groupby(time_col, sort=False)[momentum_col].transform("mean")
        )

        out[f"momentum_low_vpin_{window}"] = _microstructure_adjusted_momentum(
            out,
            momentum_col=f"vol_adj_momentum_{window}",
            risk_col=vpin_col,
            time_col=time_col,
        )
        out[f"momentum_low_roll_impact_{window}"] = _microstructure_adjusted_momentum(
            out,
            momentum_col=f"vol_adj_momentum_{window}",
            risk_col=roll_impact_col,
            time_col=time_col,
        )

    return out


def _cross_sectional_rank(
    frame: pd.DataFrame,
    value_col: str,
    time_col: str,
) -> pd.Series:
    return frame.groupby(time_col, sort=False)[value_col].rank(method="average", pct=True)


def _microstructure_adjusted_momentum(
    frame: pd.DataFrame,
    momentum_col: str,
    risk_col: str,
    time_col: str,
) -> pd.Series:
    if risk_col not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    risk_rank = frame.groupby(time_col, sort=False)[risk_col].rank(method="average", pct=True)
    low_risk_score = 1.0 - risk_rank
    return frame[momentum_col] * low_risk_score


def _grouped_breakout_quality(
    grouped: pd.core.groupby.DataFrameGroupBy,
    price_col: str,
    high_col: str,
    volume_col: str,
    window: int,
) -> pd.Series:
    parts = [
        breakout_quality(group[price_col], group[high_col], group[volume_col], window)
        for _, group in grouped
    ]
    if not parts:
        return pd.Series(dtype=float)
    return pd.concat(parts).sort_index()


def _series(values: pd.Series | np.ndarray | list[float], name: str) -> pd.Series:
    if isinstance(values, pd.Series):
        return values.astype(float)
    return pd.Series(values, dtype=float, name=name)


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    result = numerator.astype(float) / denominator.astype(float).replace(0, np.nan)
    return result.replace([np.inf, -np.inf], np.nan)


def _require_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"missing required columns: {', '.join(missing)}")

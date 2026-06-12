"""Market microstructure measures from Easley et al. crypto market study.

The paper computes five features from one-minute Binance bars over lookback
windows such as 50 and 100 bars:

- Roll measure
- Roll impact
- Amihud measure
- Kyle's lambda
- VPIN

This module works on the candle CSVs produced by `run.py collect`. Binance
`volume` is base-asset volume, so dollar volume defaults to `close * volume`.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


MICROSTRUCTURE_COLUMNS = (
    "roll_measure",
    "roll_impact",
    "amihud",
    "kyle_lambda",
    "vpin",
)


def roll_measure(price: pd.Series, window: int) -> pd.Series:
    """Roll measure: 2 * sqrt(abs(cov(delta_p_t, delta_p_t-1)))."""
    price = _series(price, "price")
    delta = price.diff()
    covariance = delta.rolling(window=window, min_periods=window).cov(delta.shift(1))
    return 2.0 * np.sqrt(covariance.abs())


def roll_impact(
    price: pd.Series,
    volume: pd.Series,
    window: int,
    dollar_volume: pd.Series | None = None,
) -> pd.Series:
    """Roll impact: Roll measure divided by current-bar dollar volume."""
    dollars = _dollar_volume(price, volume, dollar_volume)
    return _safe_divide(roll_measure(price, window), dollars)


def amihud_measure(
    price: pd.Series,
    volume: pd.Series,
    window: int,
    dollar_volume: pd.Series | None = None,
) -> pd.Series:
    """Amihud illiquidity: rolling mean of abs(return) / dollar volume."""
    price = _series(price, "price")
    dollars = _dollar_volume(price, volume, dollar_volume)
    per_bar = _safe_divide(price.pct_change().abs(), dollars)
    return per_bar.rolling(window=window, min_periods=window).mean()


def kyle_lambda(price: pd.Series, volume: pd.Series, window: int) -> pd.Series:
    """Kyle's lambda: price change over W bars divided by signed volume."""
    price = _series(price, "price")
    volume = _series(volume, "volume")
    signed_volume = np.sign(price.diff()) * volume
    numerator = price - price.shift(window)
    denominator = signed_volume.rolling(window=window + 1, min_periods=window + 1).sum()
    return _safe_divide(numerator, denominator)


def vpin(price: pd.Series, volume: pd.Series, window: int) -> pd.Series:
    """Volume-synchronized probability of informed trading proxy.

    Bulk volume classification follows the paper:
    buy_volume_i = volume_i * Phi(delta_p_i / sigma_delta_p_i)
    sell_volume_i = volume_i - buy_volume_i
    VPIN_t = rolling mean(abs(sell_volume_i - buy_volume_i) / volume_i)
    """
    price = _series(price, "price")
    volume = _series(volume, "volume")
    delta = price.diff()
    sigma = delta.rolling(window=window, min_periods=window).std()
    z_score = _safe_divide(delta, sigma)
    buy_volume = volume * _normal_cdf(z_score)
    sell_volume = volume - buy_volume
    imbalance = _safe_divide((sell_volume - buy_volume).abs(), volume)
    return imbalance.rolling(window=window, min_periods=window).mean()


def compute_microstructure_measures(
    candles: pd.DataFrame,
    window: int = 50,
    price_col: str = "close",
    volume_col: str = "volume",
    dollar_volume_col: str | None = None,
) -> pd.DataFrame:
    """Return a copy of `candles` with Easley microstructure columns appended."""
    if window < 2:
        raise ValueError("window must be at least 2")
    _require_columns(candles, [price_col, volume_col])

    out = candles.copy()
    price = out[price_col].astype(float)
    volume = out[volume_col].astype(float)
    dollars = None
    if dollar_volume_col:
        _require_columns(out, [dollar_volume_col])
        dollars = out[dollar_volume_col].astype(float)

    out["roll_measure"] = roll_measure(price, window)
    out["roll_impact"] = roll_impact(price, volume, window, dollars)
    out["amihud"] = amihud_measure(price, volume, window, dollars)
    out["kyle_lambda"] = kyle_lambda(price, volume, window)
    out["vpin"] = vpin(price, volume, window)
    return out


def latest_microstructure_snapshot(
    candles: pd.DataFrame,
    window: int = 50,
    price_col: str = "close",
    volume_col: str = "volume",
    dollar_volume_col: str | None = None,
) -> dict[str, float]:
    """Return each measure's latest non-null value as a plain dict."""
    features = compute_microstructure_measures(
        candles,
        window=window,
        price_col=price_col,
        volume_col=volume_col,
        dollar_volume_col=dollar_volume_col,
    )
    snapshot: dict[str, float] = {}
    for column in MICROSTRUCTURE_COLUMNS:
        values = features[column].dropna()
        snapshot[column] = float(values.iloc[-1]) if not values.empty else float("nan")
    return snapshot


def _series(values: pd.Series | np.ndarray | list[float], name: str) -> pd.Series:
    if isinstance(values, pd.Series):
        return values.astype(float)
    return pd.Series(values, dtype=float, name=name)


def _dollar_volume(
    price: pd.Series,
    volume: pd.Series,
    dollar_volume: pd.Series | None,
) -> pd.Series:
    if dollar_volume is not None:
        return _series(dollar_volume, "dollar_volume")
    return _series(price, "price") * _series(volume, "volume")


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    result = numerator.astype(float) / denominator.astype(float).replace(0, np.nan)
    return result.replace([np.inf, -np.inf], np.nan)


def _normal_cdf(values: pd.Series) -> pd.Series:
    cdf = values.map(
        lambda value: np.nan
        if pd.isna(value)
        else 0.5 * (1.0 + math.erf(float(value) / math.sqrt(2.0)))
    )
    return cdf.astype(float)


def _require_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"missing required columns: {', '.join(missing)}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute Easley market microstructure measures from candle CSV data"
    )
    parser.add_argument("input", help="Input candle CSV path")
    parser.add_argument("--output", help="Output CSV path. Defaults to stdout.")
    parser.add_argument("--window", type=int, default=50, help="Lookback bars")
    parser.add_argument("--price-col", default="close", help="Price column")
    parser.add_argument("--volume-col", default="volume", help="Base volume column")
    parser.add_argument(
        "--dollar-volume-col",
        help="Optional precomputed dollar-volume column",
    )
    args = parser.parse_args()

    candles = pd.read_csv(args.input)
    features = compute_microstructure_measures(
        candles,
        window=args.window,
        price_col=args.price_col,
        volume_col=args.volume_col,
        dollar_volume_col=args.dollar_volume_col,
    )

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        features.to_csv(output, index=False)
        print(output)
    else:
        print(features.to_csv(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

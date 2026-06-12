"""Forward returns and univariate information coefficients.

This module is intentionally strategy-agnostic:

- horizons are bar counts, so 50 one-minute bars and 50 hourly bars are both
  represented as `horizon=50` on the corresponding input data.
- rank methods are configurable for Spearman IC.
- IC computation can run pooled over a frame or independently by a grouping
  column such as `pair`, `symbol`, or `open_time`.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


SUPPORTED_RETURN_METHODS = ("pct", "log", "diff")
SUPPORTED_IC_METHODS = ("pearson", "spearman")


@dataclass(frozen=True)
class ICResult:
    """One univariate IC measurement."""

    feature: str
    target: str
    method: str
    ic: float
    n: int
    horizon: int | None = None
    group: str | int | float | None = None
    rank_method: str | None = None


def forward_return(
    price: pd.Series | Sequence[float] | np.ndarray,
    horizon: int,
    method: str = "pct",
) -> pd.Series:
    """Compute forward return over `horizon` future bars.

    `method` options:
    - `pct`: price[t + h] / price[t] - 1
    - `log`: log(price[t + h] / price[t])
    - `diff`: price[t + h] - price[t]
    """
    if horizon < 1:
        raise ValueError("horizon must be at least 1")
    if method not in SUPPORTED_RETURN_METHODS:
        raise ValueError(f"unsupported return method: {method}")

    price_series = _series(price, "price")
    future_price = price_series.shift(-horizon)

    if method == "pct":
        return _safe_divide(future_price, price_series) - 1.0
    if method == "log":
        ratio = _safe_divide(future_price, price_series)
        return np.log(ratio).replace([np.inf, -np.inf], np.nan)
    return future_price - price_series


def add_forward_return(
    frame: pd.DataFrame,
    horizon: int,
    price_col: str = "close",
    method: str = "pct",
    target_col: str | None = None,
    group_col: str | None = None,
    sort_col: str | None = None,
) -> pd.DataFrame:
    """Return a copy of `frame` with one forward-return target column appended."""
    _require_columns(frame, [price_col])
    if group_col:
        _require_columns(frame, [group_col])
    if sort_col:
        _require_columns(frame, [sort_col])

    out = frame.copy()
    if sort_col:
        sort_cols = [group_col, sort_col] if group_col else [sort_col]
        out = out.sort_values(sort_cols).copy()

    name = target_col or f"forward_return_{horizon}"
    if group_col:
        out[name] = out.groupby(group_col, sort=False)[price_col].transform(
            lambda values: forward_return(values, horizon, method)
        )
    else:
        out[name] = forward_return(out[price_col], horizon, method)
    return out


def pearson_ic(
    feature: pd.Series | Sequence[float] | np.ndarray,
    target: pd.Series | Sequence[float] | np.ndarray,
    min_periods: int = 3,
) -> tuple[float, int]:
    """Return Pearson IC and number of valid observations."""
    valid = _valid_pair_frame(feature, target)
    n = len(valid)
    if n < min_periods:
        return float("nan"), n
    if valid["feature"].nunique(dropna=True) < 2 or valid["target"].nunique(dropna=True) < 2:
        return float("nan"), n
    return float(valid["feature"].corr(valid["target"], method="pearson")), n


def spearman_ic(
    feature: pd.Series | Sequence[float] | np.ndarray,
    target: pd.Series | Sequence[float] | np.ndarray,
    rank_method: str = "average",
    min_periods: int = 3,
) -> tuple[float, int]:
    """Return Spearman IC and number of valid observations.

    Spearman IC is implemented as Pearson correlation of ranked feature and
    target values so callers can swap pandas rank tie-handling methods.
    """
    valid = _valid_pair_frame(feature, target)
    n = len(valid)
    if n < min_periods:
        return float("nan"), n
    ranked_feature = valid["feature"].rank(method=rank_method)
    ranked_target = valid["target"].rank(method=rank_method)
    return pearson_ic(ranked_feature, ranked_target, min_periods=min_periods)


def information_coefficient(
    feature: pd.Series | Sequence[float] | np.ndarray,
    target: pd.Series | Sequence[float] | np.ndarray,
    method: str = "spearman",
    rank_method: str = "average",
    min_periods: int = 3,
) -> tuple[float, int]:
    """Dispatch to Pearson or Spearman IC."""
    if method == "pearson":
        return pearson_ic(feature, target, min_periods=min_periods)
    if method == "spearman":
        return spearman_ic(
            feature,
            target,
            rank_method=rank_method,
            min_periods=min_periods,
        )
    raise ValueError(f"unsupported IC method: {method}")


def iter_univariate_ics(
    frame: pd.DataFrame,
    feature_cols: Iterable[str],
    target_col: str,
    methods: Iterable[str] = SUPPORTED_IC_METHODS,
    horizon: int | None = None,
    rank_method: str = "average",
    min_periods: int = 3,
    group_col: str | None = None,
) -> Iterator[ICResult]:
    """Yield univariate IC results for each feature/method/group combination."""
    feature_list = list(feature_cols)
    _require_columns(frame, [*feature_list, target_col])
    if group_col:
        _require_columns(frame, [group_col])
        groups = frame.groupby(group_col, sort=False)
    else:
        groups = [(None, frame)]

    for group, group_frame in groups:
        for feature in feature_list:
            for method in methods:
                ic, n = information_coefficient(
                    group_frame[feature],
                    group_frame[target_col],
                    method=method,
                    rank_method=rank_method,
                    min_periods=min_periods,
                )
                yield ICResult(
                    feature=feature,
                    target=target_col,
                    method=method,
                    ic=ic,
                    n=n,
                    horizon=horizon,
                    group=group,
                    rank_method=rank_method if method == "spearman" else None,
                )


def compute_univariate_ics(
    frame: pd.DataFrame,
    feature_cols: Iterable[str],
    target_col: str,
    methods: Iterable[str] = SUPPORTED_IC_METHODS,
    horizon: int | None = None,
    rank_method: str = "average",
    min_periods: int = 3,
    group_col: str | None = None,
) -> pd.DataFrame:
    """Return univariate IC results as a DataFrame."""
    return pd.DataFrame(
        asdict(result)
        for result in iter_univariate_ics(
            frame,
            feature_cols,
            target_col,
            methods=methods,
            horizon=horizon,
            rank_method=rank_method,
            min_periods=min_periods,
            group_col=group_col,
        )
    )


def iter_horizon_ics(
    frame: pd.DataFrame,
    horizons: Iterable[int],
    feature_cols: Iterable[str],
    price_col: str = "close",
    return_method: str = "pct",
    ic_methods: Iterable[str] = SUPPORTED_IC_METHODS,
    rank_method: str = "average",
    min_periods: int = 3,
    group_col: str | None = None,
    sort_col: str | None = None,
) -> Iterator[ICResult]:
    """Generate ICs for multiple forward-return horizons."""
    feature_list = list(feature_cols)
    for horizon in horizons:
        target_col = f"forward_return_{horizon}"
        with_target = add_forward_return(
            frame,
            horizon=horizon,
            price_col=price_col,
            method=return_method,
            target_col=target_col,
            group_col=group_col,
            sort_col=sort_col,
        )
        yield from iter_univariate_ics(
            with_target,
            feature_list,
            target_col,
            methods=ic_methods,
            horizon=horizon,
            rank_method=rank_method,
            min_periods=min_periods,
            group_col=group_col,
        )


def compute_horizon_ics(
    frame: pd.DataFrame,
    horizons: Iterable[int],
    feature_cols: Iterable[str],
    price_col: str = "close",
    return_method: str = "pct",
    ic_methods: Iterable[str] = SUPPORTED_IC_METHODS,
    rank_method: str = "average",
    min_periods: int = 3,
    group_col: str | None = None,
    sort_col: str | None = None,
) -> pd.DataFrame:
    """Return ICs for multiple horizons as a DataFrame."""
    return pd.DataFrame(
        asdict(result)
        for result in iter_horizon_ics(
            frame,
            horizons,
            feature_cols,
            price_col=price_col,
            return_method=return_method,
            ic_methods=ic_methods,
            rank_method=rank_method,
            min_periods=min_periods,
            group_col=group_col,
            sort_col=sort_col,
        )
    )


def _series(values: pd.Series | Sequence[float] | np.ndarray, name: str) -> pd.Series:
    if isinstance(values, pd.Series):
        return values.astype(float)
    return pd.Series(values, dtype=float, name=name)


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    result = numerator.astype(float) / denominator.astype(float).replace(0, np.nan)
    return result.replace([np.inf, -np.inf], np.nan)


def _valid_pair_frame(
    feature: pd.Series | Sequence[float] | np.ndarray,
    target: pd.Series | Sequence[float] | np.ndarray,
) -> pd.DataFrame:
    paired = pd.DataFrame(
        {
            "feature": _series(feature, "feature"),
            "target": _series(target, "target"),
        }
    )
    paired = paired.replace([np.inf, -np.inf], np.nan)
    return paired.dropna()


def _require_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"missing required columns: {', '.join(missing)}")


def _parse_csv(raw: str, cast=str) -> list:
    return [cast(value.strip()) for value in raw.split(",") if value.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute forward returns and univariate feature ICs"
    )
    parser.add_argument("input", help="Input CSV path")
    parser.add_argument("--output", help="Output IC summary CSV path. Defaults to stdout.")
    parser.add_argument(
        "--features",
        required=True,
        help="Comma-separated feature columns, e.g. roll_measure,vpin",
    )
    parser.add_argument(
        "--horizons",
        default="1",
        help="Comma-separated forward horizons in bars, e.g. 10,50,100",
    )
    parser.add_argument("--price-col", default="close", help="Price column")
    parser.add_argument(
        "--return-method",
        choices=SUPPORTED_RETURN_METHODS,
        default="pct",
        help="Forward-return method",
    )
    parser.add_argument(
        "--ic-methods",
        default="pearson,spearman",
        help="Comma-separated IC methods: pearson,spearman",
    )
    parser.add_argument(
        "--rank-method",
        default="average",
        help="pandas rank tie method for Spearman IC",
    )
    parser.add_argument("--min-periods", type=int, default=3)
    parser.add_argument("--group-col", help="Optional group column, e.g. pair")
    parser.add_argument("--sort-col", help="Optional sort column before forward returns")
    args = parser.parse_args()

    frame = pd.read_csv(args.input)
    results = compute_horizon_ics(
        frame,
        horizons=_parse_csv(args.horizons, int),
        feature_cols=_parse_csv(args.features, str),
        price_col=args.price_col,
        return_method=args.return_method,
        ic_methods=_parse_csv(args.ic_methods, str),
        rank_method=args.rank_method,
        min_periods=args.min_periods,
        group_col=args.group_col,
        sort_col=args.sort_col,
    )

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(output, index=False)
        print(output)
    else:
        print(results.to_csv(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


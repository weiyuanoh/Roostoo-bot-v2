"""No-lookahead regime diagnostics for live-style backtest trades."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bot.backtest.ridge_score_portfolio import filter_pairs, parse_pairs
from bot.strategy.ridge import (
    DEFAULT_FEATURE_PATH,
    DEFAULT_RIDGE_ALPHAS,
    make_folds,
    load_model_frame,
    score_frame,
    select_ridge_model,
)

DEFAULT_OUTPUT_DIR = Path("reports/backtests/regime_diagnostics")
DEFAULT_PREFIX = "momentum_only_slip5"
PERSISTENCE_WINDOWS = (3, 6, 12)
POST_EXIT_HORIZONS = (1, 3, 6, 24)
DECISION_TIME_COLUMNS = {
    "entry_rank",
    "entry_score",
    "rank1_score",
    "rank2_score",
    "score_gap_rank1_rank2",
    "score_gap_rank1_median",
    "score_gap_rank1_rank2_trailing_median",
    "pair_top3_count_3h",
    "pair_top3_count_6h",
    "pair_top3_count_12h",
    "pair_rank1_count_3h",
    "pair_rank1_count_6h",
    "pair_rank1_count_12h",
    "pair_return_1h_known",
    "universe_breadth_1h",
    "universe_return_1h",
    "universe_volatility_1h",
    "positions_at_entry",
}


def load_scored_frame(
    feature_path: str | Path,
    *,
    metadata: dict[str, Any],
) -> pd.DataFrame:
    """Reconstruct walk-forward OS scores used by the backtest."""
    horizon = int(metadata.get("horizon", 24))
    model = str(metadata.get("model", "momentum_only"))
    frame = load_model_frame(feature_path, horizon=horizon)
    pairs = _metadata_pairs(metadata)
    frame = filter_pairs(frame, pairs)
    folds = make_folds(
        frame,
        is_months=int(metadata.get("is_months", 4)),
        os_months=int(metadata.get("os_months", 4)),
        step_months=int(metadata.get("step_months", 1)),
    )
    ridge_alphas = tuple(float(value) for value in metadata.get("ridge_alphas", DEFAULT_RIDGE_ALPHAS))
    target_col = f"forward_return_{horizon}"
    parts = []
    for fold in folds.itertuples(index=False):
        train = frame[(frame["timestamp"] >= fold.is_start) & (frame["timestamp"] < fold.is_end)]
        test = frame[(frame["timestamp"] >= fold.os_start) & (frame["timestamp"] < fold.os_end)]
        selection = select_ridge_model(
            train.copy(),
            model,
            target_col=target_col,
            ridge_alphas=ridge_alphas,
        )
        scored = score_frame(test.copy(), selection.terms, selection.beta, score_col="ridge_score")
        scored["fold"] = fold.fold
        parts.append(scored)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True).sort_values(["open_time", "pair"]).reset_index(drop=True)


def add_decision_time_regime_features(scored: pd.DataFrame) -> pd.DataFrame:
    """Add rank, score-gap, persistence, and current/prior-only market features."""
    if scored.empty:
        return scored.copy()
    df = scored.sort_values(["open_time", "pair"]).copy()
    df["entry_rank"] = df.groupby("open_time", observed=True)["ridge_score"].rank(
        method="first",
        ascending=False,
    )
    grouped = df.groupby("open_time", observed=True)
    df["rank1_score"] = grouped["ridge_score"].transform("max")
    df["rank2_score"] = grouped["ridge_score"].transform(_second_largest)
    df["median_score"] = grouped["ridge_score"].transform("median")
    df["score_gap_rank1_rank2"] = df["rank1_score"] - df["rank2_score"]
    df["score_gap_rank1_median"] = df["rank1_score"] - df["median_score"]
    df["rank1_pair"] = df["open_time"].map(_rank1_pairs(df))
    gaps = (
        df[["open_time", "score_gap_rank1_rank2"]]
        .drop_duplicates("open_time")
        .sort_values("open_time")
        .set_index("open_time")["score_gap_rank1_rank2"]
    )
    trailing_gap = gaps.expanding(min_periods=24).median().shift(1)
    df["score_gap_rank1_rank2_trailing_median"] = df["open_time"].map(trailing_gap)

    df["is_top3"] = df["entry_rank"] <= 3
    df["is_rank1"] = df["entry_rank"] == 1
    for window in PERSISTENCE_WINDOWS:
        df[f"pair_top3_count_{window}h"] = (
            df.groupby("pair", observed=True)["is_top3"]
            .transform(lambda values, window=window: values.rolling(window, min_periods=1).sum())
            .astype(float)
        )
        df[f"pair_rank1_count_{window}h"] = (
            df.groupby("pair", observed=True)["is_rank1"]
            .transform(lambda values, window=window: values.rolling(window, min_periods=1).sum())
            .astype(float)
        )

    df["pair_return_1h_known"] = df.groupby("pair", observed=True)["close"].pct_change()
    df["universe_breadth_1h"] = df.groupby("open_time", observed=True)["pair_return_1h_known"].transform(
        lambda values: float((values > 0).mean())
    )
    df["universe_return_1h"] = df.groupby("open_time", observed=True)["pair_return_1h_known"].transform("mean")
    df["universe_volatility_1h"] = df.groupby("open_time", observed=True)["pair_return_1h_known"].transform("std")
    return df


def pair_trade_entries(trades: pd.DataFrame, equity: pd.DataFrame) -> pd.DataFrame:
    """Return BUY rows paired to their eventual SELL outcome where available."""
    if trades.empty:
        return pd.DataFrame()
    buys = trades[trades["side"].eq("BUY")].copy()
    sells = trades[trades["side"].eq("SELL")].copy()
    if buys.empty:
        return buys
    buys = buys.drop(columns=[col for col in ("pnl", "return_pct", "entry_time", "entry_price", "entry_value") if col in buys.columns])
    buys["entry_time"] = pd.to_numeric(buys["timestamp"], errors="coerce").astype("Int64")
    if not sells.empty:
        sells["entry_time_key"] = pd.to_numeric(sells["entry_time"], errors="coerce").astype("Int64")
        sell_cols = [
            "pair",
            "entry_time_key",
            "timestamp",
            "price",
            "reason",
            "pnl",
            "return_pct",
            "fee",
        ]
        paired = buys.merge(
            sells[sell_cols].rename(
                columns={
                    "timestamp": "exit_time",
                    "price": "exit_price",
                    "reason": "exit_reason",
                    "fee": "exit_fee",
                }
            ),
            left_on=["pair", "entry_time"],
            right_on=["pair", "entry_time_key"],
            how="left",
        )
    else:
        paired = buys.copy()
        paired["exit_time"] = np.nan
        paired["exit_price"] = np.nan
        paired["exit_reason"] = np.nan
        paired["pnl"] = np.nan
        paired["return_pct"] = np.nan
    if not equity.empty:
        positions = equity[["open_time", "positions"]].copy()
        positions["open_time"] = pd.to_numeric(positions["open_time"], errors="coerce").astype("Int64")
        paired = paired.merge(
            positions.rename(columns={"open_time": "entry_time", "positions": "positions_at_entry"}),
            on="entry_time",
            how="left",
        )
    return paired


def entry_regime_frame(entries: pd.DataFrame, regime_features: pd.DataFrame) -> pd.DataFrame:
    if entries.empty or regime_features.empty:
        return pd.DataFrame()
    feature_cols = [
        "open_time",
        "pair",
        "timestamp",
        "entry_rank",
        "ridge_score",
        "rank1_pair",
        "rank1_score",
        "rank2_score",
        "median_score",
        "score_gap_rank1_rank2",
        "score_gap_rank1_median",
        "score_gap_rank1_rank2_trailing_median",
        "pair_return_1h_known",
        "universe_breadth_1h",
        "universe_return_1h",
        "universe_volatility_1h",
        *[f"pair_top3_count_{window}h" for window in PERSISTENCE_WINDOWS],
        *[f"pair_rank1_count_{window}h" for window in PERSISTENCE_WINDOWS],
    ]
    out = entries.merge(
        regime_features[feature_cols].rename(
            columns={
                "open_time": "entry_time",
                "timestamp": "entry_timestamp",
                "ridge_score": "entry_score",
            }
        ),
        on=["pair", "entry_time"],
        how="left",
    )
    out["tp_hit"] = out["exit_reason"].eq("gain_threshold")
    out["sl_hit"] = out["exit_reason"].eq("loss_threshold")
    out["fold_end_exit"] = out["exit_reason"].eq("fold_end")
    out["holding_hours"] = (pd.to_numeric(out["exit_time"], errors="coerce") - out["entry_time"].astype(float)) / 3_600_000
    out = add_prior_exit_features(out)
    return out


def add_prior_exit_features(entries: pd.DataFrame) -> pd.DataFrame:
    out = entries.sort_values(["pair", "entry_time"]).copy()
    out["entry_time"] = pd.to_numeric(out["entry_time"], errors="coerce")
    out["exit_time"] = pd.to_numeric(out["exit_time"], errors="coerce")
    sells = out.dropna(subset=["exit_time"]).sort_values(["pair", "exit_time"])
    prior_rows = []
    for entry in out.itertuples(index=False):
        pair_sells = sells[(sells["pair"] == entry.pair) & (sells["exit_time"] < entry.entry_time)]
        if pair_sells.empty:
            prior_rows.append((np.nan, np.nan, np.nan, False))
            continue
        prior = pair_sells.iloc[-1]
        hours = (float(entry.entry_time) - float(prior["exit_time"])) / 3_600_000
        reason = prior.get("exit_reason")
        prior_rows.append((prior["exit_time"], reason, hours, True))
    out[["prior_exit_time", "prior_exit_reason", "hours_since_prior_exit", "is_same_pair_reentry"]] = pd.DataFrame(
        prior_rows,
        index=out.index,
    )
    return out


def post_exit_returns(sells: pd.DataFrame, regime_features: pd.DataFrame, horizons: tuple[int, ...]) -> pd.DataFrame:
    if sells.empty or regime_features.empty:
        return pd.DataFrame()
    price_map = {
        (str(row.pair), int(row.open_time)): float(row.close)
        for row in regime_features[["pair", "open_time", "close"]].dropna().itertuples(index=False)
    }
    rows = []
    for sell in sells.itertuples(index=False):
        exit_time = _optional_int(getattr(sell, "timestamp", None))
        exit_price = _optional_float(getattr(sell, "price", None))
        if exit_time is None or exit_price is None or exit_price <= 0:
            continue
        for horizon in horizons:
            future = price_map.get((str(sell.pair), exit_time + horizon * 3_600_000))
            rows.append(
                {
                    "pair": sell.pair,
                    "exit_time": exit_time,
                    "exit_reason": getattr(sell, "reason", None),
                    "horizon": horizon,
                    "post_exit_return": future / exit_price - 1.0 if future and future > 0 else np.nan,
                }
            )
    return pd.DataFrame(rows)


def build_reports(
    *,
    features: pd.DataFrame,
    trades: pd.DataFrame,
    equity: pd.DataFrame,
    horizons: tuple[int, ...] = POST_EXIT_HORIZONS,
) -> dict[str, pd.DataFrame]:
    regime = add_decision_time_regime_features(features)
    entries = pair_trade_entries(trades, equity)
    entry_regime = entry_regime_frame(entries, regime)
    sells = trades[trades["side"].eq("SELL")].copy() if not trades.empty else pd.DataFrame()
    post_exit = post_exit_returns(sells, regime, horizons)
    same_pair = entry_regime[entry_regime["is_same_pair_reentry"].fillna(False)].copy()
    return {
        "entry_regime": entry_regime,
        "trade_outcome_attribution": _trade_outcome_attribution(entry_regime),
        "rank_persistence": _rank_persistence_report(entry_regime),
        "score_gap_report": _score_gap_report(entry_regime),
        "same_pair_reentry": same_pair,
        "post_exit_returns": post_exit,
        "shadow_regime_filters": _shadow_filter_report(entry_regime),
    }


def write_reports(reports: dict[str, pd.DataFrame], output_dir: Path, prefix: str) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, frame in reports.items():
        path = output_dir / f"{prefix}_{name}.csv"
        frame.to_csv(path, index=False)
        paths.append(path)
    return paths


def _trade_outcome_attribution(entries: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "pair",
        "entry_time",
        "exit_time",
        "exit_reason",
        "entry_rank",
        "entry_score",
        "score_gap_rank1_rank2",
        "pair_top3_count_3h",
        "holding_hours",
        "return_pct",
        "pnl",
        "tp_hit",
        "sl_hit",
    ]
    return entries[[col for col in cols if col in entries.columns]].copy() if not entries.empty else pd.DataFrame(columns=cols)


def _rank_persistence_report(entries: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in [f"pair_top3_count_{window}h" for window in PERSISTENCE_WINDOWS]:
        if entries.empty or col not in entries.columns:
            continue
        for value, group in entries.groupby(col, dropna=False, observed=True):
            rows.append({"metric": col, "bucket": value, **_outcome_summary(group)})
    return pd.DataFrame(rows)


def _score_gap_report(entries: pd.DataFrame) -> pd.DataFrame:
    if entries.empty or "score_gap_rank1_rank2" not in entries.columns:
        return pd.DataFrame()
    out = entries.copy()
    out["score_gap_bucket"] = _safe_qcut(out["score_gap_rank1_rank2"], 4)
    return (
        out.groupby("score_gap_bucket", dropna=False, observed=True)
        .apply(lambda group: pd.Series(_outcome_summary(group)), include_groups=False)
        .reset_index()
    )


def _shadow_filter_report(entries: pd.DataFrame) -> pd.DataFrame:
    if entries.empty:
        return pd.DataFrame()
    filters = {
        "score_gap_top_half": entries["score_gap_rank1_rank2"]
        >= entries["score_gap_rank1_rank2_trailing_median"],
        "rank_persistent_top3": entries["pair_top3_count_3h"] >= 2,
        "momentum_hot": (
            (entries["pair_top3_count_3h"] >= 2)
            & (entries["score_gap_rank1_rank2"] >= entries["score_gap_rank1_rank2_trailing_median"])
            & (entries["universe_breadth_1h"] >= 0.5)
        ),
        "same_pair_reentry_after_tp": (
            (~entries["is_same_pair_reentry"].fillna(False))
            | (
                entries["prior_exit_reason"].eq("gain_threshold")
                & (entries["entry_rank"] == 1)
                & (entries["score_gap_rank1_rank2"] > 0)
            )
        ),
        "avoid_recent_stopped_pair": ~(
            entries["prior_exit_reason"].eq("loss_threshold")
            & (pd.to_numeric(entries["hours_since_prior_exit"], errors="coerce") <= 6)
        ),
    }
    rows = []
    for name, mask in filters.items():
        kept = entries[mask.fillna(False)]
        blocked = entries[~mask.fillna(False)]
        rows.append({"filter": name, "side": "kept", **_outcome_summary(kept)})
        rows.append({"filter": name, "side": "blocked", **_outcome_summary(blocked)})
    return pd.DataFrame(rows)


def _outcome_summary(group: pd.DataFrame) -> dict[str, float | int]:
    returns = pd.to_numeric(group.get("return_pct"), errors="coerce")
    pnl = pd.to_numeric(group.get("pnl"), errors="coerce")
    return {
        "entries": int(len(group)),
        "mean_return": float(returns.mean()) if returns.notna().any() else np.nan,
        "win_rate": float((returns > 0).mean()) if len(group) else np.nan,
        "tp_rate": float(group.get("tp_hit", pd.Series(dtype=bool)).fillna(False).mean()) if len(group) else np.nan,
        "sl_rate": float(group.get("sl_hit", pd.Series(dtype=bool)).fillna(False).mean()) if len(group) else np.nan,
        "total_pnl": float(pnl.fillna(0).sum()) if len(group) else 0.0,
    }


def _metadata_pairs(metadata: dict[str, Any]) -> tuple[str, ...] | None:
    pair_filter = metadata.get("pair_filter")
    if isinstance(pair_filter, list):
        return tuple(str(pair).upper() for pair in pair_filter)
    pair_universe = metadata.get("pair_universe")
    if isinstance(pair_universe, list):
        return tuple(str(pair).upper() for pair in pair_universe)
    return parse_pairs(str(pair_filter)) if isinstance(pair_filter, str) else None


def _rank1_pairs(df: pd.DataFrame) -> dict[int, str]:
    pairs = {}
    for open_time, group in df.groupby("open_time", observed=True):
        ranked = group.sort_values("ridge_score", ascending=False)
        if not ranked.empty:
            pairs[int(open_time)] = str(ranked.iloc[0]["pair"])
    return pairs


def _second_largest(values: pd.Series) -> float:
    clean = values.dropna().sort_values(ascending=False)
    return float(clean.iloc[1]) if len(clean) > 1 else np.nan


def _safe_qcut(values: pd.Series, q: int) -> pd.Series:
    clean = pd.to_numeric(values, errors="coerce")
    try:
        return pd.qcut(clean, q=q, duplicates="drop").astype(str)
    except ValueError:
        return pd.Series(["all"] * len(values), index=values.index)


def _optional_int(value) -> int | None:
    if value in {None, ""} or pd.isna(value):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value) -> float | None:
    if value in {None, ""} or pd.isna(value):
        return None
    return float(value)


def parse_horizons(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def main() -> int:
    parser = argparse.ArgumentParser(description="No-lookahead regime diagnostics for backtest trades")
    parser.add_argument("--features", default=str(DEFAULT_FEATURE_PATH))
    parser.add_argument("--trades", required=True)
    parser.add_argument("--equity", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--horizons", default="1,3,6,24")
    args = parser.parse_args()

    metadata = json.loads(Path(args.metadata).read_text(encoding="utf-8"))
    scored = load_scored_frame(args.features, metadata=metadata)
    trades = pd.read_csv(args.trades)
    equity = pd.read_csv(args.equity)
    reports = build_reports(
        features=scored,
        trades=trades,
        equity=equity,
        horizons=parse_horizons(args.horizons),
    )
    paths = write_reports(reports, Path(args.output_dir), args.prefix)
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

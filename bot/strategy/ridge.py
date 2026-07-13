"""Central ridge momentum + roll-impact strategy implementation.

This module is the source of truth for the ridge score used by notebooks,
portfolio backtests, and live trading.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bot.features import add_alpha_features
from bot.forward_ic import add_forward_return, information_coefficient
from bot.microstructure import compute_microstructure_measures
from bot.strategy.regime import (
    ClusterRegimeGate,
    ClusterGateDecision,
    RegimeDecision,
    RegimeThrottleConfig,
    decide_regime_throttle,
)


DEFAULT_FEATURE_PATH = Path("data/features/live_1h_features_30m.csv")
DEFAULT_HORIZON = 24
DEFAULT_IS_MONTHS = 3
DEFAULT_OS_MONTHS = 2
DEFAULT_STEP_MONTHS = 1
DEFAULT_RIDGE_ALPHAS = (0.1, 1.0, 10.0, 100.0)
DEFAULT_TOP_KS = (3, 5)
DEFAULT_COST_BPS = 20.0
DEFAULT_MIN_ASSETS_PER_TIMESTAMP = 8
DEFAULT_MIN_IC_ASSETS = 8
DEFAULT_MICROSTRUCTURE_WINDOW = 50
DEFAULT_ALPHA_WINDOWS = (6, 12, 24)
DEFAULT_MODEL = "momentum_plus_roll_plus_interaction"

BASE_SIGNAL_SPECS = (
    {
        "name": "momentum",
        "source_col": "vol_adj_momentum_24",
        "transform": "identity",
    },
    {
        "name": "low_roll_impact",
        "source_col": "roll_impact",
        "transform": "negative",
    },
)

MODEL_SPECS = (
    {
        "name": "momentum_only",
        "terms": ("z_momentum",),
    },
    {
        "name": "momentum_plus_roll",
        "terms": ("z_momentum", "z_low_roll_impact"),
    },
    {
        "name": "momentum_roll_interaction",
        "terms": ("z_momentum", "z_momentum_x_low_roll_impact"),
    },
    {
        "name": "momentum_plus_roll_plus_interaction",
        "terms": ("z_momentum", "z_low_roll_impact", "z_momentum_x_low_roll_impact"),
    },
)


@dataclass(frozen=True)
class RidgeSelection:
    model: str
    alpha: float
    terms: tuple[str, ...]
    beta: np.ndarray
    is_mean_spearman: float
    is_spearman_hit_rate: float


@dataclass(frozen=True)
class TradeIntent:
    pair: str
    side: str
    quantity: float = 0.0
    notional_usd: float = 0.0
    score: float | None = None
    reason: str = ""
    price: float | None = None
    cluster_gate_enabled: bool = False
    cluster_gate_allowed: bool | None = None
    cluster_id: int | None = None
    cluster_distance: float | None = None
    cluster_reason: str | None = None


@dataclass(frozen=True)
class CycleIntents:
    exits: list[TradeIntent]
    entries: list[TradeIntent]
    regime: RegimeDecision | None = None
    entry_gate_decisions: list[dict[str, Any]] = field(default_factory=list)


def model_terms(model_name: str) -> tuple[str, ...]:
    for spec in MODEL_SPECS:
        if spec["name"] == model_name:
            return tuple(spec["terms"])
    known = ", ".join(str(spec["name"]) for spec in MODEL_SPECS)
    raise ValueError(f"unknown model {model_name!r}; known models: {known}")


def normalize_time(frame: pd.DataFrame, time_col: str = "open_time") -> pd.DataFrame:
    out = frame.copy()
    if pd.api.types.is_numeric_dtype(out[time_col]):
        out["timestamp"] = pd.to_datetime(out[time_col], unit="ms", utc=True)
    else:
        out["timestamp"] = pd.to_datetime(out[time_col], utc=True)
    return out


def cross_sectional_zscore(
    frame: pd.DataFrame,
    value_col: str,
    time_col: str = "open_time",
) -> pd.Series:
    grouped = frame.groupby(time_col, observed=True)[value_col]
    mean = grouped.transform("mean")
    std = grouped.transform("std").replace(0, np.nan)
    return ((frame[value_col] - mean) / std).replace([np.inf, -np.inf], np.nan)


def add_ridge_signals(frame: pd.DataFrame) -> pd.DataFrame:
    """Append raw/z-scored ridge signal columns to an existing feature frame."""
    df = frame.copy()
    for spec in BASE_SIGNAL_SPECS:
        raw_col = f"raw_{spec['name']}"
        if spec["transform"] == "negative":
            df[raw_col] = -df[spec["source_col"]]
        elif spec["transform"] == "identity":
            df[raw_col] = df[spec["source_col"]]
        else:
            raise ValueError(f"unknown transform: {spec['transform']}")
        df[f"z_{spec['name']}"] = cross_sectional_zscore(df, raw_col)

    df["raw_momentum_x_low_roll_impact"] = df["z_momentum"] * df["z_low_roll_impact"]
    df["z_momentum_x_low_roll_impact"] = cross_sectional_zscore(
        df,
        "raw_momentum_x_low_roll_impact",
    )
    return df


def build_feature_frame(
    candles: pd.DataFrame,
    *,
    microstructure_window: int = DEFAULT_MICROSTRUCTURE_WINDOW,
    alpha_windows: tuple[int, ...] = DEFAULT_ALPHA_WINDOWS,
    pair_col: str = "pair",
    time_col: str = "open_time",
) -> pd.DataFrame:
    """Compute microstructure, alpha, and ridge signal columns from candles."""
    required = {pair_col, time_col, "open", "high", "low", "close", "volume"}
    missing = sorted(required - set(candles.columns))
    if missing:
        raise ValueError(f"missing required columns: {', '.join(missing)}")

    parts = []
    ordered = candles.sort_values([pair_col, time_col]).copy()
    for _, group in ordered.groupby(pair_col, sort=False):
        parts.append(
            compute_microstructure_measures(
                group,
                window=microstructure_window,
                price_col="close",
                volume_col="volume",
            )
        )
    micro = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    features = add_alpha_features(
        micro,
        windows=alpha_windows,
        pair_col=pair_col,
        time_col=time_col,
        price_col="close",
        high_col="high",
        volume_col="volume",
        vpin_col="vpin",
        roll_impact_col="roll_impact",
    )
    return add_ridge_signals(normalize_time(features, time_col=time_col))


def add_training_target(
    frame: pd.DataFrame,
    horizon: int = DEFAULT_HORIZON,
    *,
    price_col: str = "close",
    target_col: str | None = None,
    group_col: str = "pair",
    sort_col: str = "open_time",
) -> pd.DataFrame:
    """Append forward-return target and cross-sectional target z-score."""
    df = normalize_time(frame, time_col=sort_col) if "timestamp" not in frame.columns else frame.copy()
    target = target_col or f"forward_return_{horizon}"
    if target not in df.columns:
        df = add_forward_return(
            df,
            horizon=horizon,
            price_col=price_col,
            target_col=target,
            group_col=group_col,
            sort_col=sort_col,
        )
    df["target_z"] = cross_sectional_zscore(df, target, time_col=sort_col)
    return df


def load_model_frame(
    feature_path: str | Path = DEFAULT_FEATURE_PATH,
    horizon: int = DEFAULT_HORIZON,
) -> pd.DataFrame:
    df = pd.read_csv(feature_path)
    df = normalize_time(df)
    df = add_ridge_signals(df)
    df = add_training_target(df, horizon=horizon)
    return df.sort_values(["pair", "timestamp"]).reset_index(drop=True)


def bar_delta(frame: pd.DataFrame) -> pd.Timedelta:
    """Infer the median bar spacing from a timestamped model frame."""
    times = pd.Series(pd.to_datetime(frame["timestamp"], utc=True).drop_duplicates().sort_values())
    if len(times) < 2:
        return pd.Timedelta(0)
    return pd.Timedelta(times.diff().dropna().median())


def labeled_training_slice(
    frame: pd.DataFrame,
    *,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    horizon: int,
) -> pd.DataFrame:
    """Select training rows whose forward labels are known before train_end."""
    cutoff = pd.Timestamp(train_end) - bar_delta(frame) * horizon
    mask = (
        (frame["timestamp"] >= pd.Timestamp(train_start))
        & (frame["timestamp"] < pd.Timestamp(train_end))
        & (frame["timestamp"] <= cutoff)
    )
    return frame[mask].copy()


def make_folds(
    frame: pd.DataFrame,
    is_months: int = DEFAULT_IS_MONTHS,
    os_months: int = DEFAULT_OS_MONTHS,
    step_months: int = DEFAULT_STEP_MONTHS,
) -> pd.DataFrame:
    start = pd.Timestamp(frame["timestamp"].min())
    end = pd.Timestamp(frame["timestamp"].max())
    rows = []
    fold = 0
    cursor = start
    while True:
        is_start = cursor
        is_end = is_start + pd.DateOffset(months=is_months)
        os_start = is_end
        os_end = os_start + pd.DateOffset(months=os_months)
        if os_end > end:
            break
        rows.append(
            {
                "fold": fold,
                "is_start": is_start,
                "is_end": is_end,
                "os_start": os_start,
                "os_end": os_end,
            }
        )
        fold += 1
        cursor = cursor + pd.DateOffset(months=step_months)
    if not rows:
        raise ValueError(
            f"not enough data for {is_months}m IS + {os_months}m OS. "
            f"Available: {start} to {end}."
        )
    return pd.DataFrame(rows)


def fit_ridge(train: pd.DataFrame, terms: tuple[str, ...], alpha: float) -> np.ndarray:
    valid = train.dropna(subset=[*terms, "target_z"])
    x = valid.loc[:, terms].to_numpy(dtype=float)
    y = valid["target_z"].to_numpy(dtype=float)
    penalty = alpha * np.eye(x.shape[1])
    return np.linalg.pinv(x.T @ x + penalty) @ x.T @ y


def score_frame(
    frame: pd.DataFrame,
    terms: tuple[str, ...],
    beta: np.ndarray,
    score_col: str = "ridge_score",
) -> pd.DataFrame:
    out = frame.copy()
    valid = out.loc[:, terms].notna().all(axis=1)
    out[score_col] = np.nan
    out.loc[valid, score_col] = out.loc[valid, terms].to_numpy(dtype=float) @ beta
    return out


def latest_scores(
    frame: pd.DataFrame,
    selection: RidgeSelection,
    *,
    score_col: str = "ridge_score",
) -> pd.DataFrame:
    scored = score_frame(frame, selection.terms, selection.beta, score_col=score_col)
    latest_time = scored["open_time"].max()
    latest = scored[scored["open_time"].eq(latest_time)].copy()
    return latest.sort_values(score_col, ascending=False, na_position="last")


def ic_by_time(
    frame: pd.DataFrame,
    score_col: str,
    target_col: str,
    sample: str,
    min_assets: int = DEFAULT_MIN_IC_ASSETS,
) -> pd.DataFrame:
    rows = []
    for open_time, group in frame.dropna(subset=[score_col, target_col]).groupby(
        "open_time",
        observed=True,
    ):
        if len(group) < min_assets:
            continue
        if group[score_col].nunique(dropna=True) < 2 or group[target_col].nunique(dropna=True) < 2:
            continue
        pearson, pearson_n = information_coefficient(
            group[score_col],
            group[target_col],
            method="pearson",
            min_periods=min_assets,
        )
        spearman, spearman_n = information_coefficient(
            group[score_col],
            group[target_col],
            method="spearman",
            min_periods=min_assets,
        )
        rows.append(
            {
                "sample": sample,
                "open_time": open_time,
                "timestamp": group["timestamp"].iloc[0],
                "pearson": pearson,
                "pearson_n": pearson_n,
                "spearman": spearman,
                "spearman_n": spearman_n,
            }
        )
    return pd.DataFrame(rows)


def summarize_ic(ic: pd.DataFrame) -> dict[str, float | int]:
    if ic.empty:
        return {
            "periods": 0,
            "mean_pearson": np.nan,
            "mean_spearman": np.nan,
            "median_spearman": np.nan,
            "spearman_hit_rate": np.nan,
        }
    return {
        "periods": int(len(ic)),
        "mean_pearson": float(ic["pearson"].mean()),
        "mean_spearman": float(ic["spearman"].mean()),
        "median_spearman": float(ic["spearman"].median()),
        "spearman_hit_rate": float((ic["spearman"] > 0).mean()),
    }


def select_ridge_model(
    train: pd.DataFrame,
    model_name: str,
    target_col: str,
    ridge_alphas: tuple[float, ...] = DEFAULT_RIDGE_ALPHAS,
    min_assets: int = DEFAULT_MIN_IC_ASSETS,
) -> RidgeSelection:
    terms = model_terms(model_name)
    candidates: list[tuple[dict[str, float | int], np.ndarray, float]] = []
    for alpha in ridge_alphas:
        beta = fit_ridge(train, terms, alpha)
        scored = score_frame(train, terms, beta)
        ic = summarize_ic(
            ic_by_time(
                scored,
                "ridge_score",
                target_col,
                "IS",
                min_assets=min_assets,
            )
        )
        candidates.append((ic, beta, alpha))

    selected_ic, selected_beta, selected_alpha = sorted(
        candidates,
        key=lambda item: (
            item[0]["mean_spearman"],
            item[0]["spearman_hit_rate"],
            -item[2],
        ),
        reverse=True,
    )[0]
    return RidgeSelection(
        model=model_name,
        alpha=float(selected_alpha),
        terms=terms,
        beta=selected_beta,
        is_mean_spearman=float(selected_ic["mean_spearman"]),
        is_spearman_hit_rate=float(selected_ic["spearman_hit_rate"]),
    )


def entry_intents(
    scores: pd.DataFrame,
    held_pairs: set[str],
    portfolio_value: float,
    available_cash: float,
    *,
    position_fraction: float,
    max_positions: int,
    top_k: int | None = None,
    max_new_entries: int | None = None,
    price_col: str = "close",
    score_col: str = "ridge_score",
) -> list[TradeIntent]:
    entries, _ = entry_intents_with_gate(
        scores,
        held_pairs,
        portfolio_value,
        available_cash,
        position_fraction=position_fraction,
        max_positions=max_positions,
        top_k=top_k,
        max_new_entries=max_new_entries,
        price_col=price_col,
        score_col=score_col,
    )
    return entries


def entry_intents_with_gate(
    scores: pd.DataFrame,
    held_pairs: set[str],
    portfolio_value: float,
    available_cash: float,
    *,
    position_fraction: float,
    max_positions: int,
    top_k: int | None = None,
    max_new_entries: int | None = None,
    price_col: str = "close",
    score_col: str = "ridge_score",
    cluster_gate: ClusterRegimeGate | None = None,
) -> tuple[list[TradeIntent], list[dict[str, Any]]]:
    if portfolio_value <= 0 or position_fraction <= 0 or max_positions <= 0:
        return [], []
    if top_k is not None and top_k <= 0:
        return [], []
    if max_new_entries is not None and max_new_entries <= 0:
        return [], []
    slots = max_positions - len(held_pairs)
    if slots <= 0 or available_cash <= 0:
        return [], []
    entry_limit = slots
    if max_new_entries is not None:
        entry_limit = min(entry_limit, max_new_entries)
    intents = []
    gate_decisions = []
    ranked = scores.dropna(subset=[score_col, price_col]).sort_values(score_col, ascending=False)
    if top_k is not None:
        ranked = ranked.head(top_k)
    for row in ranked.itertuples(index=False):
        if len(intents) >= entry_limit:
            break
        pair = str(getattr(row, "pair"))
        price = float(getattr(row, price_col))
        score = float(getattr(row, score_col))
        if pair in held_pairs or price <= 0:
            continue
        decision: ClusterGateDecision | None = None
        if cluster_gate is not None:
            row_series = pd.Series(row._asdict())
            decision = cluster_gate.decide(
                row_series,
                positions_at_entry=len(held_pairs) + len(intents),
            )
            gate_decisions.append(
                {
                    "pair": pair,
                    "score": score,
                    "cluster_gate_allowed": decision.allowed,
                    "cluster_id": decision.cluster_id,
                    "cluster_distance": decision.distance,
                    "cluster_reason": decision.reason,
                }
            )
            if not decision.allowed:
                continue
        notional = min(portfolio_value * position_fraction, available_cash)
        if notional <= 0:
            break
        intents.append(
            TradeIntent(
                pair=pair,
                side="BUY",
                notional_usd=notional,
                score=score,
                reason="ridge_entry",
                price=price,
                cluster_gate_enabled=cluster_gate is not None,
                cluster_gate_allowed=decision.allowed if decision is not None else None,
                cluster_id=decision.cluster_id if decision is not None else None,
                cluster_distance=decision.distance if decision is not None else None,
                cluster_reason=decision.reason if decision is not None else None,
            )
        )
        available_cash -= notional
    return intents, gate_decisions


def exit_intent(
    pair: str,
    quantity: float,
    entry_price: float | None,
    current_price: float,
    *,
    threshold: float | None = None,
    take_profit: float | None = None,
    stop_loss: float | None = None,
) -> TradeIntent | None:
    if quantity <= 0 or current_price <= 0 or not entry_price or entry_price <= 0:
        return None
    tp = threshold if take_profit is None else take_profit
    sl = threshold if stop_loss is None else stop_loss
    if tp is None or sl is None:
        raise ValueError("exit_intent requires threshold or both take_profit and stop_loss")
    if tp <= 0 or sl <= 0:
        raise ValueError("take_profit and stop_loss must be positive")
    trade_return = current_price / entry_price - 1.0
    tolerance = 1e-12
    if trade_return <= -sl + tolerance:
        return TradeIntent(
            pair=pair,
            side="SELL",
            quantity=quantity,
            reason="loss_threshold",
            price=current_price,
        )
    if trade_return >= tp - tolerance:
        return TradeIntent(
            pair=pair,
            side="SELL",
            quantity=quantity,
            reason="gain_threshold",
            price=current_price,
        )
    return None


def rank_decay_exit_intent(
    pair: str,
    quantity: float,
    current_price: float,
    rank: int | None,
    *,
    rank_exit_threshold: int,
) -> TradeIntent | None:
    if quantity <= 0 or current_price <= 0:
        return None
    if rank is None or rank > rank_exit_threshold:
        return TradeIntent(
            pair=pair,
            side="SELL",
            quantity=quantity,
            reason="rank_decay",
            price=current_price,
        )
    return None


def score_ranks(
    scores: pd.DataFrame,
    *,
    score_col: str = "ridge_score",
) -> dict[str, int]:
    ranked = scores.dropna(subset=[score_col]).sort_values(score_col, ascending=False)
    return {str(row.pair): idx for idx, row in enumerate(ranked.itertuples(index=False), start=1)}


def build_cycle_intents(
    scores: pd.DataFrame,
    positions: dict[str, Any],
    price_map: dict[str, float],
    *,
    portfolio_value: float,
    available_cash: float,
    position_fraction: float,
    max_positions: int,
    top_k: int | None = None,
    max_new_entries: int | None = None,
    regime_config: RegimeThrottleConfig | None = None,
    cluster_gate: ClusterRegimeGate | None = None,
    rank_exit_threshold: int | None = None,
    take_profit: float,
    stop_loss: float,
) -> CycleIntents:
    """Build live-style exit and entry intents without mutating portfolio state."""
    held_pairs = set(positions)
    exits = []
    ranks = score_ranks(scores) if rank_exit_threshold is not None else {}
    for pair, position in positions.items():
        current_price = price_map.get(pair)
        if current_price is None:
            continue
        if rank_exit_threshold is not None:
            intent = rank_decay_exit_intent(
                pair,
                float(getattr(position, "quantity")),
                current_price,
                ranks.get(pair),
                rank_exit_threshold=rank_exit_threshold,
            )
            if intent:
                exits.append(intent)
                continue
        intent = exit_intent(
            pair,
            float(getattr(position, "quantity")),
            getattr(position, "entry_price"),
            current_price,
            take_profit=take_profit,
            stop_loss=stop_loss,
        )
        if intent:
            exits.append(intent)

    regime = decide_regime_throttle(scores, regime_config) if regime_config is not None else None
    effective_max_new_entries = 0 if regime and regime.entries_blocked else max_new_entries

    entries, gate_decisions = entry_intents_with_gate(
        scores,
        held_pairs,
        portfolio_value,
        available_cash,
        position_fraction=position_fraction,
        max_positions=max_positions,
        top_k=top_k,
        max_new_entries=effective_max_new_entries,
        cluster_gate=cluster_gate,
    )
    entries = [intent for intent in entries if intent.pair in price_map]
    return CycleIntents(exits=exits, entries=entries, regime=regime, entry_gate_decisions=gate_decisions)

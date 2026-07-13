"""Train persisted live model artifacts from local historical features."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from bot.backtest.ridge_score_portfolio import _entry_regime_for_training, _simulate_scored_window
from bot.config import DEPLOYMENT_POLICY
from bot.model_store import save_cluster_gate, save_ridge_selection
from bot.strategy.regime import ClusterRegimeGateConfig, add_decision_time_regime_features, train_cluster_regime_gate
from bot.strategy.ridge import (
    DEFAULT_RIDGE_ALPHAS,
    RidgeSelection,
    labeled_training_slice,
    load_model_frame,
    score_frame,
    select_ridge_model,
)


def train_live_artifacts(
    *,
    feature_path: str | Path,
    pairs: tuple[str, ...],
    model_dir: str | Path,
    model: str,
    horizon: int,
    ridge_train_months: int,
    regime_train_months: int,
    ridge_alphas: tuple[float, ...] = DEFAULT_RIDGE_ALPHAS,
    cluster_config: ClusterRegimeGateConfig,
    trading_params: dict[str, Any],
    as_of: pd.Timestamp | None = None,
) -> dict[str, Path]:
    """Train and persist ridge and cluster-regime artifacts from local features."""
    frame = load_model_frame(feature_path, horizon=horizon)
    frame = frame[frame["pair"].isin(pairs)].copy()
    if frame.empty:
        raise ValueError("no feature rows for requested pairs")
    as_of_ts = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp(frame["timestamp"].max())
    if as_of_ts.tzinfo is None:
        as_of_ts = as_of_ts.tz_localize("UTC")
    else:
        as_of_ts = as_of_ts.tz_convert("UTC")

    ridge_start = as_of_ts - pd.DateOffset(months=ridge_train_months)
    regime_start = as_of_ts - pd.DateOffset(months=regime_train_months)
    _validate_pair_history(
        frame,
        pairs=pairs,
        required_start=regime_start,
        required_end=as_of_ts,
    )
    ridge_train = labeled_training_slice(
        frame,
        train_start=ridge_start,
        train_end=as_of_ts,
        horizon=horizon,
    )
    if ridge_train.empty:
        raise ValueError("ridge training window is empty")
    target_col = f"forward_return_{horizon}"
    selection = select_ridge_model(
        ridge_train,
        model,
        target_col=target_col,
        ridge_alphas=ridge_alphas,
    )

    regime_frame = frame[(frame["timestamp"] >= regime_start) & (frame["timestamp"] < as_of_ts)].copy()
    if regime_frame.empty:
        raise ValueError("regime training window is empty")
    regime_scored = add_decision_time_regime_features(
        score_frame(regime_frame, selection.terms, selection.beta, score_col="ridge_score")
    )
    equity, trades, _ = _simulate_scored_window(
        regime_scored,
        fold="live_regime_train",
        initial_cash=float(trading_params["initial_cash"]),
        position_fraction=float(trading_params["position_fraction"]),
        take_profit=float(trading_params["take_profit"]),
        stop_loss=float(trading_params["stop_loss"]),
        fee_bps=float(trading_params["fee_bps"]),
        slippage_bps=float(trading_params["slippage_bps"]),
        max_positions=int(trading_params["max_positions"]),
        top_k=trading_params["top_k"],
        max_new_entries=trading_params["max_new_entries"],
        regime_config=None,
        cluster_gate=None,
        rank_exit_threshold=None,
        exchange_info=None,
    )
    entries = _entry_regime_for_training(
        trades=pd.DataFrame(trades),
        equity=pd.DataFrame(equity),
        regime_features=regime_scored,
    )
    gate = train_cluster_regime_gate(entries, cluster_config)

    common_metadata = {
        "feature_path": str(feature_path),
        "deployment_policy": DEPLOYMENT_POLICY,
        "pairs": list(pairs),
        "as_of": as_of_ts.isoformat(),
        "horizon": horizon,
        "model": model,
        **trading_params,
    }
    ridge_metadata = {
        **common_metadata,
        "train_start": ridge_start.isoformat(),
        "train_end": as_of_ts.isoformat(),
        "train_months": ridge_train_months,
        "ridge_alphas": ridge_alphas,
        "train_rows": int(len(ridge_train)),
    }
    regime_metadata = {
        **common_metadata,
        "train_start": regime_start.isoformat(),
        "train_end": as_of_ts.isoformat(),
        "train_months": regime_train_months,
        "train_rows": int(len(regime_frame)),
        "simulated_entries": int(len(entries)),
    }
    return {
        "ridge": save_ridge_selection(selection, ridge_metadata, model_dir),
        "cluster_gate": save_cluster_gate(gate, regime_metadata, model_dir),
    }


def selection_metadata_summary(selection: RidgeSelection, metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": selection.model,
        "alpha": selection.alpha,
        "terms": ",".join(selection.terms),
        "train_start": metadata.get("train_start"),
        "train_end": metadata.get("train_end"),
        "train_months": metadata.get("train_months"),
    }


def _validate_pair_history(
    frame: pd.DataFrame,
    *,
    pairs: tuple[str, ...],
    required_start: pd.Timestamp,
    required_end: pd.Timestamp,
) -> None:
    coverage = frame.groupby("pair", observed=True)["timestamp"].agg(["min", "max"])
    missing = sorted(set(pairs) - set(coverage.index))
    too_short = [
        pair
        for pair, row in coverage.loc[[pair for pair in pairs if pair in coverage.index]].iterrows()
        if pd.Timestamp(row["min"]) > required_start or pd.Timestamp(row["max"]) < required_end
    ]
    problems = []
    if missing:
        problems.append("missing pairs: " + ", ".join(missing))
    if too_short:
        problems.append(
            f"pairs without full history from {required_start} through {required_end}: "
            + ", ".join(sorted(too_short))
        )
    if problems:
        raise ValueError("; ".join(problems))

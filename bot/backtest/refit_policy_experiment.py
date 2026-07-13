"""Compare fixed and scheduled OOS refit policies for ridge + cluster gates."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from bot.backtest.portfolio import Portfolio, compute_equity_metrics
from bot.backtest.ridge_score_portfolio import (
    DEFAULT_INITIAL_CASH,
    DEFAULT_POSITION_FRACTION,
    _attach_return_over_drawdown,
    _entry_regime_for_training,
    _fold_trade_summary,
    _os_scored_with_regime_features,
    _simulate_scored_window,
    filter_pairs,
    parse_alphas,
    parse_pairs,
    write_outputs,
)
from bot.backtest.simulated_executor import SimulatedExecutor
from bot.strategy.regime import (
    ClusterRegimeGate,
    ClusterRegimeGateConfig,
    add_decision_time_regime_features,
    train_cluster_regime_gate,
)
from bot.strategy.ridge import (
    DEFAULT_FEATURE_PATH,
    DEFAULT_HORIZON,
    DEFAULT_RIDGE_ALPHAS,
    RidgeSelection,
    bar_delta,
    labeled_training_slice,
    load_model_frame,
    score_frame,
    select_ridge_model,
)


DEFAULT_OUTPUT_DIR = Path("reports/backtests/refit_policy")
DEFAULT_MODEL = "momentum_only"
_WORKER_FRAME: pd.DataFrame | None = None
_WORKER_SETTINGS: dict | None = None


@dataclass(frozen=True)
class RefitSegment:
    segment_id: int
    start: pd.Timestamp
    end: pd.Timestamp
    ridge_refit: bool
    regime_refit: bool


@dataclass(frozen=True)
class ActiveArtifacts:
    ridge_selection: RidgeSelection
    ridge_train_start: pd.Timestamp
    ridge_train_end: pd.Timestamp
    ridge_effective_end: pd.Timestamp
    regime_train_start: pd.Timestamp | None
    regime_train_end: pd.Timestamp | None
    cluster_gate: ClusterRegimeGate | None
    regime_scored: pd.DataFrame


Policy = Literal["baseline", "fixed", "scheduled"]


@dataclass(frozen=True)
class OosFold:
    fold_id: int
    os_start: pd.Timestamp
    os_end: pd.Timestamp


def build_refit_segments(
    os_start: pd.Timestamp,
    os_end: pd.Timestamp,
    *,
    ridge_refit_days: int,
    regime_refit_months: int,
    scheduled: bool,
) -> list[RefitSegment]:
    """Return contiguous OOS trading segments split at scheduled refit times."""
    os_start = pd.Timestamp(os_start)
    os_end = pd.Timestamp(os_end)
    if os_end <= os_start:
        raise ValueError("os_end must be after os_start")
    if ridge_refit_days <= 0:
        raise ValueError("ridge_refit_days must be positive")
    if regime_refit_months <= 0:
        raise ValueError("regime_refit_months must be positive")

    ridge_times = {os_start}
    regime_times = {os_start}
    if scheduled:
        cursor = os_start + pd.Timedelta(days=ridge_refit_days)
        while cursor < os_end:
            ridge_times.add(cursor)
            cursor += pd.Timedelta(days=ridge_refit_days)

        cursor = os_start + pd.DateOffset(months=regime_refit_months)
        while cursor < os_end:
            regime_times.add(cursor)
            cursor += pd.DateOffset(months=regime_refit_months)

    boundaries = sorted(ridge_times | regime_times | {os_end})
    segments = []
    for idx, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
        segments.append(
            RefitSegment(
                segment_id=idx,
                start=start,
                end=end,
                ridge_refit=start in ridge_times,
                regime_refit=start in regime_times,
            )
        )
    return segments


def latest_oos_window(
    frame: pd.DataFrame,
    *,
    os_months: int,
    required_train_months: int,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Pick the latest complete OOS window and validate required lookback history."""
    if os_months <= 0:
        raise ValueError("os_months must be positive")
    if required_train_months <= 0:
        raise ValueError("required_train_months must be positive")
    if frame.empty:
        raise ValueError("feature frame is empty")
    start = pd.Timestamp(frame["timestamp"].min())
    data_end = _data_end_exclusive(frame)
    os_start = data_end - pd.DateOffset(months=os_months)
    required_start = os_start - pd.DateOffset(months=required_train_months)
    if start > required_start:
        raise ValueError(
            f"not enough history for a {required_train_months}-month regime window before OOS. "
            f"Need data from <= {required_start}; available starts at {start}."
        )
    return os_start, data_end


def build_oos_folds(
    frame: pd.DataFrame,
    *,
    os_months: int,
    required_train_months: int,
    folds: int,
    fold_step_days: int,
) -> list[OosFold]:
    """Build latest-first walk-forward OOS folds stepped backward by whole days."""
    if folds <= 0:
        raise ValueError("folds must be positive")
    if fold_step_days <= 0:
        raise ValueError("fold_step_days must be positive")
    latest_start, latest_end = latest_oos_window(
        frame,
        os_months=os_months,
        required_train_months=required_train_months,
    )
    data_start = pd.Timestamp(frame["timestamp"].min())
    built = []
    for offset in range(folds):
        os_start = latest_start - pd.Timedelta(days=fold_step_days * offset)
        os_end = os_start + pd.DateOffset(months=os_months)
        required_start = os_start - pd.DateOffset(months=required_train_months)
        if required_start < data_start:
            break
        if os_end > latest_end:
            continue
        built.append(OosFold(fold_id=offset, os_start=os_start, os_end=os_end))
    if len(built) < folds:
        raise ValueError(
            f"only {len(built)} folds are available with {required_train_months} months of lookback, "
            f"{os_months} OOS months, and {fold_step_days}-day fold steps. "
            f"Requested {folds}; available data starts at {data_start}."
        )
    ordered = list(reversed(built))
    return [
        OosFold(fold_id=fold_id, os_start=fold.os_start, os_end=fold.os_end)
        for fold_id, fold in enumerate(ordered)
    ]


def filter_common_history_pairs(
    frame: pd.DataFrame,
    *,
    os_folds: list[OosFold],
    required_train_months: int,
) -> tuple[pd.DataFrame, list[str]]:
    """Keep pairs that span the full train+OOS date range used by every fold."""
    if not os_folds:
        raise ValueError("os_folds must not be empty")
    required_start = min(fold.os_start - pd.DateOffset(months=required_train_months) for fold in os_folds)
    required_end = max(fold.os_end for fold in os_folds)
    bar_delta = _bar_delta(frame)
    required_last_bar = required_end - bar_delta
    coverage = frame.groupby("pair")["timestamp"].agg(["min", "max", "count"])
    eligible = coverage[
        (coverage["min"] <= required_start)
        & (coverage["max"] >= required_last_bar)
        & (coverage["count"] == coverage["count"].max())
    ].index.tolist()
    if not eligible:
        raise ValueError(
            f"no pairs have full common history from {required_start} through {required_last_bar}"
        )
    return filter_pairs(frame, eligible), sorted(eligible)


def ridge_training_slice(
    frame: pd.DataFrame,
    *,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    horizon: int,
) -> pd.DataFrame:
    """Select ridge training rows whose labels are known before train_end."""
    return labeled_training_slice(
        frame,
        train_start=train_start,
        train_end=train_end,
        horizon=horizon,
    )


def run_refit_policy_backtest(
    frame: pd.DataFrame,
    *,
    policy: Policy,
    model: str = DEFAULT_MODEL,
    horizon: int = DEFAULT_HORIZON,
    os_months: int = 2,
    ridge_train_months: int = 3,
    regime_train_months: int = 24,
    ridge_refit_days: int = 7,
    regime_refit_months: int = 1,
    ridge_alphas: tuple[float, ...] = DEFAULT_RIDGE_ALPHAS,
    initial_cash: float = DEFAULT_INITIAL_CASH,
    position_fraction: float = DEFAULT_POSITION_FRACTION,
    take_profit: float = 0.03,
    stop_loss: float = 0.015,
    fee_bps: float = 10.0,
    slippage_bps: float = 5.0,
    max_positions: int | None = 3,
    top_k: int | None = 1,
    max_new_entries: int | None = 1,
    cluster_gate_config: ClusterRegimeGateConfig | None = None,
    exchange_info: dict | None = None,
    os_start: pd.Timestamp | None = None,
    os_end: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run one latest-window OOS policy and return summary, equity, trades, segments."""
    if policy not in {"baseline", "fixed", "scheduled"}:
        raise ValueError("policy must be one of: baseline, fixed, scheduled")
    if not 0 < position_fraction <= 1:
        raise ValueError("position_fraction must be in (0, 1]")
    if take_profit <= 0 or stop_loss <= 0:
        raise ValueError("take_profit and stop_loss must be positive")
    max_positions = max_positions or max(1, int(np.floor(1 / position_fraction)))
    if os_start is None or os_end is None:
        os_start, os_end = latest_oos_window(
            frame,
            os_months=os_months,
            required_train_months=max(ridge_train_months, regime_train_months),
        )
    else:
        os_start = pd.Timestamp(os_start)
        os_end = pd.Timestamp(os_end)
        required_start = os_start - pd.DateOffset(months=max(ridge_train_months, regime_train_months))
        if pd.Timestamp(frame["timestamp"].min()) > required_start:
            raise ValueError(
                f"not enough history for the requested OOS fold. Need data from <= {required_start}; "
                f"available starts at {pd.Timestamp(frame['timestamp'].min())}."
            )
        if _data_end_exclusive(frame) < os_end:
            raise ValueError(
                f"not enough OOS data for fold ending {os_end}; available ends at {_data_end_exclusive(frame)}."
            )
    segments = build_refit_segments(
        os_start,
        os_end,
        ridge_refit_days=ridge_refit_days,
        regime_refit_months=regime_refit_months,
        scheduled=policy == "scheduled",
    )
    if policy == "baseline":
        segments = [
            RefitSegment(
                segment_id=segment.segment_id,
                start=segment.start,
                end=segment.end,
                ridge_refit=segment.ridge_refit,
                regime_refit=False,
            )
            for segment in segments
        ]

    portfolio = Portfolio(initial_cash=initial_cash, fee_bps=fee_bps)
    executor = SimulatedExecutor(portfolio, exchange_info=exchange_info, slippage_bps=slippage_bps)
    active: ActiveArtifacts | None = None
    equity_rows: list[dict] = []
    trade_rows: list[dict] = []
    gate_rows: list[dict] = []
    segment_rows: list[dict] = []
    cluster_summary_rows: list[pd.DataFrame] = []
    cluster_profile_rows: list[pd.DataFrame] = []

    for segment in segments:
        if active is None or segment.ridge_refit:
            active = _fit_active_ridge(
                frame,
                refit_time=segment.start,
                ridge_train_months=ridge_train_months,
                horizon=horizon,
                model=model,
                ridge_alphas=ridge_alphas,
                previous=active,
            )
        if segment.regime_refit:
            active = _fit_active_regime(
                frame,
                refit_time=segment.start,
                regime_train_months=regime_train_months,
                active=active,
                cluster_gate_config=None if policy == "baseline" else cluster_gate_config,
                initial_cash=initial_cash,
                position_fraction=position_fraction,
                take_profit=take_profit,
                stop_loss=stop_loss,
                fee_bps=fee_bps,
                slippage_bps=slippage_bps,
                max_positions=max_positions,
                top_k=top_k,
                max_new_entries=max_new_entries,
                exchange_info=exchange_info,
            )
            if active.cluster_gate is not None:
                cluster_summary_rows.append(_tag_cluster_frame(active.cluster_gate.cluster_summary, segment))
                cluster_profile_rows.append(_tag_cluster_frame(active.cluster_gate.cluster_profiles, segment))

        segment_frame = _slice(frame, segment.start, segment.end)
        scored = score_frame(
            segment_frame,
            active.ridge_selection.terms,
            active.ridge_selection.beta,
            score_col="ridge_score",
        )
        if active.cluster_gate is not None:
            scored = _os_scored_with_regime_features(active.regime_scored, scored)
        is_final_segment = segment.segment_id == len(segments) - 1
        seg_equity, seg_trades, seg_gate = _simulate_scored_window(
            scored,
            fold=policy,
            initial_cash=initial_cash,
            position_fraction=position_fraction,
            take_profit=take_profit,
            stop_loss=stop_loss,
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
            max_positions=max_positions,
            top_k=top_k,
            max_new_entries=max_new_entries,
            regime_config=None,
            cluster_gate=active.cluster_gate,
            rank_exit_threshold=None,
            exchange_info=exchange_info,
            portfolio=portfolio,
            executor=executor,
            liquidate_at_end=is_final_segment,
            final_liquidation_timestamp="oos_end",
        )
        _annotate_rows(seg_equity, active, segment)
        _annotate_rows(seg_trades, active, segment)
        _annotate_rows(seg_gate, active, segment)
        equity_rows.extend(seg_equity)
        trade_rows.extend(seg_trades)
        gate_rows.extend(seg_gate)
        segment_rows.append(_segment_row(segment, active, seg_gate))

    if equity_rows:
        equity_rows.append(
            {
                **equity_rows[-1],
                "open_time": "oos_end",
                "timestamp": os_end,
                "equity": portfolio.value({}),
                "cash": portfolio.cash,
                "positions": len(portfolio.positions),
                "cluster_gate_checked": 0,
                "cluster_gate_blocked": 0,
            }
        )

    summary = _summary_frame(
        policy=policy,
        model=model,
        os_start=os_start,
        os_end=os_end,
        initial_cash=initial_cash,
        equity_rows=equity_rows,
        trade_rows=trade_rows,
        gate_rows=gate_rows,
        segments=segments,
    )
    summary.attrs["cluster_gate_decisions"] = pd.DataFrame(gate_rows)
    summary.attrs["cluster_gate_summary"] = (
        pd.concat(cluster_summary_rows, ignore_index=True) if cluster_summary_rows else pd.DataFrame()
    )
    summary.attrs["cluster_gate_profiles"] = (
        pd.concat(cluster_profile_rows, ignore_index=True) if cluster_profile_rows else pd.DataFrame()
    )
    return summary, pd.DataFrame(equity_rows), pd.DataFrame(trade_rows), pd.DataFrame(segment_rows)


def write_refit_outputs(
    summary: pd.DataFrame,
    equity: pd.DataFrame,
    trades: pd.DataFrame,
    segments: pd.DataFrame,
    output_dir: Path,
    prefix: str,
    metadata: dict,
) -> None:
    write_outputs(summary, equity, trades, output_dir, prefix, metadata)
    if not segments.empty:
        segments.to_csv(output_dir / f"{prefix}_segments.csv", index=False)


def _fit_active_ridge(
    frame: pd.DataFrame,
    *,
    refit_time: pd.Timestamp,
    ridge_train_months: int,
    horizon: int,
    model: str,
    ridge_alphas: tuple[float, ...],
    previous: ActiveArtifacts | None,
) -> ActiveArtifacts:
    train_start = pd.Timestamp(refit_time) - pd.DateOffset(months=ridge_train_months)
    train = ridge_training_slice(frame, train_start=train_start, train_end=refit_time, horizon=horizon)
    selection = select_ridge_model(
        train,
        model,
        target_col=f"forward_return_{horizon}",
        ridge_alphas=ridge_alphas,
    )
    return ActiveArtifacts(
        ridge_selection=selection,
        ridge_train_start=train_start,
        ridge_train_end=pd.Timestamp(refit_time),
        ridge_effective_end=pd.Timestamp(train["timestamp"].max()) if not train.empty else train_start,
        regime_train_start=previous.regime_train_start if previous else None,
        regime_train_end=previous.regime_train_end if previous else None,
        cluster_gate=previous.cluster_gate if previous else None,
        regime_scored=previous.regime_scored if previous else pd.DataFrame(),
    )


def _fit_active_regime(
    frame: pd.DataFrame,
    *,
    refit_time: pd.Timestamp,
    regime_train_months: int,
    active: ActiveArtifacts,
    cluster_gate_config: ClusterRegimeGateConfig | None,
    initial_cash: float,
    position_fraction: float,
    take_profit: float,
    stop_loss: float,
    fee_bps: float,
    slippage_bps: float,
    max_positions: int,
    top_k: int | None,
    max_new_entries: int | None,
    exchange_info: dict | None,
) -> ActiveArtifacts:
    train_start = pd.Timestamp(refit_time) - pd.DateOffset(months=regime_train_months)
    history = _slice(frame, train_start, refit_time)
    scored = score_frame(
        history,
        active.ridge_selection.terms,
        active.ridge_selection.beta,
        score_col="ridge_score",
    )
    regime_scored = add_decision_time_regime_features(scored) if not scored.empty else scored
    cluster_gate = None
    if cluster_gate_config is not None:
        equity, trades, _ = _simulate_scored_window(
            regime_scored,
            fold="cluster_train",
            initial_cash=initial_cash,
            position_fraction=position_fraction,
            take_profit=take_profit,
            stop_loss=stop_loss,
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
            max_positions=max_positions,
            top_k=top_k,
            max_new_entries=max_new_entries,
            regime_config=None,
            cluster_gate=None,
            rank_exit_threshold=None,
            exchange_info=exchange_info,
            liquidate_at_end=True,
            final_liquidation_timestamp="regime_train_end",
        )
        entries = _entry_regime_for_training(
            trades=pd.DataFrame(trades),
            equity=pd.DataFrame(equity),
            regime_features=regime_scored,
        )
        cluster_gate = train_cluster_regime_gate(entries, cluster_gate_config)
    return ActiveArtifacts(
        ridge_selection=active.ridge_selection,
        ridge_train_start=active.ridge_train_start,
        ridge_train_end=active.ridge_train_end,
        ridge_effective_end=active.ridge_effective_end,
        regime_train_start=train_start,
        regime_train_end=pd.Timestamp(refit_time),
        cluster_gate=cluster_gate,
        regime_scored=regime_scored,
    )


def _slice(frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    return frame[(frame["timestamp"] >= pd.Timestamp(start)) & (frame["timestamp"] < pd.Timestamp(end))].copy()


def _bar_delta(frame: pd.DataFrame) -> pd.Timedelta:
    return bar_delta(frame)


def _data_end_exclusive(frame: pd.DataFrame) -> pd.Timestamp:
    return pd.Timestamp(frame["timestamp"].max()) + _bar_delta(frame)


def _annotate_rows(rows: list[dict], active: ActiveArtifacts, segment: RefitSegment) -> None:
    for row in rows:
        row.update(
            {
                "refit_segment_id": segment.segment_id,
                "segment_start": segment.start,
                "segment_end": segment.end,
                "ridge_refit": segment.ridge_refit,
                "regime_refit": segment.regime_refit,
                "ridge_train_start": active.ridge_train_start,
                "ridge_train_end": active.ridge_train_end,
                "ridge_effective_label_end": active.ridge_effective_end,
                "ridge_alpha": active.ridge_selection.alpha,
                "ridge_beta": ",".join(f"{value:.12g}" for value in active.ridge_selection.beta),
                "regime_train_start": active.regime_train_start,
                "regime_train_end": active.regime_train_end,
                "cluster_allowed_clusters": _allowed_clusters(active.cluster_gate),
                "cluster_training_average_return": (
                    float(active.cluster_gate.training_average_return)
                    if active.cluster_gate is not None
                    else np.nan
                ),
            }
        )


def _segment_row(segment: RefitSegment, active: ActiveArtifacts, gate_rows: list[dict]) -> dict:
    allowed = [bool(row.get("cluster_gate_allowed")) for row in gate_rows]
    return {
        "refit_segment_id": segment.segment_id,
        "segment_start": segment.start,
        "segment_end": segment.end,
        "ridge_refit": segment.ridge_refit,
        "regime_refit": segment.regime_refit,
        "ridge_train_start": active.ridge_train_start,
        "ridge_train_end": active.ridge_train_end,
        "ridge_effective_label_end": active.ridge_effective_end,
        "ridge_alpha": active.ridge_selection.alpha,
        "ridge_beta": ",".join(f"{value:.12g}" for value in active.ridge_selection.beta),
        "regime_train_start": active.regime_train_start,
        "regime_train_end": active.regime_train_end,
        "allowed_clusters": _allowed_clusters(active.cluster_gate),
        "cluster_training_average_return": (
            float(active.cluster_gate.training_average_return) if active.cluster_gate is not None else np.nan
        ),
        "candidate_checks": len(gate_rows),
        "allowed_checks": int(sum(allowed)),
        "blocked_checks": int(len(gate_rows) - sum(allowed)),
    }


def _summary_frame(
    *,
    policy: Policy,
    model: str,
    os_start: pd.Timestamp,
    os_end: pd.Timestamp,
    initial_cash: float,
    equity_rows: list[dict],
    trade_rows: list[dict],
    gate_rows: list[dict],
    segments: list[RefitSegment],
) -> pd.DataFrame:
    equity_values = [row["equity"] for row in equity_rows]
    metrics = compute_equity_metrics(equity_values, initial_cash) if equity_values else {}
    exits = [row for row in trade_rows if row.get("side") == "SELL"]
    trade_summary = _fold_trade_summary(exits)
    returns = [float(row["return_pct"]) for row in exits if pd.notna(row.get("return_pct"))]
    pnls = [float(row["pnl"]) for row in exits if pd.notna(row.get("pnl"))]
    allowed = [bool(row.get("cluster_gate_allowed")) for row in gate_rows]
    row = {
        "policy": policy,
        "model": model,
        "os_start": os_start,
        "os_end": os_end,
        "segments": len(segments),
        "ridge_refits": int(sum(segment.ridge_refit for segment in segments)),
        "regime_refits": int(sum(segment.regime_refit for segment in segments)),
        "trades": len(trade_rows),
        "closed_trades": len(exits),
        "win_rate": float(np.mean([row.get("pnl", 0) > 0 for row in exits])) if exits else np.nan,
        "mean_return_pct": float(np.mean(returns) * 100) if returns else np.nan,
        "gross_profit": float(sum(value for value in pnls if value > 0)),
        "gross_loss": float(sum(value for value in pnls if value < 0)),
        "cluster_gate_checks": len(gate_rows),
        "cluster_gate_allowed_checks": int(sum(allowed)),
        "cluster_gate_blocked_checks": int(len(gate_rows) - sum(allowed)),
        **metrics,
        **trade_summary,
    }
    return _attach_return_over_drawdown(pd.DataFrame([row]))


def _comparison_row(label: str, summary: pd.DataFrame) -> dict[str, object]:
    if summary.empty:
        return {"label": label}
    row = summary.iloc[0]
    columns = [
        "total_return_pct",
        "max_drawdown_pct",
        "sharpe",
        "sortino",
        "closed_trades",
        "tp_exits",
        "stop_exits",
        "win_rate",
        "mean_return_pct",
        "gross_profit",
        "gross_loss",
        "cluster_gate_checks",
        "cluster_gate_blocked_checks",
    ]
    return {"label": label, **{column: row.get(column) for column in columns}}


def _aggregate_walk_forward(comparison: pd.DataFrame) -> pd.DataFrame:
    if comparison.empty:
        return pd.DataFrame()
    rows = []
    for label, group in comparison.groupby("label", sort=False):
        gross_profit = pd.to_numeric(group["gross_profit"], errors="coerce")
        gross_loss = pd.to_numeric(group["gross_loss"], errors="coerce").abs()
        returns = pd.to_numeric(group["total_return_pct"], errors="coerce")
        drawdowns = pd.to_numeric(group["max_drawdown_pct"], errors="coerce")
        sharpes = pd.to_numeric(group["sharpe"], errors="coerce")
        sortinos = pd.to_numeric(group["sortino"], errors="coerce")
        win_rates = pd.to_numeric(group["win_rate"], errors="coerce")
        mean_trade_returns = pd.to_numeric(group["mean_return_pct"], errors="coerce")
        trades = pd.to_numeric(group["closed_trades"], errors="coerce")
        blocked = pd.to_numeric(group["cluster_gate_blocked_checks"], errors="coerce")
        checks = pd.to_numeric(group["cluster_gate_checks"], errors="coerce")
        rows.append(
            {
                "label": label,
                "folds": int(len(group)),
                "positive_return_folds": int((returns > 0).sum()),
                "traded_folds": int((trades > 0).sum()),
                "flat_folds": int((trades == 0).sum()),
                "mean_return_pct": float(returns.mean()),
                "mean_return_when_trading_pct": float(returns[trades > 0].mean()) if (trades > 0).any() else np.nan,
                "median_return_pct": float(returns.median()),
                "std_return_pct": float(returns.std(ddof=0)),
                "worst_return_pct": float(returns.min()),
                "best_return_pct": float(returns.max()),
                "mean_max_drawdown_pct": float(drawdowns.mean()),
                "worst_max_drawdown_pct": float(drawdowns.max()),
                "mean_sharpe": float(sharpes.mean()),
                "mean_sortino": float(sortinos.mean()),
                "mean_return_to_drawdown": float((returns / drawdowns.replace(0, np.nan)).mean()),
                "mean_closed_trades": float(trades.mean()),
                "mean_win_rate": float(win_rates.mean()),
                "mean_trade_return_pct": float(mean_trade_returns.mean()),
                "profit_factor": float(gross_profit.sum() / gross_loss.sum()) if gross_loss.sum() else np.nan,
                "total_gross_profit": float(gross_profit.sum()),
                "total_gross_loss": float(-gross_loss.sum()),
                "gate_block_rate": float(blocked.sum() / checks.sum()) if checks.sum() else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["mean_return_pct", "mean_sharpe", "mean_return_to_drawdown"],
        ascending=[False, False, False],
    )


def _paired_policy_deltas(comparison: pd.DataFrame) -> pd.DataFrame:
    if comparison.empty:
        return pd.DataFrame()
    metrics = [
        "total_return_pct",
        "max_drawdown_pct",
        "sharpe",
        "sortino",
        "win_rate",
        "mean_return_pct",
        "closed_trades",
    ]
    rows = []
    pairs = [
        ("v1_fixed_regime", "v0_no_regime"),
        ("v2_scheduled_refit", "v1_fixed_regime"),
        ("v2_scheduled_refit", "v0_no_regime"),
    ]
    for left, right in pairs:
        left_rows = comparison[comparison["label"].eq(left)].set_index("fold_id")
        right_rows = comparison[comparison["label"].eq(right)].set_index("fold_id")
        common = sorted(set(left_rows.index) & set(right_rows.index))
        if not common:
            continue
        row: dict[str, object] = {"comparison": f"{left}_minus_{right}", "folds": len(common)}
        for metric in metrics:
            diff = pd.to_numeric(left_rows.loc[common, metric], errors="coerce") - pd.to_numeric(
                right_rows.loc[common, metric], errors="coerce"
            )
            row[f"mean_delta_{metric}"] = float(diff.mean())
            row[f"median_delta_{metric}"] = float(diff.median())
            if metric == "total_return_pct":
                row["better_return_folds"] = int((diff > 0).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def _run_fold_policy(
    frame: pd.DataFrame,
    *,
    fold: OosFold,
    label: str,
    policy: Policy,
    gate_config: ClusterRegimeGateConfig | None,
    settings: dict,
) -> dict[str, object]:
    summary, equity, trades, segments = run_refit_policy_backtest(
        frame,
        policy=policy,
        model=settings["model"],
        horizon=settings["horizon"],
        os_months=settings["os_months"],
        ridge_train_months=settings["ridge_train_months"],
        regime_train_months=settings["regime_train_months"],
        ridge_refit_days=settings["ridge_refit_days"],
        regime_refit_months=settings["regime_refit_months"],
        ridge_alphas=settings["ridge_alphas"],
        initial_cash=settings["initial_cash"],
        position_fraction=settings["position_fraction"],
        take_profit=settings["take_profit"],
        stop_loss=settings["stop_loss"],
        fee_bps=settings["fee_bps"],
        slippage_bps=settings["slippage_bps"],
        max_positions=settings["max_positions"],
        top_k=settings["top_k"],
        max_new_entries=settings["max_new_entries"],
        cluster_gate_config=gate_config,
        os_start=fold.os_start,
        os_end=fold.os_end,
    )
    for frame_out in (summary, equity, trades, segments):
        if not frame_out.empty:
            frame_out.insert(0, "fold_id", fold.fold_id)
            frame_out.insert(1, "fold_os_start", fold.os_start)
            frame_out.insert(2, "fold_os_end", fold.os_end)
    fold_tag = f"fold{fold.fold_id:02d}_" if settings["folds"] > 1 else ""
    run_prefix = f"{settings['prefix']}_{fold_tag}{label}_slip{_tag_number(settings['slippage_bps'])}"
    metadata = {
        "label": label,
        "policy": policy,
        "fold_id": fold.fold_id,
        "fold_os_start": str(fold.os_start),
        "fold_os_end": str(fold.os_end),
        "feature_path": settings["feature_path"],
        "model": settings["model"],
        "horizon": settings["horizon"],
        "os_months": settings["os_months"],
        "folds": settings["folds"],
        "fold_step_days": settings["fold_step_days"],
        "common_history_universe": settings["common_history_universe"],
        "ridge_train_months": settings["ridge_train_months"],
        "regime_train_months": settings["regime_train_months"],
        "ridge_refit_days": settings["ridge_refit_days"],
        "regime_refit_months": settings["regime_refit_months"],
        "ridge_alphas": settings["ridge_alphas"],
        "initial_cash": settings["initial_cash"],
        "position_fraction": settings["position_fraction"],
        "take_profit": settings["take_profit"],
        "stop_loss": settings["stop_loss"],
        "fee_bps": settings["fee_bps"],
        "slippage_bps": settings["slippage_bps"],
        "max_positions": settings["max_positions"],
        "top_k": settings["top_k"],
        "max_new_entries": settings["max_new_entries"],
        "cluster_regime_gate": gate_config is not None,
        "cluster_n_clusters": settings["cluster_n_clusters"],
        "cluster_min_trades": settings["cluster_min_trades"],
        "cluster_random_seed": settings["cluster_random_seed"],
        "pair_universe": settings["eligible_pairs"],
        "rows": settings["rows"],
        "start": settings["start"],
        "end": settings["end"],
    }
    write_refit_outputs(summary, equity, trades, segments, settings["output_dir"], run_prefix, metadata)
    row = _comparison_row(label, summary)
    row.update({"fold_id": fold.fold_id, "os_start": fold.os_start, "os_end": fold.os_end})
    return row


def _init_worker(frame: pd.DataFrame, settings: dict) -> None:
    global _WORKER_FRAME, _WORKER_SETTINGS
    _WORKER_FRAME = frame
    _WORKER_SETTINGS = settings


def _run_fold_policy_worker(job: tuple[OosFold, str, Policy, ClusterRegimeGateConfig | None]) -> dict[str, object]:
    if _WORKER_FRAME is None or _WORKER_SETTINGS is None:
        raise RuntimeError("worker was not initialized")
    fold, label, policy, gate_config = job
    return _run_fold_policy(
        _WORKER_FRAME,
        fold=fold,
        label=label,
        policy=policy,
        gate_config=gate_config,
        settings=_WORKER_SETTINGS,
    )


def _tag_cluster_frame(frame: pd.DataFrame, segment: RefitSegment) -> pd.DataFrame:
    out = frame.copy()
    out.insert(0, "refit_segment_id", segment.segment_id)
    out.insert(1, "segment_start", segment.start)
    out.insert(2, "segment_end", segment.end)
    return out


def _allowed_clusters(cluster_gate: ClusterRegimeGate | None) -> str:
    if cluster_gate is None:
        return ""
    return ",".join(str(cluster_id) for cluster_id in sorted(cluster_gate.allowed_clusters))


def _tag_number(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else str(value).replace(".", "p")


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare fixed vs scheduled OOS refit policies")
    parser.add_argument("--features", default=str(DEFAULT_FEATURE_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--prefix", default="refit_policy")
    parser.add_argument("--pairs", default="all")
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--os-months", type=int, default=2)
    parser.add_argument("--folds", type=int, default=1)
    parser.add_argument("--fold-step-days", type=int, default=7)
    parser.add_argument("--common-history-universe", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--ridge-train-months", type=int, default=3)
    parser.add_argument("--regime-train-months", type=int, default=24)
    parser.add_argument("--ridge-refit-days", type=int, default=7)
    parser.add_argument("--regime-refit-months", type=int, default=1)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--ridge-alphas", default=",".join(str(alpha) for alpha in DEFAULT_RIDGE_ALPHAS))
    parser.add_argument("--initial-cash", type=float, default=DEFAULT_INITIAL_CASH)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--max-new-entries", type=int, default=1)
    parser.add_argument("--max-positions", type=int, default=3)
    parser.add_argument("--position-fraction", type=float, default=0.25)
    parser.add_argument("--tp", type=float, default=0.03)
    parser.add_argument("--sl", type=float, default=0.015)
    parser.add_argument("--fee-bps", type=float, default=10.0)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--cluster-n-clusters", type=int, default=4)
    parser.add_argument("--cluster-min-trades", type=int, default=50)
    parser.add_argument("--cluster-random-seed", type=int, default=42)
    args = parser.parse_args()

    feature_path = Path(args.features)
    pairs = parse_pairs(args.pairs)
    frame = filter_pairs(load_model_frame(feature_path, horizon=args.horizon), pairs)
    ridge_alphas = parse_alphas(args.ridge_alphas)
    output_dir = Path(args.output_dir)
    required_train_months = max(args.ridge_train_months, args.regime_train_months)
    os_folds = build_oos_folds(
        frame,
        os_months=args.os_months,
        required_train_months=required_train_months,
        folds=args.folds,
        fold_step_days=args.fold_step_days,
    )
    if args.common_history_universe or args.folds > 1:
        frame, eligible_pairs = filter_common_history_pairs(
            frame,
            os_folds=os_folds,
            required_train_months=required_train_months,
        )
    else:
        eligible_pairs = sorted(frame["pair"].unique())
    cluster_config = ClusterRegimeGateConfig(
        n_clusters=args.cluster_n_clusters,
        min_cluster_trades=args.cluster_min_trades,
        lookback_months=args.regime_train_months,
        random_seed=args.cluster_random_seed,
    )
    if args.workers <= 0:
        raise SystemExit("error: --workers must be positive")
    settings = {
        "feature_path": str(feature_path),
        "output_dir": output_dir,
        "prefix": args.prefix,
        "model": args.model,
        "horizon": args.horizon,
        "os_months": args.os_months,
        "folds": args.folds,
        "fold_step_days": args.fold_step_days,
        "common_history_universe": bool(args.common_history_universe or args.folds > 1),
        "ridge_train_months": args.ridge_train_months,
        "regime_train_months": args.regime_train_months,
        "ridge_refit_days": args.ridge_refit_days,
        "regime_refit_months": args.regime_refit_months,
        "ridge_alphas": ridge_alphas,
        "initial_cash": args.initial_cash,
        "position_fraction": args.position_fraction,
        "take_profit": args.tp,
        "stop_loss": args.sl,
        "fee_bps": args.fee_bps,
        "slippage_bps": args.slippage_bps,
        "max_positions": args.max_positions,
        "top_k": args.top_k,
        "max_new_entries": args.max_new_entries,
        "cluster_n_clusters": args.cluster_n_clusters,
        "cluster_min_trades": args.cluster_min_trades,
        "cluster_random_seed": args.cluster_random_seed,
        "eligible_pairs": eligible_pairs,
        "rows": int(len(frame)),
        "start": str(frame["timestamp"].min()),
        "end": str(frame["timestamp"].max()),
    }
    jobs: list[tuple[OosFold, str, Policy, ClusterRegimeGateConfig | None]] = [
        (fold, label, policy, gate_config)
        for fold in os_folds
        for label, policy, gate_config in (
            ("v0_no_regime", "baseline", None),
            ("v1_fixed_regime", "fixed", cluster_config),
            ("v2_scheduled_refit", "scheduled", cluster_config),
        )
    ]

    comparison = []
    try:
        if args.workers == 1:
            for fold, label, policy, gate_config in jobs:
                print(f"running fold {fold.fold_id} {label}: {fold.os_start} -> {fold.os_end}", flush=True)
                comparison.append(
                    _run_fold_policy(
                        frame,
                        fold=fold,
                        label=label,
                        policy=policy,
                        gate_config=gate_config,
                        settings=settings,
                    )
                )
        else:
            print(f"running {len(jobs)} jobs with {args.workers} workers...", flush=True)
            with ProcessPoolExecutor(
                max_workers=args.workers,
                initializer=_init_worker,
                initargs=(frame, settings),
            ) as pool:
                future_to_job = {pool.submit(_run_fold_policy_worker, job): job for job in jobs}
                for future in as_completed(future_to_job):
                    fold, label, _, _ = future_to_job[future]
                    comparison.append(future.result())
                    print(f"completed fold {fold.fold_id} {label}", flush=True)
    except ValueError as exc:
        raise SystemExit(f"error: {exc}") from None

    comparison_frame = pd.DataFrame(comparison)
    if not comparison_frame.empty:
        comparison_frame = comparison_frame.sort_values(["fold_id", "label"]).reset_index(drop=True)
        first_columns = ["fold_id", "os_start", "os_end", "label"]
        comparison_frame = comparison_frame[
            first_columns + [column for column in comparison_frame.columns if column not in first_columns]
        ]
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = output_dir / f"{args.prefix}_comparison.csv"
    comparison_frame.to_csv(comparison_path, index=False)
    aggregate_frame = _aggregate_walk_forward(comparison_frame)
    aggregate_path = output_dir / f"{args.prefix}_aggregate.csv"
    aggregate_frame.to_csv(aggregate_path, index=False)
    deltas_frame = _paired_policy_deltas(comparison_frame)
    deltas_path = output_dir / f"{args.prefix}_paired_deltas.csv"
    deltas_frame.to_csv(deltas_path, index=False)
    print(comparison_frame.to_string(index=False))
    print(comparison_path)
    if not aggregate_frame.empty:
        print(aggregate_frame.to_string(index=False))
        print(aggregate_path)
    if not deltas_frame.empty:
        print(deltas_frame.to_string(index=False))
        print(deltas_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

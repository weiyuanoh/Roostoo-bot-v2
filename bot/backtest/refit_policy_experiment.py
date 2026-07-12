"""Compare fixed and scheduled OOS refit policies for ridge + cluster gates."""

from __future__ import annotations

import argparse
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
    load_model_frame,
    score_frame,
    select_ridge_model,
)


DEFAULT_OUTPUT_DIR = Path("reports/backtests/refit_policy")
DEFAULT_MODEL = "momentum_only"


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


def ridge_training_slice(
    frame: pd.DataFrame,
    *,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    horizon: int,
) -> pd.DataFrame:
    """Select ridge training rows whose labels are known before train_end."""
    cutoff = pd.Timestamp(train_end) - _bar_delta(frame) * horizon
    mask = (
        (frame["timestamp"] >= pd.Timestamp(train_start))
        & (frame["timestamp"] < pd.Timestamp(train_end))
        & (frame["timestamp"] <= cutoff)
    )
    return frame[mask].copy()


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
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run one latest-window OOS policy and return summary, equity, trades, segments."""
    if policy not in {"baseline", "fixed", "scheduled"}:
        raise ValueError("policy must be one of: baseline, fixed, scheduled")
    if not 0 < position_fraction <= 1:
        raise ValueError("position_fraction must be in (0, 1]")
    if take_profit <= 0 or stop_loss <= 0:
        raise ValueError("take_profit and stop_loss must be positive")
    max_positions = max_positions or max(1, int(np.floor(1 / position_fraction)))
    os_start, os_end = latest_oos_window(
        frame,
        os_months=os_months,
        required_train_months=max(ridge_train_months, regime_train_months),
    )
    segments = build_refit_segments(
        os_start,
        os_end,
        ridge_refit_days=ridge_refit_days,
        regime_refit_months=regime_refit_months,
        scheduled=policy == "scheduled",
    )

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
    times = pd.Series(pd.to_datetime(frame["timestamp"], utc=True).drop_duplicates().sort_values())
    if len(times) < 2:
        return pd.Timedelta(0)
    deltas = times.diff().dropna()
    return pd.Timedelta(deltas.median())


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
    cluster_config = ClusterRegimeGateConfig(
        n_clusters=args.cluster_n_clusters,
        min_cluster_trades=args.cluster_min_trades,
        lookback_months=args.regime_train_months,
        random_seed=args.cluster_random_seed,
    )

    comparison = []
    try:
        for label, policy, gate_config in (
            ("baseline", "baseline", None),
            ("fixed_cluster_gate", "fixed", cluster_config),
            ("scheduled_cluster_gate", "scheduled", cluster_config),
        ):
            summary, equity, trades, segments = run_refit_policy_backtest(
                frame,
                policy=policy,
                model=args.model,
                horizon=args.horizon,
                os_months=args.os_months,
                ridge_train_months=args.ridge_train_months,
                regime_train_months=args.regime_train_months,
                ridge_refit_days=args.ridge_refit_days,
                regime_refit_months=args.regime_refit_months,
                ridge_alphas=ridge_alphas,
                initial_cash=args.initial_cash,
                position_fraction=args.position_fraction,
                take_profit=args.tp,
                stop_loss=args.sl,
                fee_bps=args.fee_bps,
                slippage_bps=args.slippage_bps,
                max_positions=args.max_positions,
                top_k=args.top_k,
                max_new_entries=args.max_new_entries,
                cluster_gate_config=gate_config,
            )
            run_prefix = f"{args.prefix}_{label}_slip{_tag_number(args.slippage_bps)}"
            metadata = {
                "label": label,
                "policy": policy,
                "feature_path": str(feature_path),
                "model": args.model,
                "horizon": args.horizon,
                "os_months": args.os_months,
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
                "cluster_regime_gate": gate_config is not None,
                "cluster_n_clusters": args.cluster_n_clusters,
                "cluster_min_trades": args.cluster_min_trades,
                "cluster_random_seed": args.cluster_random_seed,
                "pair_universe": sorted(frame["pair"].unique()),
                "rows": int(len(frame)),
                "start": str(frame["timestamp"].min()),
                "end": str(frame["timestamp"].max()),
            }
            write_refit_outputs(summary, equity, trades, segments, output_dir, run_prefix, metadata)
            comparison.append(_comparison_row(label, summary))
    except ValueError as exc:
        raise SystemExit(f"error: {exc}") from None

    comparison_frame = pd.DataFrame(comparison)
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = output_dir / f"{args.prefix}_comparison.csv"
    comparison_frame.to_csv(comparison_path, index=False)
    print(comparison_frame.to_string(index=False))
    print(comparison_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

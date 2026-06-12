"""Portfolio backtest for the ridge momentum + roll-impact score.

This script deliberately imports score construction, fold construction, and
ridge fitting from `bot.strategy.ridge`, the same source of truth used by live
trading. Only portfolio accounting and trade simulation live here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from bot.backtest.portfolio import Portfolio, compute_equity_metrics
from bot.backtest.simulated_executor import SimulatedExecutor
from bot.strategy.ridge import (
    DEFAULT_FEATURE_PATH,
    DEFAULT_HORIZON,
    DEFAULT_IS_MONTHS,
    DEFAULT_OS_MONTHS,
    DEFAULT_RIDGE_ALPHAS,
    DEFAULT_STEP_MONTHS,
    build_cycle_intents,
    make_folds,
    load_model_frame,
    score_frame,
    select_ridge_model,
)


DEFAULT_OUTPUT_DIR = Path("notebooks/microstructure")
DEFAULT_MODEL = "momentum_plus_roll_plus_interaction"
DEFAULT_INITIAL_CASH = 1_000_000.0
DEFAULT_POSITION_FRACTION = 1 / 20
DEFAULT_EXIT_THRESHOLD = 0.50
DEFAULT_FEE_BPS = 10.0
DEFAULT_SLIPPAGE_BPS = 0.0


def run_portfolio_backtest(
    frame: pd.DataFrame,
    *,
    model: str = DEFAULT_MODEL,
    horizon: int = DEFAULT_HORIZON,
    is_months: int = DEFAULT_IS_MONTHS,
    os_months: int = DEFAULT_OS_MONTHS,
    step_months: int = DEFAULT_STEP_MONTHS,
    ridge_alphas: tuple[float, ...] = DEFAULT_RIDGE_ALPHAS,
    initial_cash: float = DEFAULT_INITIAL_CASH,
    position_fraction: float = DEFAULT_POSITION_FRACTION,
    exit_threshold: float = DEFAULT_EXIT_THRESHOLD,
    take_profit: float | None = None,
    stop_loss: float | None = None,
    fee_bps: float = DEFAULT_FEE_BPS,
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS,
    max_positions: int | None = None,
    top_k: int | None = None,
    max_new_entries: int | None = None,
    exchange_info: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not 0 < position_fraction <= 1:
        raise ValueError("position_fraction must be in (0, 1]")
    tp = exit_threshold if take_profit is None else take_profit
    sl = exit_threshold if stop_loss is None else stop_loss
    if not 0 < tp:
        raise ValueError("take_profit must be positive")
    if not 0 < sl:
        raise ValueError("stop_loss must be positive")
    if top_k is not None and top_k <= 0:
        raise ValueError("top_k must be positive when set")
    if max_new_entries is not None and max_new_entries <= 0:
        raise ValueError("max_new_entries must be positive when set")
    max_positions = max_positions or max(1, int(np.floor(1 / position_fraction)))
    folds = make_folds(frame, is_months=is_months, os_months=os_months, step_months=step_months)
    target_col = f"forward_return_{horizon}"

    equity_rows: list[dict] = []
    trade_rows: list[dict] = []
    fold_rows: list[dict] = []

    for fold in folds.itertuples(index=False):
        is_mask = (frame["timestamp"] >= fold.is_start) & (frame["timestamp"] < fold.is_end)
        os_mask = (frame["timestamp"] >= fold.os_start) & (frame["timestamp"] < fold.os_end)
        train = frame[is_mask].copy()
        test = frame[os_mask].copy()
        selection = select_ridge_model(
            train,
            model,
            target_col=target_col,
            ridge_alphas=ridge_alphas,
        )
        scored = score_frame(test, selection.terms, selection.beta, score_col="ridge_score")
        portfolio = Portfolio(initial_cash=initial_cash, fee_bps=fee_bps)
        executor = SimulatedExecutor(
            portfolio,
            exchange_info=exchange_info,
            slippage_bps=slippage_bps,
        )

        for open_time, bar in scored.sort_values(["open_time", "pair"]).groupby(
            "open_time",
            observed=True,
        ):
            timestamp = bar["timestamp"].iloc[0]
            prices = {
                str(row.pair): float(row.close)
                for row in bar[["pair", "close"]].itertuples(index=False)
                if pd.notna(row.close) and float(row.close) > 0
            }

            equity_value = portfolio.value(prices)
            cycle = build_cycle_intents(
                bar,
                portfolio.positions,
                prices,
                portfolio_value=equity_value,
                available_cash=portfolio.cash,
                position_fraction=position_fraction,
                max_positions=max_positions,
                top_k=top_k,
                max_new_entries=max_new_entries,
                take_profit=tp,
                stop_loss=sl,
            )

            for intent in cycle.exits:
                executor.set_context(timestamp=open_time, reason=intent.reason)
                before = len(executor.events)
                executor.sell(intent.pair, intent.quantity, prices[intent.pair], use_limit=False)
                if len(executor.events) > before:
                    executor.events[-1]["fold"] = fold.fold
                    trade_rows.append(executor.events[-1])

            for intent in cycle.entries:
                executor.set_context(
                    timestamp=open_time,
                    score=intent.score,
                    reason=intent.reason,
                )
                before = len(executor.events)
                executor.buy(intent.pair, intent.notional_usd, prices[intent.pair], use_limit=False)
                if len(executor.events) > before:
                    executor.events[-1]["fold"] = fold.fold
                    trade_rows.append(executor.events[-1])

            equity_after = portfolio.value(prices)
            equity_rows.append(
                {
                    "fold": fold.fold,
                    "open_time": open_time,
                    "timestamp": timestamp,
                    "equity": equity_after,
                    "cash": portfolio.cash,
                    "positions": len(portfolio.positions),
                }
            )

        final_prices = (
            scored.sort_values("open_time")
            .groupby("pair", observed=True)["close"]
            .last()
            .dropna()
            .astype(float)
            .to_dict()
        )
        for pair in list(portfolio.positions):
            price = final_prices.get(pair)
            if price is None:
                continue
            executor.set_context(timestamp="fold_end", reason="fold_end")
            before = len(executor.events)
            executor.sell(pair, portfolio.positions[pair].quantity, price, use_limit=False)
            if len(executor.events) > before:
                executor.events[-1]["fold"] = fold.fold
                trade_rows.append(executor.events[-1])

        fold_equity = [row["equity"] for row in equity_rows if row["fold"] == fold.fold]
        metrics = compute_equity_metrics(fold_equity, initial_cash)
        exits = [row for row in trade_rows if row.get("fold") == fold.fold and row["side"] == "SELL"]
        fold_rows.append(
            {
                "fold": fold.fold,
                "model": model,
                "alpha": selection.alpha,
                "is_start": fold.is_start,
                "is_end": fold.is_end,
                "os_start": fold.os_start,
                "os_end": fold.os_end,
                "terms": ",".join(selection.terms),
                "beta": ",".join(f"{value:.12g}" for value in selection.beta),
                "is_mean_spearman": selection.is_mean_spearman,
                "is_spearman_hit_rate": selection.is_spearman_hit_rate,
                "trades": len([row for row in trade_rows if row.get("fold") == fold.fold]),
                "closed_trades": len(exits),
                "win_rate": float(np.mean([row.get("pnl", 0) > 0 for row in exits])) if exits else np.nan,
                **metrics,
            }
        )

    return pd.DataFrame(fold_rows), pd.DataFrame(equity_rows), pd.DataFrame(trade_rows)


def write_outputs(
    summary: pd.DataFrame,
    equity: pd.DataFrame,
    trades: pd.DataFrame,
    output_dir: Path,
    prefix: str,
    metadata: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / f"{prefix}_summary.csv", index=False)
    equity.to_csv(output_dir / f"{prefix}_equity.csv", index=False)
    trades.to_csv(output_dir / f"{prefix}_trades.csv", index=False)
    (output_dir / f"{prefix}_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )


def parse_alphas(value: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in value.split(",") if part.strip())


def main() -> int:
    parser = argparse.ArgumentParser(description="Portfolio backtest for ridge score trading")
    parser.add_argument("--features", default=str(DEFAULT_FEATURE_PATH), help="Feature CSV path")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory")
    parser.add_argument("--prefix", default="ridge_score_portfolio_1h", help="Output filename prefix")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ridge model name")
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--is-months", type=int, default=DEFAULT_IS_MONTHS)
    parser.add_argument("--os-months", type=int, default=DEFAULT_OS_MONTHS)
    parser.add_argument("--step-months", type=int, default=DEFAULT_STEP_MONTHS)
    parser.add_argument("--ridge-alphas", default=",".join(str(alpha) for alpha in DEFAULT_RIDGE_ALPHAS))
    parser.add_argument("--initial-cash", type=float, default=DEFAULT_INITIAL_CASH)
    parser.add_argument("--position-fraction", type=float, default=DEFAULT_POSITION_FRACTION)
    parser.add_argument(
        "--exit-threshold",
        type=float,
        default=None,
        help="Deprecated: set both --tp and --sl to the same value.",
    )
    parser.add_argument("--tp", type=float, default=DEFAULT_EXIT_THRESHOLD)
    parser.add_argument("--sl", type=float, default=DEFAULT_EXIT_THRESHOLD)
    parser.add_argument("--fee-bps", type=float, default=DEFAULT_FEE_BPS)
    parser.add_argument("--slippage-bps", type=float, default=DEFAULT_SLIPPAGE_BPS)
    parser.add_argument("--max-positions", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--max-new-entries", type=int, default=None)
    args = parser.parse_args()

    feature_path = Path(args.features)
    frame = load_model_frame(feature_path, horizon=args.horizon)
    ridge_alphas = parse_alphas(args.ridge_alphas)
    summary, equity, trades = run_portfolio_backtest(
        frame,
        model=args.model,
        horizon=args.horizon,
        is_months=args.is_months,
        os_months=args.os_months,
        step_months=args.step_months,
        ridge_alphas=ridge_alphas,
        initial_cash=args.initial_cash,
        position_fraction=args.position_fraction,
        exit_threshold=DEFAULT_EXIT_THRESHOLD if args.exit_threshold is None else args.exit_threshold,
        take_profit=args.exit_threshold if args.exit_threshold is not None else args.tp,
        stop_loss=args.exit_threshold if args.exit_threshold is not None else args.sl,
        fee_bps=args.fee_bps,
        slippage_bps=args.slippage_bps,
        max_positions=args.max_positions,
        top_k=args.top_k,
        max_new_entries=args.max_new_entries,
    )
    metadata = {
        "feature_path": str(feature_path),
        "model": args.model,
        "horizon": args.horizon,
        "is_months": args.is_months,
        "os_months": args.os_months,
        "step_months": args.step_months,
        "ridge_alphas": ridge_alphas,
        "initial_cash": args.initial_cash,
        "position_fraction": args.position_fraction,
        "take_profit": args.exit_threshold if args.exit_threshold is not None else args.tp,
        "stop_loss": args.exit_threshold if args.exit_threshold is not None else args.sl,
        "exit_threshold": args.exit_threshold,
        "fee_bps": args.fee_bps,
        "slippage_bps": args.slippage_bps,
        "max_positions": args.max_positions or max(1, int(np.floor(1 / args.position_fraction))),
        "top_k": args.top_k,
        "max_new_entries": args.max_new_entries,
        "rows": int(len(frame)),
        "pairs": int(frame["pair"].nunique()),
        "start": str(frame["timestamp"].min()),
        "end": str(frame["timestamp"].max()),
    }
    write_outputs(summary, equity, trades, Path(args.output_dir), args.prefix, metadata)
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

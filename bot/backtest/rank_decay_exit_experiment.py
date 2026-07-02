"""Compare TP/SL-only exits against rank-decay exits for momentum-only."""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from bot.backtest.ridge_score_portfolio import (
    DEFAULT_INITIAL_CASH,
    filter_pairs,
    parse_alphas,
    parse_pairs,
    run_portfolio_backtest,
    write_outputs,
)
from bot.strategy.ridge import (
    add_ridge_signals,
    add_training_target,
    DEFAULT_FEATURE_PATH,
    DEFAULT_HORIZON,
    DEFAULT_IS_MONTHS,
    DEFAULT_OS_MONTHS,
    DEFAULT_RIDGE_ALPHAS,
    DEFAULT_STEP_MONTHS,
)


FIXED_TOP30_PAIRS = (
    "BTC/USD",
    "ETH/USD",
    "SOL/USD",
    "XRP/USD",
    "ZEC/USD",
    "BNB/USD",
    "DOGE/USD",
    "SUI/USD",
    "TRX/USD",
    "ADA/USD",
    "PEPE/USD",
    "PAXG/USD",
    "LINK/USD",
    "TAO/USD",
    "AVAX/USD",
    "NEAR/USD",
    "LTC/USD",
    "ENA/USD",
    "UNI/USD",
    "WLD/USD",
    "AAVE/USD",
    "HBAR/USD",
    "FET/USD",
    "FIL/USD",
    "TRUMP/USD",
    "TON/USD",
    "DOT/USD",
    "ICP/USD",
    "APT/USD",
    "VIRTUAL/USD",
)
DEFAULT_PAIRS = ",".join(FIXED_TOP30_PAIRS)
DEFAULT_OUTPUT_DIR = Path("reports/backtests")
UNIVERSE_NAME = "fixed_top30_median_dollar_volume"
UNIVERSE_SELECTION = "top 30 by median hourly dollar volume from feature file; fixed before backtest"


def main() -> int:
    parser = argparse.ArgumentParser(description="Rank-decay exit experiment")
    parser.add_argument("--features", default=str(DEFAULT_FEATURE_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--prefix", default="rank_decay_exit_top3_top30")
    parser.add_argument("--pairs", default=DEFAULT_PAIRS)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--is-months", type=int, default=DEFAULT_IS_MONTHS)
    parser.add_argument("--os-months", type=int, default=DEFAULT_OS_MONTHS)
    parser.add_argument("--step-months", type=int, default=DEFAULT_STEP_MONTHS)
    parser.add_argument("--ridge-alphas", default=",".join(str(alpha) for alpha in DEFAULT_RIDGE_ALPHAS))
    parser.add_argument("--initial-cash", type=float, default=DEFAULT_INITIAL_CASH)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--max-new-entries", type=int, default=1)
    parser.add_argument("--max-positions", type=int, default=3)
    parser.add_argument("--position-fraction", type=float, default=0.25)
    parser.add_argument("--tp", type=float, default=0.03)
    parser.add_argument("--sl", type=float, default=0.015)
    parser.add_argument("--fee-bps", type=float, default=10.0)
    parser.add_argument("--slippage-bps", default="0,5")
    parser.add_argument("--rank-exit-threshold", type=int, default=3)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    feature_path = Path(args.features)
    pairs = parse_pairs(args.pairs)
    frame = build_model_frame_for_universe(feature_path, pairs, horizon=args.horizon)
    ridge_alphas = parse_alphas(args.ridge_alphas)
    slippage_values = tuple(float(part.strip()) for part in args.slippage_bps.split(",") if part.strip())

    results = []
    artifacts: list[str] = []
    for slippage_bps in slippage_values:
        for label, rank_exit_threshold in (
            ("baseline", None),
            ("candidate", args.rank_exit_threshold),
        ):
            prefix = f"{args.prefix}_{label}_slip{_tag_number(slippage_bps)}"
            summary, equity, trades = run_portfolio_backtest(
                frame,
                model="momentum_only",
                horizon=args.horizon,
                is_months=args.is_months,
                os_months=args.os_months,
                step_months=args.step_months,
                ridge_alphas=ridge_alphas,
                initial_cash=args.initial_cash,
                position_fraction=args.position_fraction,
                take_profit=args.tp,
                stop_loss=args.sl,
                fee_bps=args.fee_bps,
                slippage_bps=slippage_bps,
                max_positions=args.max_positions,
                top_k=args.top_k,
                max_new_entries=args.max_new_entries,
                rank_exit_threshold=rank_exit_threshold,
            )
            metadata = {
                "label": label,
                "feature_path": str(feature_path),
                "universe_name": universe_name(pairs),
                "universe_size": int(frame["pair"].nunique()),
                "universe_selection": universe_selection(pairs),
                "model": "momentum_only",
                "horizon": args.horizon,
                "is_months": args.is_months,
                "os_months": args.os_months,
                "step_months": args.step_months,
                "ridge_alphas": ridge_alphas,
                "initial_cash": args.initial_cash,
                "position_fraction": args.position_fraction,
                "take_profit": args.tp,
                "stop_loss": args.sl,
                "fee_bps": args.fee_bps,
                "slippage_bps": slippage_bps,
                "max_positions": args.max_positions,
                "top_k": args.top_k,
                "max_new_entries": args.max_new_entries,
                "rank_exit_threshold": rank_exit_threshold,
                "pair_universe": list(pairs) if pairs is not None else sorted(frame["pair"].unique()),
                "rows": int(len(frame)),
                "start": str(frame["timestamp"].min()),
                "end": str(frame["timestamp"].max()),
            }
            write_outputs(summary, equity, trades, output_dir, prefix, metadata)
            artifacts.extend(
                [
                    f"{prefix}_summary.csv",
                    f"{prefix}_equity.csv",
                    f"{prefix}_trades.csv",
                    f"{prefix}_metadata.json",
                ]
            )
            results.append(
                {
                    "label": label,
                    "slippage_bps": slippage_bps,
                    "summary": summary,
                    "trades": trades,
                    "prefix": prefix,
                }
            )

    report = build_report(args, frame, results, artifacts)
    report_path = output_dir / f"{args.prefix}.md"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    print(f"Wrote {report_path}")
    return 0


def build_model_frame_for_universe(
    feature_path: str | Path,
    pairs: tuple[str, ...] | None,
    *,
    horizon: int = DEFAULT_HORIZON,
) -> pd.DataFrame:
    """Build model features after restricting the raw frame to the traded universe."""
    raw = pd.read_csv(feature_path)
    filtered = filter_pairs(raw, pairs)
    frame = add_ridge_signals(filtered)
    frame = add_training_target(frame, horizon=horizon)
    return frame.sort_values(["pair", "timestamp"]).reset_index(drop=True)


def universe_name(pairs: tuple[str, ...] | None) -> str:
    if pairs == FIXED_TOP30_PAIRS:
        return UNIVERSE_NAME
    return "custom_pair_override" if pairs is not None else "all_pairs_override"


def universe_selection(pairs: tuple[str, ...] | None) -> str:
    if pairs == FIXED_TOP30_PAIRS:
        return UNIVERSE_SELECTION
    return "--pairs override supplied by caller"


def build_report(
    args: argparse.Namespace,
    frame: pd.DataFrame,
    results: list[dict],
    artifacts: list[str],
) -> str:
    lines = [
        "# Backtest: rank-decay exit for momentum-only",
        "",
        f"Run date: {date.today().isoformat()}",
        "",
        "## Experiment",
        "",
        "Baseline uses the current momentum-only score with TP/SL exits. Candidate uses the same score, sizing, entries, fees, and slippage, but also exits a held pair when its current score rank is worse than the configured threshold.",
        "",
        "```text",
        "score[i,t] = beta_momentum * z_momentum[i,t]",
        "",
        f"if held_pair_rank[t] > {args.rank_exit_threshold}:",
        "    exit position",
        "```",
        "",
        "The rank-decay exit is evaluated before TP/SL. Same-cycle re-entry of the exiting pair is blocked by the existing held-pair entry logic.",
        "",
        "## Parameters",
        "",
        f"- Universe name: `{universe_name(parse_pairs(args.pairs))}`",
        f"- Universe size: `{frame['pair'].nunique()}`",
        f"- Universe selection: {universe_selection(parse_pairs(args.pairs))}",
        f"- Pairs: `{', '.join(parse_pairs(args.pairs) or tuple(sorted(frame['pair'].unique())))}`",
        "- Model: `momentum_only`",
        f"- Horizon: `{args.horizon}`",
        f"- IS/OS: `{args.is_months}m/{args.os_months}m`",
        f"- Top K: `{args.top_k}`",
        f"- Max new entries: `{args.max_new_entries}`",
        f"- Max positions: `{args.max_positions}`",
        f"- Position fraction: `{args.position_fraction}`",
        f"- TP/SL: `{args.tp}` / `{args.sl}`",
        f"- Fee bps: `{args.fee_bps}`",
        f"- Slippage bps: `{args.slippage_bps}`",
        f"- Rank exit threshold: `{args.rank_exit_threshold}`",
        "",
        "## Comparison",
        "",
        _markdown_table(
            [
                "Run",
                "Slippage",
                "Return",
                "Max DD",
                "Ret/DD",
                "Sharpe",
                "Sortino",
                "Win",
                "Stop",
                "TP",
                "Rank exits",
                "Median hold",
                "Trades",
            ],
            [_comparison_row(result) for result in results],
        ),
        "",
        "## Pair attribution",
        "",
    ]
    for result in results:
        lines.extend(
            [
                f"### {result['label']} slippage={result['slippage_bps']:g} bps",
                "",
                _pair_attribution_table(result["trades"]),
                "",
            ]
        )
    lines.extend(["## Artifacts", ""])
    lines.extend(f"- `{Path('reports/backtests') / artifact}`" for artifact in artifacts)
    lines.append("")
    return "\n".join(lines)


def _comparison_row(result: dict) -> list[str]:
    summary = result["summary"]
    trades = result["trades"]
    row = summary.iloc[-1] if not summary.empty else pd.Series(dtype=float)
    exits = trades[trades["side"].eq("SELL")] if not trades.empty else pd.DataFrame()
    win_rate = float((exits["pnl"] > 0).mean() * 100) if not exits.empty else np.nan
    stop_rate = float((exits["reason"].eq("loss_threshold")).mean() * 100) if not exits.empty else np.nan
    tp_rate = float((exits["reason"].eq("gain_threshold")).mean() * 100) if not exits.empty else np.nan
    return [
        result["label"],
        f"{result['slippage_bps']:g}",
        _pct(row.get("total_return_pct")),
        _pct(row.get("max_drawdown_pct")),
        _num(row.get("return_over_max_drawdown")),
        _num(row.get("sharpe")),
        _num(row.get("sortino")),
        _pct(win_rate),
        _pct(stop_rate),
        _pct(tp_rate),
        str(int(row.get("rank_decay_exits", 0) or 0)),
        _hours(row.get("median_holding_hours")),
        str(int(row.get("trades", 0) or 0)),
    ]


def _pair_attribution_table(trades: pd.DataFrame) -> str:
    if trades.empty or "side" not in trades.columns:
        return "_No trades._"
    exits = trades[trades["side"].eq("SELL")].copy()
    if exits.empty:
        return "_No closed trades._"
    grouped = exits.groupby("pair", observed=True)
    rows = []
    for pair, group in grouped:
        rows.append(
            [
                str(pair),
                str(len(group)),
                _pct(float((group["pnl"] > 0).mean() * 100)),
                _pct(float(group["return_pct"].mean() * 100)),
                _pct(float(group["return_pct"].median() * 100)),
                _usd(float(group["pnl"].sum())),
            ]
        )
    rows.sort(key=lambda item: float(item[-1].replace(",", "")), reverse=True)
    return _markdown_table(["Pair", "Closed", "Win", "Avg return", "Median return", "PnL"], rows)


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def _pct(value) -> str:
    return "n/a" if pd.isna(value) else f"{float(value):.2f}%"


def _num(value) -> str:
    return "n/a" if pd.isna(value) else f"{float(value):.3f}"


def _hours(value) -> str:
    return "n/a" if pd.isna(value) else f"{float(value):.2f}h"


def _usd(value: float) -> str:
    return f"{value:,.0f}"


def _tag_number(value: float) -> str:
    return str(value).replace(".", "p").rstrip("0").rstrip("p")


if __name__ == "__main__":
    raise SystemExit(main())

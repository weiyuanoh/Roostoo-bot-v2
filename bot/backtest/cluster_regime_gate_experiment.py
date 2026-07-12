"""Compare momentum-only baseline against the learned regime cluster gate."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from bot.backtest.ridge_score_portfolio import (
    DEFAULT_INITIAL_CASH,
    filter_pairs,
    parse_alphas,
    parse_pairs,
    run_portfolio_backtest,
    write_outputs,
)
from bot.strategy.regime import ClusterRegimeGateConfig
from bot.strategy.ridge import (
    DEFAULT_FEATURE_PATH,
    DEFAULT_HORIZON,
    DEFAULT_RIDGE_ALPHAS,
    load_model_frame,
)

DEFAULT_OUTPUT_DIR = Path("reports/backtests/cluster_regime")


def main() -> int:
    parser = argparse.ArgumentParser(description="Learned regime cluster gate experiment")
    parser.add_argument("--features", default=str(DEFAULT_FEATURE_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--prefix", default="momentum_only_cluster_gate")
    parser.add_argument("--pairs", default="all")
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--is-months", type=int, default=3)
    parser.add_argument("--os-months", type=int, default=2)
    parser.add_argument("--step-months", type=int, default=1)
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
    parser.add_argument("--cluster-lookback-months", type=int, default=24)
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
        lookback_months=args.cluster_lookback_months,
        random_seed=args.cluster_random_seed,
    )

    comparison = []
    for label, config in (("baseline", None), ("cluster_gate", cluster_config)):
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
            slippage_bps=args.slippage_bps,
            max_positions=args.max_positions,
            top_k=args.top_k,
            max_new_entries=args.max_new_entries,
            cluster_gate_config=config,
        )
        run_prefix = f"{args.prefix}_{label}_slip{_tag_number(args.slippage_bps)}"
        metadata = {
            "label": label,
            "feature_path": str(feature_path),
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
            "slippage_bps": args.slippage_bps,
            "max_positions": args.max_positions,
            "top_k": args.top_k,
            "max_new_entries": args.max_new_entries,
            "cluster_regime_gate": config is not None,
            "cluster_lookback_months": args.cluster_lookback_months,
            "cluster_n_clusters": args.cluster_n_clusters,
            "cluster_min_trades": args.cluster_min_trades,
            "cluster_random_seed": args.cluster_random_seed,
            "pair_universe": sorted(frame["pair"].unique()),
            "rows": int(len(frame)),
            "start": str(frame["timestamp"].min()),
            "end": str(frame["timestamp"].max()),
        }
        write_outputs(summary, equity, trades, output_dir, run_prefix, metadata)
        comparison.append(_comparison_row(label, args.slippage_bps, summary))

    comparison_frame = pd.DataFrame(comparison)
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = output_dir / f"{args.prefix}_comparison.csv"
    comparison_frame.to_csv(comparison_path, index=False)
    print(comparison_frame.to_string(index=False))
    print(comparison_path)
    return 0


def _comparison_row(label: str, slippage_bps: float, summary: pd.DataFrame) -> dict[str, float | str]:
    if summary.empty:
        return {"label": label, "slippage_bps": slippage_bps}
    return {
        "label": label,
        "slippage_bps": slippage_bps,
        "folds": int(len(summary)),
        "mean_return_pct": float(summary["total_return_pct"].mean()),
        "sum_return_pct": float(summary["total_return_pct"].sum()),
        "mean_max_drawdown_pct": float(summary["max_drawdown_pct"].mean()),
        "closed_trades": int(summary["closed_trades"].sum()),
        "tp_exits": int(summary["tp_exits"].sum()),
        "stop_exits": int(summary["stop_exits"].sum()),
        "cluster_gate_checks": int(summary.get("cluster_gate_checks", pd.Series(dtype=float)).sum()),
        "cluster_gate_blocked_checks": int(summary.get("cluster_gate_blocked_checks", pd.Series(dtype=float)).sum()),
    }


def _tag_number(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else str(value).replace(".", "p")


if __name__ == "__main__":
    raise SystemExit(main())

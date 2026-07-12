"""Market-regime helpers shared by live and backtest trading."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

PERSISTENCE_WINDOWS = (3, 6, 12)
CLUSTER_FEATURE_COLUMNS = (
    "entry_score",
    "score_gap_rank1_rank2",
    "score_gap_rank1_median",
    "score_gap_rank1_rank2_trailing_median",
    "pair_return_1h_known",
    "universe_breadth_1h",
    "universe_return_1h",
    "universe_volatility_1h",
    "pair_top3_count_3h",
    "pair_top3_count_6h",
    "pair_top3_count_12h",
    "pair_rank1_count_3h",
    "pair_rank1_count_6h",
    "pair_rank1_count_12h",
    "positions_at_entry",
)


@dataclass(frozen=True)
class RegimeThrottleConfig:
    metric_col: str = "roll_impact"
    aggregation: str = "median"
    lookback_bars: int = 720
    percentile: float = 0.80
    min_history_bars: int = 168
    time_col: str = "open_time"
    market_col: str = "market_roll_impact"
    threshold_col: str = "market_roll_impact_threshold"
    stressed_col: str = "market_roll_stressed"
    history_col: str = "market_roll_history_bars"

    def __post_init__(self) -> None:
        if self.lookback_bars <= 0:
            raise ValueError("lookback_bars must be positive")
        if not 0 <= self.percentile <= 1:
            raise ValueError("percentile must be in [0, 1]")
        if self.min_history_bars <= 0:
            raise ValueError("min_history_bars must be positive")
        if self.aggregation not in {"median", "mean"}:
            raise ValueError("aggregation must be 'median' or 'mean'")


@dataclass(frozen=True)
class RegimeDecision:
    is_stressed: bool
    market_roll_impact: float | None
    threshold: float | None
    history_bars: int
    entries_blocked: bool
    reason: str


@dataclass(frozen=True)
class ClusterRegimeGateConfig:
    """Settings for the learned cluster entry gate."""

    n_clusters: int = 4
    min_cluster_trades: int = 50
    min_cluster_mean_return: float = 0.0
    require_better_than_training_average: bool = True
    lookback_months: int = 24
    random_seed: int = 42
    feature_columns: tuple[str, ...] = CLUSTER_FEATURE_COLUMNS

    def __post_init__(self) -> None:
        if self.n_clusters <= 0:
            raise ValueError("n_clusters must be positive")
        if self.min_cluster_trades <= 0:
            raise ValueError("min_cluster_trades must be positive")
        if self.lookback_months <= 0:
            raise ValueError("lookback_months must be positive")
        if not self.feature_columns:
            raise ValueError("feature_columns must not be empty")


@dataclass(frozen=True)
class ClusterGateDecision:
    """Decision for one candidate entry assigned to a historical regime cluster."""

    enabled: bool
    allowed: bool
    cluster_id: int | None
    distance: float | None
    reason: str


@dataclass(frozen=True)
class ClusterRegimeGate:
    """Frozen cluster gate trained from prior simulated entries."""

    config: ClusterRegimeGateConfig
    feature_columns: tuple[str, ...]
    means: pd.Series
    stds: pd.Series
    centroids: np.ndarray
    allowed_clusters: frozenset[int]
    training_average_return: float
    cluster_summary: pd.DataFrame = field(repr=False)
    cluster_profiles: pd.DataFrame = field(repr=False)

    def decide(self, row: pd.Series | dict[str, Any], *, positions_at_entry: int) -> ClusterGateDecision:
        if self.centroids.size == 0:
            return ClusterGateDecision(True, False, None, None, "no_centroids")
        values = _feature_vector(row, self.feature_columns, positions_at_entry=positions_at_entry)
        if values.isna().any():
            return ClusterGateDecision(True, False, None, None, "missing_features")
        scaled = ((values - self.means) / self.stds).to_numpy(dtype=float)
        distances = np.sqrt(((self.centroids - scaled) ** 2).sum(axis=1))
        cluster_id = int(np.argmin(distances))
        distance = float(distances[cluster_id])
        allowed = cluster_id in self.allowed_clusters
        return ClusterGateDecision(
            enabled=True,
            allowed=allowed,
            cluster_id=cluster_id,
            distance=distance,
            reason="allowed_cluster" if allowed else "blocked_cluster",
        )


def add_roll_impact_regime(
    frame: pd.DataFrame,
    config: RegimeThrottleConfig,
) -> pd.DataFrame:
    """Append universe-level roll-impact regime columns without lookahead."""
    missing = {config.time_col, config.metric_col} - set(frame.columns)
    if missing:
        raise ValueError(f"missing required regime columns: {', '.join(sorted(missing))}")

    out = frame.copy()
    grouped = out.groupby(config.time_col, observed=True)[config.metric_col]
    if config.aggregation == "mean":
        market = grouped.mean().sort_index()
    else:
        market = grouped.median().sort_index()
    prior_market = market.shift(1)
    history = prior_market.rolling(config.lookback_bars, min_periods=1).count()
    threshold = prior_market.rolling(
        config.lookback_bars,
        min_periods=config.min_history_bars,
    ).quantile(config.percentile)
    stressed = market.ge(threshold) & threshold.notna() & history.ge(config.min_history_bars)

    out[config.market_col] = out[config.time_col].map(market)
    out[config.threshold_col] = out[config.time_col].map(threshold)
    out[config.history_col] = out[config.time_col].map(history).fillna(0).astype(int)
    out[config.stressed_col] = out[config.time_col].map(stressed).fillna(False).astype(bool)
    return out


def add_decision_time_regime_features(
    scored: pd.DataFrame,
    *,
    score_col: str = "ridge_score",
    time_col: str = "open_time",
) -> pd.DataFrame:
    """Add rank, score-gap, persistence, and current/prior-only market features."""
    if scored.empty:
        return scored.copy()
    missing = {time_col, "pair", "close", score_col} - set(scored.columns)
    if missing:
        raise ValueError(f"missing decision-time regime columns: {', '.join(sorted(missing))}")
    df = scored.sort_values([time_col, "pair"]).copy()
    df["entry_rank"] = df.groupby(time_col, observed=True)[score_col].rank(
        method="first",
        ascending=False,
    )
    grouped = df.groupby(time_col, observed=True)
    df["rank1_score"] = grouped[score_col].transform("max")
    df["rank2_score"] = grouped[score_col].transform(_second_largest)
    df["median_score"] = grouped[score_col].transform("median")
    df["score_gap_rank1_rank2"] = df["rank1_score"] - df["rank2_score"]
    df["score_gap_rank1_median"] = df["rank1_score"] - df["median_score"]
    df["rank1_pair"] = df[time_col].map(_rank1_pairs(df, score_col=score_col, time_col=time_col))
    gaps = (
        df[[time_col, "score_gap_rank1_rank2"]]
        .drop_duplicates(time_col)
        .sort_values(time_col)
        .set_index(time_col)["score_gap_rank1_rank2"]
    )
    trailing_gap = gaps.expanding(min_periods=24).median().shift(1)
    df["score_gap_rank1_rank2_trailing_median"] = df[time_col].map(trailing_gap)

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
    df["universe_breadth_1h"] = df.groupby(time_col, observed=True)["pair_return_1h_known"].transform(
        lambda values: float((values > 0).mean())
    )
    df["universe_return_1h"] = df.groupby(time_col, observed=True)["pair_return_1h_known"].transform("mean")
    df["universe_volatility_1h"] = df.groupby(time_col, observed=True)["pair_return_1h_known"].transform("std")
    return df


def train_cluster_regime_gate(
    entries: pd.DataFrame,
    config: ClusterRegimeGateConfig,
) -> ClusterRegimeGate:
    """Fit a cluster gate from prior entries and realized outcomes only."""
    columns = list(config.feature_columns) + ["return_pct", "pnl"]
    available = [col for col in columns if col in entries.columns]
    data = entries[available].copy() if not entries.empty else pd.DataFrame(columns=columns)
    for col in config.feature_columns:
        data[col] = pd.to_numeric(data.get(col), errors="coerce")
    data["return_pct"] = pd.to_numeric(data.get("return_pct"), errors="coerce")
    data["pnl"] = pd.to_numeric(data.get("pnl"), errors="coerce")
    data = data.dropna(subset=[*config.feature_columns, "return_pct"]).reset_index(drop=True)
    empty = _empty_cluster_gate(config)
    if len(data) < config.n_clusters:
        return empty

    features = data[list(config.feature_columns)].astype(float)
    means = features.mean()
    stds = features.std(ddof=0).replace(0, 1.0)
    scaled = ((features - means) / stds).to_numpy(dtype=float)
    labels, centroids = _kmeans(scaled, config.n_clusters, seed=config.random_seed)
    data["cluster_id"] = labels
    training_average = float(data["return_pct"].mean())
    summary = (
        data.groupby("cluster_id", observed=True)
        .agg(
            entries=("return_pct", "size"),
            mean_return=("return_pct", "mean"),
            median_return=("return_pct", "median"),
            win_rate=("return_pct", lambda values: float((values > 0).mean())),
            total_pnl=("pnl", "sum"),
        )
        .reset_index()
    )
    summary["allowed"] = (
        (summary["entries"] >= config.min_cluster_trades)
        & (summary["mean_return"] >= config.min_cluster_mean_return)
    )
    if config.require_better_than_training_average:
        summary["allowed"] = summary["allowed"] & (summary["mean_return"] > training_average)
    profiles = data.groupby("cluster_id", observed=True)[list(config.feature_columns)].mean().reset_index()
    allowed = frozenset(int(row.cluster_id) for row in summary.itertuples(index=False) if bool(row.allowed))
    return ClusterRegimeGate(
        config=config,
        feature_columns=tuple(config.feature_columns),
        means=means,
        stds=stds,
        centroids=centroids,
        allowed_clusters=allowed,
        training_average_return=training_average,
        cluster_summary=summary,
        cluster_profiles=profiles,
    )


def decide_regime_throttle(
    scores_at_timestamp: pd.DataFrame,
    config: RegimeThrottleConfig,
) -> RegimeDecision:
    """Return the entry-throttle decision for one scored timestamp."""
    if scores_at_timestamp.empty:
        return RegimeDecision(
            is_stressed=False,
            market_roll_impact=None,
            threshold=None,
            history_bars=0,
            entries_blocked=False,
            reason="empty_scores",
        )

    required = {
        config.market_col,
        config.threshold_col,
        config.stressed_col,
        config.history_col,
    }
    missing = required - set(scores_at_timestamp.columns)
    if missing:
        raise ValueError(f"missing regime decision columns: {', '.join(sorted(missing))}")

    row = scores_at_timestamp.iloc[0]
    market_value = _none_if_na(row[config.market_col])
    threshold = _none_if_na(row[config.threshold_col])
    history_bars = int(row[config.history_col]) if pd.notna(row[config.history_col]) else 0
    is_stressed = bool(row[config.stressed_col])

    if history_bars < config.min_history_bars:
        reason = "insufficient_history"
        is_stressed = False
    elif threshold is None:
        reason = "missing_threshold"
        is_stressed = False
    elif is_stressed:
        reason = "market_roll_impact_stressed"
    else:
        reason = "market_roll_impact_normal"

    return RegimeDecision(
        is_stressed=is_stressed,
        market_roll_impact=market_value,
        threshold=threshold,
        history_bars=history_bars,
        entries_blocked=is_stressed,
        reason=reason,
    )


def _feature_vector(
    row: pd.Series | dict[str, Any],
    feature_columns: tuple[str, ...],
    *,
    positions_at_entry: int,
) -> pd.Series:
    values: dict[str, float] = {}
    for col in feature_columns:
        if col == "positions_at_entry":
            values[col] = float(positions_at_entry)
        elif col == "entry_score" and _row_get(row, col) is None:
            values[col] = pd.to_numeric(_row_get(row, "ridge_score"), errors="coerce")
        elif isinstance(row, pd.Series):
            values[col] = pd.to_numeric(row.get(col), errors="coerce")
        else:
            values[col] = pd.to_numeric(row.get(col), errors="coerce")
    return pd.Series(values, index=feature_columns, dtype=float)


def _row_get(row: pd.Series | dict[str, Any], key: str) -> Any:
    return row.get(key) if isinstance(row, (pd.Series, dict)) else None


def _empty_cluster_gate(config: ClusterRegimeGateConfig) -> ClusterRegimeGate:
    feature_columns = tuple(config.feature_columns)
    return ClusterRegimeGate(
        config=config,
        feature_columns=feature_columns,
        means=pd.Series(dtype=float, index=feature_columns),
        stds=pd.Series(dtype=float, index=feature_columns),
        centroids=np.empty((0, len(feature_columns))),
        allowed_clusters=frozenset(),
        training_average_return=np.nan,
        cluster_summary=pd.DataFrame(
            columns=["cluster_id", "entries", "mean_return", "median_return", "win_rate", "total_pnl", "allowed"]
        ),
        cluster_profiles=pd.DataFrame(columns=["cluster_id", *feature_columns]),
    )


def _kmeans(values: np.ndarray, n_clusters: int, *, seed: int, max_iter: int = 200) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    centers = values[rng.choice(len(values), size=n_clusters, replace=False)].copy()
    labels = np.zeros(len(values), dtype=int)
    for _ in range(max_iter):
        distances = ((values[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels = distances.argmin(axis=1)
        updated = centers.copy()
        for cluster_id in range(n_clusters):
            mask = labels == cluster_id
            if mask.any():
                updated[cluster_id] = values[mask].mean(axis=0)
        if np.allclose(updated, centers):
            break
        centers = updated
    return labels, centers


def _rank1_pairs(df: pd.DataFrame, *, score_col: str, time_col: str) -> dict[int, str]:
    pairs = {}
    for open_time, group in df.groupby(time_col, observed=True):
        ranked = group.sort_values(score_col, ascending=False)
        if not ranked.empty:
            pairs[int(open_time)] = str(ranked.iloc[0]["pair"])
    return pairs


def _second_largest(values: pd.Series) -> float:
    clean = values.dropna().sort_values(ascending=False)
    return float(clean.iloc[1]) if len(clean) > 1 else np.nan


def _none_if_na(value: object) -> float | None:
    if pd.isna(value):
        return None
    return float(value)

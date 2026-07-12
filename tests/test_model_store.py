import numpy as np
import pandas as pd

from bot.model_store import load_cluster_gate, load_ridge_selection, save_cluster_gate, save_ridge_selection
from bot.strategy.regime import ClusterRegimeGateConfig, train_cluster_regime_gate
from bot.strategy.ridge import RidgeSelection


def test_ridge_selection_roundtrip(tmp_path):
    selection = RidgeSelection(
        model="momentum_only",
        alpha=0.1,
        terms=("z_momentum",),
        beta=np.array([0.2]),
        is_mean_spearman=0.03,
        is_spearman_hit_rate=0.55,
    )

    save_ridge_selection(selection, {"train_months": 3}, tmp_path)
    loaded, metadata = load_ridge_selection(tmp_path)

    assert loaded.model == "momentum_only"
    assert loaded.terms == ("z_momentum",)
    assert loaded.beta.tolist() == [0.2]
    assert metadata["train_months"] == 3


def test_cluster_gate_roundtrip(tmp_path):
    rows = []
    for idx in range(12):
        rows.append(
            {
                "entry_score": float(idx),
                "score_gap_rank1_rank2": 1.0,
                "score_gap_rank1_median": 1.0,
                "score_gap_rank1_rank2_trailing_median": 0.5,
                "pair_return_1h_known": 0.0,
                "universe_breadth_1h": 0.5,
                "universe_return_1h": 0.0,
                "universe_volatility_1h": 0.01,
                "pair_top3_count_3h": 3.0,
                "pair_top3_count_6h": 6.0,
                "pair_top3_count_12h": 12.0,
                "pair_rank1_count_3h": 3.0,
                "pair_rank1_count_6h": 6.0,
                "pair_rank1_count_12h": 12.0,
                "positions_at_entry": 1.0,
                "return_pct": 0.01,
                "pnl": 1.0,
            }
        )
    gate = train_cluster_regime_gate(
        pd.DataFrame(rows),
        ClusterRegimeGateConfig(n_clusters=2, min_cluster_trades=2, random_seed=1),
    )

    save_cluster_gate(gate, {"train_months": 24}, tmp_path)
    loaded, metadata = load_cluster_gate(tmp_path)

    assert loaded.config.n_clusters == 2
    assert loaded.centroids.shape == gate.centroids.shape
    assert metadata["train_months"] == 24

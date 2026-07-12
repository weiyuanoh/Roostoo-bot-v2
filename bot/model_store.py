"""Persist and load trained live model artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bot.config import MODEL_DIR
from bot.strategy.regime import ClusterRegimeGate, ClusterRegimeGateConfig
from bot.strategy.ridge import RidgeSelection

RIDGE_MODEL_FILE = "ridge_selection.json"
CLUSTER_GATE_FILE = "cluster_regime_gate.json"
ARTIFACT_VERSION = 1


def ridge_model_path(model_dir: str | Path = MODEL_DIR) -> Path:
    return Path(model_dir) / RIDGE_MODEL_FILE


def cluster_gate_path(model_dir: str | Path = MODEL_DIR) -> Path:
    return Path(model_dir) / CLUSTER_GATE_FILE


def save_ridge_selection(selection: RidgeSelection, metadata: dict[str, Any], model_dir: str | Path = MODEL_DIR) -> Path:
    path = ridge_model_path(model_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "artifact_version": ARTIFACT_VERSION,
        "kind": "ridge_selection",
        "metadata": metadata,
        "selection": {
            "model": selection.model,
            "alpha": selection.alpha,
            "terms": list(selection.terms),
            "beta": selection.beta.tolist(),
            "is_mean_spearman": selection.is_mean_spearman,
            "is_spearman_hit_rate": selection.is_spearman_hit_rate,
        },
    }
    path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True), encoding="utf-8")
    return path


def load_ridge_selection(model_dir: str | Path = MODEL_DIR) -> tuple[RidgeSelection, dict[str, Any]]:
    path = ridge_model_path(model_dir)
    payload = json.loads(path.read_text(encoding="utf-8"))
    selection = payload["selection"]
    return (
        RidgeSelection(
            model=str(selection["model"]),
            alpha=float(selection["alpha"]),
            terms=tuple(str(term) for term in selection["terms"]),
            beta=np.array(selection["beta"], dtype=float),
            is_mean_spearman=float(selection["is_mean_spearman"]),
            is_spearman_hit_rate=float(selection["is_spearman_hit_rate"]),
        ),
        dict(payload.get("metadata", {})),
    )


def save_cluster_gate(gate: ClusterRegimeGate, metadata: dict[str, Any], model_dir: str | Path = MODEL_DIR) -> Path:
    path = cluster_gate_path(model_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "artifact_version": ARTIFACT_VERSION,
        "kind": "cluster_regime_gate",
        "metadata": metadata,
        "config": {
            "n_clusters": gate.config.n_clusters,
            "min_cluster_trades": gate.config.min_cluster_trades,
            "min_cluster_mean_return": gate.config.min_cluster_mean_return,
            "require_better_than_training_average": gate.config.require_better_than_training_average,
            "lookback_months": gate.config.lookback_months,
            "random_seed": gate.config.random_seed,
            "feature_columns": list(gate.feature_columns),
        },
        "means": gate.means.to_dict(),
        "stds": gate.stds.to_dict(),
        "centroids": gate.centroids.tolist(),
        "allowed_clusters": sorted(gate.allowed_clusters),
        "training_average_return": gate.training_average_return,
        "cluster_summary": gate.cluster_summary.to_dict(orient="records"),
        "cluster_profiles": gate.cluster_profiles.to_dict(orient="records"),
    }
    path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True), encoding="utf-8")
    return path


def load_cluster_gate(model_dir: str | Path = MODEL_DIR) -> tuple[ClusterRegimeGate, dict[str, Any]]:
    path = cluster_gate_path(model_dir)
    payload = json.loads(path.read_text(encoding="utf-8"))
    config_payload = payload["config"]
    config = ClusterRegimeGateConfig(
        n_clusters=int(config_payload["n_clusters"]),
        min_cluster_trades=int(config_payload["min_cluster_trades"]),
        min_cluster_mean_return=float(config_payload["min_cluster_mean_return"]),
        require_better_than_training_average=bool(config_payload["require_better_than_training_average"]),
        lookback_months=int(config_payload["lookback_months"]),
        random_seed=int(config_payload["random_seed"]),
        feature_columns=tuple(str(col) for col in config_payload["feature_columns"]),
    )
    feature_columns = tuple(config.feature_columns)
    gate = ClusterRegimeGate(
        config=config,
        feature_columns=feature_columns,
        means=pd.Series(payload["means"], dtype=float).reindex(feature_columns),
        stds=pd.Series(payload["stds"], dtype=float).reindex(feature_columns),
        centroids=np.array(payload["centroids"], dtype=float),
        allowed_clusters=frozenset(int(value) for value in payload["allowed_clusters"]),
        training_average_return=(
            float(payload["training_average_return"])
            if payload.get("training_average_return") is not None
            else np.nan
        ),
        cluster_summary=pd.DataFrame(payload.get("cluster_summary", [])),
        cluster_profiles=pd.DataFrame(payload.get("cluster_profiles", [])),
    )
    return gate, dict(payload.get("metadata", {}))


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        if np.isnan(value) or np.isinf(value):
            return None
    return value

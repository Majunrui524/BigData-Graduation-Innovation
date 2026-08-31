"""Grouping-oriented baselines centered on train-majority label projection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.cluster import KMeans

from .community_purity import DEFAULT_PURITY_SMOOTHING_ALPHA, DEFAULT_PURITY_THRESHOLD, evaluate_assignment_rows
from .external_baseline_common import (
    DEFAULT_BASELINE_SEED,
    build_dense_matrix,
    infer_numeric_feature_names,
    load_feature_rows,
)
from .readers import write_csv, write_json

DEFAULT_KMEANS_K_VALUES = (32, 64, 128, 256, 512, 898)


def run_kmeans_grouping_baseline(
    sample_root: Path,
    output_root: Path,
    *,
    feature_root: Path | None = None,
    k_values: tuple[int, ...] = DEFAULT_KMEANS_K_VALUES,
    threshold: float = DEFAULT_PURITY_THRESHOLD,
    smoothing_alpha: float = DEFAULT_PURITY_SMOOTHING_ALPHA,
    seed: int = DEFAULT_BASELINE_SEED,
) -> dict[str, Any]:
    """Run K-Means as a grouping baseline and evaluate via train-majority purity."""

    rows = load_feature_rows(sample_root, feature_root=feature_root)
    if not rows:
        raise ValueError("No labeled feature rows available for K-Means grouping baseline")
    feature_names = infer_numeric_feature_names(rows)
    if not feature_names:
        raise ValueError("No numeric feature columns found for K-Means grouping baseline")
    matrix = build_dense_matrix(rows, feature_names)

    candidates: list[dict[str, Any]] = []
    seen_k: set[int] = set()
    for k_value in k_values:
        cluster_count = min(max(int(k_value), 2), len(rows))
        if cluster_count in seen_k:
            continue
        seen_k.add(cluster_count)
        model = KMeans(n_clusters=cluster_count, random_state=seed, n_init=10)
        labels = model.fit_predict(matrix)
        assignments = _build_assignment_rows(rows, labels)
        candidate_root = output_root / f"_candidate_k{cluster_count}"
        manifest = evaluate_assignment_rows(
            sample_root,
            assignments,
            candidate_root,
            threshold=threshold,
            smoothing_alpha=smoothing_alpha,
            encoding_tree_path=None,
            method_key="kmeans",
            method_name="K-Means",
            model_family="grouping",
            graph_source="user_feature_table",
            selected_params={
                "n_clusters": cluster_count,
                "threshold": float(threshold),
                "smoothing_alpha": float(smoothing_alpha),
            },
            source_root=feature_root or (sample_root / "analysis" / "user_features"),
        )
        candidates.append(
            {
                "cluster_count": cluster_count,
                "manifest": manifest,
                "assignments": assignments,
                "centers": model.cluster_centers_.astype(np.float32),
            }
        )

    if not candidates:
        raise ValueError("No K-Means candidates were evaluated")
    best = max(
        candidates,
        key=lambda item: (
            float(item["manifest"]["metrics"].get("valid", {}).get("f1", 0.0)),
            float(item["manifest"]["metrics"].get("valid", {}).get("auc", 0.0)),
            float(item["manifest"]["global_purity"]),
        ),
    )

    output_root.mkdir(parents=True, exist_ok=True)
    write_csv(
        output_root / "community_assignments.csv",
        ["user_id", "community_id", "community_size", "split", "label"],
        best["assignments"],
    )
    centers_path = output_root / "cluster_centers.json"
    write_json(
        centers_path,
        {
            "n_clusters": int(best["cluster_count"]),
            "feature_names": feature_names,
            "centers": [[round(float(value), 8) for value in row] for row in best["centers"]],
        },
    )
    bundle_root = output_root
    manifest = evaluate_assignment_rows(
        sample_root,
        best["assignments"],
        bundle_root,
        threshold=threshold,
        smoothing_alpha=smoothing_alpha,
        encoding_tree_path=None,
        method_key="kmeans",
        method_name="K-Means",
        model_family="grouping",
        graph_source="user_feature_table",
        selected_params={
            "n_clusters": int(best["cluster_count"]),
            "threshold": float(threshold),
            "smoothing_alpha": float(smoothing_alpha),
            "feature_count": len(feature_names),
        },
        source_root=feature_root or (sample_root / "analysis" / "user_features"),
    )
    manifest["files"]["cluster_centers_json"] = str(centers_path)
    write_json(output_root / "community_purity_manifest.json", manifest)
    _cleanup_candidate_dirs(output_root, keep_name="")
    return manifest


def _build_assignment_rows(rows: list[dict[str, Any]], cluster_labels: np.ndarray) -> list[dict[str, Any]]:
    cluster_sizes: dict[int, int] = {}
    for cluster_label in cluster_labels:
        cluster_sizes[int(cluster_label)] = cluster_sizes.get(int(cluster_label), 0) + 1

    assignments: list[dict[str, Any]] = []
    for row, cluster_label in zip(rows, cluster_labels, strict=True):
        cluster_id = int(cluster_label)
        assignments.append(
            {
                "user_id": str(row.get("user_id") or ""),
                "community_id": f"km{cluster_id:04d}",
                "community_size": int(cluster_sizes[cluster_id]),
                "split": str(row.get("split") or ""),
                "label": str(row.get("label") or ""),
            }
        )
    assignments.sort(key=lambda row: row["user_id"])
    return assignments


def _cleanup_candidate_dirs(output_root: Path, *, keep_name: str) -> None:
    for candidate_dir in output_root.glob("_candidate_k*"):
        if candidate_dir.name == keep_name:
            continue
        for child in candidate_dir.iterdir():
            if child.is_file():
                child.unlink()
        candidate_dir.rmdir()

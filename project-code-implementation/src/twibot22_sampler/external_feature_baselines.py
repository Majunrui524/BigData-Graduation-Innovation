"""Classical supervised baselines over user-level feature tables."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .external_baseline_common import (
    DEFAULT_BASELINE_SEED,
    DEFAULT_CLASS_WEIGHT_VALUES,
    DEFAULT_LR_C_VALUES,
    DEFAULT_RF_ESTIMATORS,
    DEFAULT_RF_MAX_DEPTHS,
    DEFAULT_SELECTION_SPLIT,
    build_dense_matrix,
    build_prediction_rows,
    compute_split_metrics,
    infer_numeric_feature_names,
    labels_to_binary,
    load_feature_rows,
    render_baseline_summary,
    select_best_candidate,
    write_baseline_bundle,
)


def run_feature_baselines(
    sample_root: Path,
    output_root: Path,
    *,
    feature_root: Path | None = None,
    lr_c_values: tuple[float, ...] = DEFAULT_LR_C_VALUES,
    class_weight_values: tuple[str | None, ...] = DEFAULT_CLASS_WEIGHT_VALUES,
    rf_estimators: tuple[int, ...] = DEFAULT_RF_ESTIMATORS,
    rf_max_depths: tuple[int | None, ...] = DEFAULT_RF_MAX_DEPTHS,
    seed: int = DEFAULT_BASELINE_SEED,
) -> dict[str, dict[str, Any]]:
    """Run Logistic Regression and Random Forest on 10k user features."""

    rows = load_feature_rows(sample_root, feature_root=feature_root)
    if not rows:
        raise ValueError("No labeled feature rows available for external feature baselines")
    feature_names = infer_numeric_feature_names(rows)
    if not feature_names:
        raise ValueError("No numeric feature columns found in the user feature table")

    matrix = build_dense_matrix(rows, feature_names)
    labels = labels_to_binary(rows)
    index_lists = {"train": [], "valid": [], "test": []}
    for index, row in enumerate(rows):
        split = str(row.get("split") or "")
        if split in index_lists:
            index_lists[split].append(index)
    index_by_split = {
        split: np.asarray(indices, dtype=np.int32)
        for split, indices in index_lists.items()
    }

    output_root.mkdir(parents=True, exist_ok=True)
    manifests = {
        "logreg": _run_logistic_regression(
            sample_root,
            output_root / "logreg",
            rows=rows,
            matrix=matrix,
            labels=labels,
            feature_names=feature_names,
            index_by_split=index_by_split,
            c_values=lr_c_values,
            class_weight_values=class_weight_values,
            seed=seed,
        ),
        "random_forest": _run_random_forest(
            sample_root,
            output_root / "random_forest",
            rows=rows,
            matrix=matrix,
            labels=labels,
            feature_names=feature_names,
            index_by_split=index_by_split,
            estimators=rf_estimators,
            max_depths=rf_max_depths,
            class_weight_values=class_weight_values,
            seed=seed,
        ),
    }
    return manifests


def _run_logistic_regression(
    sample_root: Path,
    output_root: Path,
    *,
    rows: list[dict[str, Any]],
    matrix: np.ndarray,
    labels: np.ndarray,
    feature_names: list[str],
    index_by_split: dict[str, np.ndarray],
    c_values: tuple[float, ...],
    class_weight_values: tuple[str | None, ...],
    seed: int,
) -> dict[str, Any]:
    train_idx = index_by_split["train"]
    valid_idx = index_by_split["valid"]
    candidates = []
    for c_value in c_values:
        for class_weight in class_weight_values:
            pipeline = Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            C=float(c_value),
                            class_weight=class_weight,
                            max_iter=2000,
                            random_state=seed,
                        ),
                    ),
                ]
            )
            pipeline.fit(matrix[train_idx], labels[train_idx])
            valid_scores = pipeline.predict_proba(matrix[valid_idx])[:, 1]
            valid_metrics = compute_split_metrics(labels[valid_idx], valid_scores, threshold=0.5)
            candidates.append(
                {
                    "selected_params": {
                        "C": float(c_value),
                        "class_weight": class_weight or "none",
                    },
                    "pipeline": pipeline,
                    "valid_metrics": valid_metrics,
                }
            )
    best = select_best_candidate(candidates)
    bot_scores = best["pipeline"].predict_proba(matrix)[:, 1]
    metrics = {
        split: compute_split_metrics(labels[index_by_split[split]], bot_scores[index_by_split[split]], threshold=0.5)
        for split in ("train", "valid", "test")
    }
    prediction_rows = build_prediction_rows(rows, bot_scores, threshold=0.5)
    manifest = {
        "sample_root": str(sample_root),
        "output_root": str(output_root),
        "method_key": "logreg",
        "method_name": "Logistic Regression",
        "model_family": "feature_supervised",
        "graph_source": "",
        "selection_split": DEFAULT_SELECTION_SPLIT,
        "selected_params": best["selected_params"],
        "counts": {
            "users": len(rows),
            "train_users": int(len(index_by_split["train"])),
            "valid_users": int(len(index_by_split["valid"])),
            "test_users": int(len(index_by_split["test"])),
            "features": len(feature_names),
        },
        "files": {
            "metrics_json": str(output_root / "metrics.json"),
            "predictions_csv": str(output_root / "predictions.csv"),
            "summary_md": str(output_root / "summary.md"),
        },
    }
    summary = render_baseline_summary(
        method_name=manifest["method_name"],
        selected_params=best["selected_params"],
        metrics=metrics,
        feature_names=feature_names,
    )
    write_baseline_bundle(output_root, manifest=manifest, metrics=metrics, prediction_rows=prediction_rows, summary_markdown=summary)
    return manifest


def _run_random_forest(
    sample_root: Path,
    output_root: Path,
    *,
    rows: list[dict[str, Any]],
    matrix: np.ndarray,
    labels: np.ndarray,
    feature_names: list[str],
    index_by_split: dict[str, np.ndarray],
    estimators: tuple[int, ...],
    max_depths: tuple[int | None, ...],
    class_weight_values: tuple[str | None, ...],
    seed: int,
) -> dict[str, Any]:
    train_idx = index_by_split["train"]
    valid_idx = index_by_split["valid"]
    candidates = []
    for n_estimators in estimators:
        for max_depth in max_depths:
            for class_weight in class_weight_values:
                model = RandomForestClassifier(
                    n_estimators=int(n_estimators),
                    max_depth=max_depth,
                    class_weight=class_weight,
                    random_state=seed,
                    n_jobs=-1,
                )
                model.fit(matrix[train_idx], labels[train_idx])
                valid_scores = model.predict_proba(matrix[valid_idx])[:, 1]
                valid_metrics = compute_split_metrics(labels[valid_idx], valid_scores, threshold=0.5)
                candidates.append(
                    {
                        "selected_params": {
                            "n_estimators": int(n_estimators),
                            "max_depth": max_depth if max_depth is not None else "none",
                            "class_weight": class_weight or "none",
                        },
                        "model": model,
                        "valid_metrics": valid_metrics,
                    }
                )
    best = select_best_candidate(candidates)
    bot_scores = best["model"].predict_proba(matrix)[:, 1]
    metrics = {
        split: compute_split_metrics(labels[index_by_split[split]], bot_scores[index_by_split[split]], threshold=0.5)
        for split in ("train", "valid", "test")
    }
    prediction_rows = build_prediction_rows(rows, bot_scores, threshold=0.5)
    manifest = {
        "sample_root": str(sample_root),
        "output_root": str(output_root),
        "method_key": "random_forest",
        "method_name": "Random Forest",
        "model_family": "feature_supervised",
        "graph_source": "",
        "selection_split": DEFAULT_SELECTION_SPLIT,
        "selected_params": best["selected_params"],
        "counts": {
            "users": len(rows),
            "train_users": int(len(index_by_split["train"])),
            "valid_users": int(len(index_by_split["valid"])),
            "test_users": int(len(index_by_split["test"])),
            "features": len(feature_names),
        },
        "files": {
            "metrics_json": str(output_root / "metrics.json"),
            "predictions_csv": str(output_root / "predictions.csv"),
            "summary_md": str(output_root / "summary.md"),
        },
    }
    summary = render_baseline_summary(
        method_name=manifest["method_name"],
        selected_params=best["selected_params"],
        metrics=metrics,
        feature_names=feature_names,
    )
    write_baseline_bundle(output_root, manifest=manifest, metrics=metrics, prediction_rows=prediction_rows, summary_markdown=summary)
    return manifest

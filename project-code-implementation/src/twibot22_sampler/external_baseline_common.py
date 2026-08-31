"""Shared utilities for external baseline experiments."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from .readers import read_csv_rows, read_jsonl_records, write_csv, write_json

BOT_LABEL = "bot"
HUMAN_LABEL = "human"
DEFAULT_BASELINE_SEED = 42
DEFAULT_SELECTION_SPLIT = "valid"

DEFAULT_LR_C_VALUES = (0.1, 1.0, 10.0)
DEFAULT_CLASS_WEIGHT_VALUES = (None, "balanced")
DEFAULT_RF_ESTIMATORS = (200, 500)
DEFAULT_RF_MAX_DEPTHS = (8, 16, None)
DEFAULT_WALK_DIMENSION = 128
DEFAULT_WALK_LENGTH = 40
DEFAULT_NUM_WALKS = 10
DEFAULT_WALK_WINDOW = 5
DEFAULT_WALK_EPOCHS = 5
DEFAULT_NODE2VEC_P_VALUES = (0.5, 1.0, 2.0)
DEFAULT_NODE2VEC_Q_VALUES = (0.5, 1.0, 2.0)

EXCLUDED_FEATURE_FIELDS = {
    "user_id",
    "username",
    "name",
    "label",
    "split",
    "created_at",
    "description",
    "profile_url",
    "triplet_document",
    "verified_bucket",
}

SUMMARY_METHOD_ORDER = (
    "logreg",
    "random_forest",
    "deepwalk_lr",
    "node2vec_lr",
)


def load_label_split_maps(sample_root: Path) -> tuple[dict[str, str], dict[str, str]]:
    """Load label and split maps from the exported sample."""

    label_map: dict[str, str] = {}
    split_map: dict[str, str] = {}
    for row in read_csv_rows(sample_root / "label.csv"):
        identifier = str(row.get("id") or row.get("user_id") or "")
        label = str(row.get("label") or "")
        if identifier and label in {BOT_LABEL, HUMAN_LABEL}:
            label_map[identifier] = label
    for row in read_csv_rows(sample_root / "split.csv"):
        identifier = str(row.get("id") or row.get("user_id") or "")
        split = str(row.get("split") or "")
        if identifier and split:
            split_map[identifier] = split
    return label_map, split_map


def load_feature_rows(sample_root: Path, feature_root: Path | None = None) -> list[dict[str, Any]]:
    """Load labeled feature rows from the exported user feature table."""

    label_map, split_map = load_label_split_maps(sample_root)
    feature_root = feature_root or (sample_root / "analysis" / "user_features")
    rows: list[dict[str, Any]] = []
    for row in read_jsonl_records(feature_root / "user_feature_table.jsonl"):
        user_id = str(row.get("user_id") or "")
        if not user_id:
            continue
        label = label_map.get(user_id, str(row.get("label") or ""))
        split = split_map.get(user_id, str(row.get("split") or ""))
        if label not in {BOT_LABEL, HUMAN_LABEL} or not split:
            continue
        enriched = dict(row)
        enriched["user_id"] = user_id
        enriched["label"] = label
        enriched["split"] = split
        rows.append(enriched)
    return rows


def infer_numeric_feature_names(rows: Iterable[dict[str, Any]]) -> list[str]:
    """Infer stable numeric features from exported user rows."""

    numeric_names: dict[str, bool] = {}
    for row in rows:
        for key, value in row.items():
            if key in EXCLUDED_FEATURE_FIELDS:
                continue
            if value is None or value == "":
                continue
            is_numeric = isinstance(value, (int, float, bool, np.integer, np.floating, np.bool_))
            numeric_names[key] = numeric_names.get(key, True) and is_numeric
    return sorted(name for name, is_numeric in numeric_names.items() if is_numeric)


def build_dense_matrix(rows: list[dict[str, Any]], feature_names: list[str]) -> np.ndarray:
    """Build a dense numeric feature matrix."""

    matrix = np.zeros((len(rows), len(feature_names)), dtype=np.float32)
    for row_index, row in enumerate(rows):
        for feature_index, feature_name in enumerate(feature_names):
            matrix[row_index, feature_index] = safe_float(row.get(feature_name, 0.0))
    return matrix


def labels_to_binary(rows: list[dict[str, Any]]) -> np.ndarray:
    """Encode bot as 1 and human as 0."""

    return np.asarray([1 if str(row.get("label")) == BOT_LABEL else 0 for row in rows], dtype=np.int32)


def split_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group rows by train/valid/test split."""

    grouped = {"train": [], "valid": [], "test": []}
    for row in rows:
        split = str(row.get("split") or "")
        if split in grouped:
            grouped[split].append(row)
    return grouped


def safe_float(value: Any) -> float:
    """Coerce values to finite floats."""

    if value is None or value == "":
        return 0.0
    if isinstance(value, bool):
        return float(int(value))
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(numeric) or math.isinf(numeric):
        return 0.0
    return numeric


def compute_split_metrics(y_true: np.ndarray, bot_scores: np.ndarray, *, threshold: float = 0.5) -> dict[str, Any]:
    """Compute binary metrics for a single split."""

    predicted = (bot_scores >= threshold).astype(np.int32)
    return {
        "accuracy": round(float(accuracy_score(y_true, predicted)), 8) if len(y_true) else 0.0,
        "precision": round(float(precision_score(y_true, predicted, zero_division=0)), 8) if len(y_true) else 0.0,
        "recall": round(float(recall_score(y_true, predicted, zero_division=0)), 8) if len(y_true) else 0.0,
        "f1": round(float(f1_score(y_true, predicted, zero_division=0)), 8) if len(y_true) else 0.0,
        "auc": round(_safe_auc(y_true, bot_scores), 8) if len(y_true) else 0.0,
        "threshold": float(threshold),
        "labeled_users": int(len(y_true)),
        "tp": int(np.sum((predicted == 1) & (y_true == 1))),
        "fp": int(np.sum((predicted == 1) & (y_true == 0))),
        "fn": int(np.sum((predicted == 0) & (y_true == 1))),
        "tn": int(np.sum((predicted == 0) & (y_true == 0))),
    }


def select_best_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    """Pick the candidate with the best validation metrics."""

    if not candidates:
        raise ValueError("No baseline candidates were evaluated")
    return max(
        candidates,
        key=lambda item: (
            float(item["valid_metrics"]["f1"]),
            float(item["valid_metrics"]["auc"]),
            float(item["valid_metrics"]["precision"]),
        ),
    )


def build_prediction_rows(
    rows: list[dict[str, Any]],
    bot_scores: np.ndarray,
    *,
    threshold: float = 0.5,
) -> list[dict[str, Any]]:
    """Convert scores into prediction CSV rows."""

    output_rows: list[dict[str, Any]] = []
    for row, bot_score in zip(rows, bot_scores, strict=True):
        score = round(float(bot_score), 8)
        output_rows.append(
            {
                "user_id": str(row.get("user_id") or ""),
                "split": str(row.get("split") or ""),
                "label": str(row.get("label") or ""),
                "bot_score": score,
                "predicted_label": BOT_LABEL if score >= threshold else HUMAN_LABEL,
                "threshold": float(threshold),
            }
        )
    return output_rows


def write_baseline_bundle(
    output_root: Path,
    *,
    manifest: dict[str, Any],
    metrics: dict[str, Any],
    prediction_rows: list[dict[str, Any]],
    summary_markdown: str,
) -> None:
    """Write manifest, metrics, predictions, and markdown summary."""

    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root / "manifest.json", manifest)
    write_json(output_root / "metrics.json", metrics)
    write_csv(
        output_root / "predictions.csv",
        ["user_id", "split", "label", "bot_score", "predicted_label", "threshold"],
        prediction_rows,
    )
    (output_root / "summary.md").write_text(summary_markdown, encoding="utf-8")


def render_baseline_summary(
    *,
    method_name: str,
    selected_params: dict[str, Any],
    metrics: dict[str, dict[str, Any]],
    feature_names: list[str] | None = None,
    extra_lines: list[str] | None = None,
) -> str:
    """Render a compact markdown summary."""

    lines = [f"# {method_name}", "", "## Selected Hyperparameters", ""]
    for key in sorted(selected_params):
        lines.append(f"- `{key}`: `{selected_params[key]}`")
    lines.append("")
    lines.append("## Metrics")
    lines.append("")
    for split_name in ("train", "valid", "test"):
        lines.append(f"### {split_name}")
        lines.append("")
        for metric_name in ("accuracy", "precision", "recall", "f1", "auc"):
            lines.append(f"- `{metric_name}`: `{metrics[split_name].get(metric_name, 0.0)}`")
        lines.append("")
    if feature_names:
        lines.append("## Features")
        lines.append("")
        lines.append(f"- feature count: `{len(feature_names)}`")
        lines.append("")
    if extra_lines:
        lines.append("## Notes")
        lines.append("")
        lines.extend(f"- {line}" for line in extra_lines)
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def build_summary_rows(baselines_root: Path, *, expected_methods: Iterable[str] = SUMMARY_METHOD_ORDER) -> list[dict[str, Any]]:
    """Collect rows from completed external baseline outputs."""

    rows: list[dict[str, Any]] = []
    for method_key in expected_methods:
        manifest_path = baselines_root / method_key / "manifest.json"
        metrics_path = baselines_root / method_key / "metrics.json"
        if not manifest_path.exists() or not metrics_path.exists():
            continue
        import json

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        test_metrics = metrics.get("test", {})
        rows.append(
            {
                "method_key": method_key,
                "method": manifest.get("method_name", method_key),
                "model_family": manifest.get("model_family", ""),
                "graph_source": manifest.get("graph_source", ""),
                "selection_split": manifest.get("selection_split", DEFAULT_SELECTION_SPLIT),
                "selected_params": manifest.get("selected_params", {}),
                "test_accuracy": test_metrics.get("accuracy", 0.0),
                "test_precision": test_metrics.get("precision", 0.0),
                "test_recall": test_metrics.get("recall", 0.0),
                "test_f1": test_metrics.get("f1", 0.0),
                "test_auc": test_metrics.get("auc", 0.0),
            }
        )
    return rows


def write_summary_bundle(
    output_root: Path,
    rows: list[dict[str, Any]],
    *,
    sample_root: Path,
    baselines_root: Path,
) -> dict[str, Any]:
    """Write aggregated CSV/Markdown/manifest outputs."""

    output_root.mkdir(parents=True, exist_ok=True)
    write_csv(
        output_root / "external_baseline_results.csv",
        [
            "method",
            "model_family",
            "graph_source",
            "selection_split",
            "selected_params",
            "test_accuracy",
            "test_precision",
            "test_recall",
            "test_f1",
            "test_auc",
        ],
        [
            {
                "method": row["method"],
                "model_family": row["model_family"],
                "graph_source": row["graph_source"],
                "selection_split": row["selection_split"],
                "selected_params": str(row["selected_params"]),
                "test_accuracy": row["test_accuracy"],
                "test_precision": row["test_precision"],
                "test_recall": row["test_recall"],
                "test_f1": row["test_f1"],
                "test_auc": row["test_auc"],
            }
            for row in rows
        ],
    )
    lines = [
        "# External Baseline Results",
        "",
        f"- sample root: `{sample_root}`",
        f"- baselines root: `{baselines_root}`",
        f"- completed methods: `{len(rows)}`",
        "",
        "| Method | Model Family | Graph Source | Test ACC | Test P | Test R | Test F1 | Test AUC |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['method']} | {row['model_family']} | {row['graph_source'] or '-'} | "
            f"{row['test_accuracy']:.4f} | {row['test_precision']:.4f} | {row['test_recall']:.4f} | "
            f"{row['test_f1']:.4f} | {row['test_auc']:.4f} |"
        )
    (output_root / "external_baseline_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = {
        "sample_root": str(sample_root),
        "baselines_root": str(baselines_root),
        "output_root": str(output_root),
        "methods": [row["method"] for row in rows],
        "counts": {"methods": len(rows)},
        "files": {
            "results_csv": str(output_root / "external_baseline_results.csv"),
            "results_md": str(output_root / "external_baseline_results.md"),
        },
    }
    write_json(output_root / "external_baseline_manifest.json", manifest)
    return manifest


def _safe_auc(y_true: np.ndarray, bot_scores: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return 0.0
    return float(roc_auc_score(y_true, bot_scores))

"""Second-stage reranker over community predictions and user features."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .community_evaluation import _compute_metrics_by_split
from .readers import read_csv_rows, read_jsonl_records, write_csv, write_json

DEFAULT_RERANKER_MAX_EPOCHS = 300
DEFAULT_RERANKER_LEARNING_RATE = 0.05
DEFAULT_RERANKER_L2 = 0.001
DEFAULT_RERANKER_THRESHOLD_VALUES = (0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5)
DEFAULT_RERANKER_EARLY_STOPPING = 25

RERANK_FEATURE_NAMES = (
    "community_bot_score",
    "log1p_community_size",
    "log1p_followers_count",
    "log1p_following_count",
    "log1p_tweets_total",
    "log1p_triplet_tweet_count",
    "log1p_post_type_tweet_count",
    "log1p_following_in_degree",
    "log1p_following_out_degree",
    "verified",
    "can_triplet",
    "can_post_type",
    "can_time_feature",
    "can_network_feature",
    "can_full_pipeline",
    "triplet_document_present",
    "triplet_incomplete_flag",
    "post_type_incomplete_flag",
    "tweets_with_created_at_ratio",
    "tweets_with_public_metrics_ratio",
    "tweets_with_references_ratio",
    "tweets_with_external_url_ratio",
    "post_type_coarse_ratio_original",
    "post_type_coarse_ratio_retweet",
    "post_type_coarse_ratio_comment_reply",
    "post_type_coarse_ratio_link_share",
)


def train_community_reranker(
    sample_root: Path,
    best_root: Path,
    output_root: Path,
    *,
    learning_rate: float = DEFAULT_RERANKER_LEARNING_RATE,
    max_epochs: int = DEFAULT_RERANKER_MAX_EPOCHS,
    l2: float = DEFAULT_RERANKER_L2,
    threshold_values: list[float] | tuple[float, ...] = DEFAULT_RERANKER_THRESHOLD_VALUES,
    early_stopping_rounds: int = DEFAULT_RERANKER_EARLY_STOPPING,
) -> dict[str, Any]:
    """Train a lightweight logistic reranker over community predictions and user features."""

    prediction_rows = list(read_csv_rows(best_root / "evaluation" / "community_user_predictions.csv"))
    feature_rows = {
        str(row.get("user_id") or ""): row
        for row in read_jsonl_records(sample_root / "analysis" / "user_features" / "user_feature_table.jsonl")
        if row.get("user_id")
    }
    if not prediction_rows:
        raise ValueError(f"No prediction rows found under {best_root / 'evaluation'}")

    dataset_rows = []
    for row in prediction_rows:
        label = str(row.get("label") or "")
        if label not in {"bot", "human"}:
            continue
        user_id = str(row.get("user_id") or "")
        feature_row = feature_rows.get(user_id, {})
        feature_values = _extract_feature_values(row, feature_row)
        dataset_rows.append(
            {
                "user_id": user_id,
                "split": str(row.get("split") or ""),
                "label": label,
                "community_id": str(row.get("community_id") or ""),
                "community_size": int(float(row.get("community_size") or 0)),
                "community_bot_score": float(row.get("bot_score") or 0.0),
                "baseline_predicted_label": str(row.get("predicted_label") or ""),
                "feature_values": feature_values,
            }
        )

    train_rows = [row for row in dataset_rows if row["split"] == "train"]
    valid_rows = [row for row in dataset_rows if row["split"] == "valid"]
    if not train_rows:
        raise ValueError("No train rows available for reranker training")

    feature_stats = _compute_feature_stats(train_rows)
    for row in dataset_rows:
        row["scaled_features"] = _scale_features(row["feature_values"], feature_stats)

    weights, bias, training_trace = _fit_logistic_regression(
        train_rows,
        valid_rows=valid_rows,
        learning_rate=learning_rate,
        max_epochs=max_epochs,
        l2=l2,
        early_stopping_rounds=early_stopping_rounds,
    )

    for row in dataset_rows:
        probability = _predict_probability(weights, bias, row["scaled_features"])
        row["reranker_bot_score"] = round(probability, 8)

    threshold_grid = sorted(set(float(value) for value in threshold_values))
    selected_threshold, threshold_metrics = _select_threshold(dataset_rows, threshold_grid)
    prediction_output_rows = _build_prediction_output_rows(dataset_rows, selected_threshold)
    metrics = _compute_metrics_by_split(prediction_output_rows)

    baseline_manifest = _load_baseline_manifest(best_root / "evaluation" / "community_eval_manifest.json")
    weight_rows = _build_weight_rows(weights)
    threshold_rows = _build_threshold_rows(threshold_metrics)

    output_root.mkdir(parents=True, exist_ok=True)
    predictions_path = output_root / "reranker_predictions.csv"
    metrics_path = output_root / "reranker_metrics.json"
    weights_path = output_root / "reranker_weights.csv"
    threshold_path = output_root / "threshold_search.csv"
    manifest_path = output_root / "reranker_manifest.json"
    summary_path = output_root / "reranker_summary.md"

    write_csv(
        predictions_path,
        [
            "user_id",
            "split",
            "label",
            "community_id",
            "community_size",
            "community_bot_score",
            "baseline_predicted_label",
            "reranker_bot_score",
            "bot_score",
            "predicted_label",
            "selected_threshold",
        ],
        prediction_output_rows,
    )
    write_csv(
        weights_path,
        ["feature_name", "weight", "direction", "abs_weight"],
        weight_rows,
    )
    write_csv(
        threshold_path,
        ["threshold", "valid_accuracy", "valid_precision", "valid_recall", "valid_f1", "valid_auc", "valid_labeled_users"],
        threshold_rows,
    )
    write_json(metrics_path, metrics)

    manifest = {
        "sample_root": str(sample_root),
        "best_root": str(best_root),
        "output_root": str(output_root),
        "feature_names": list(RERANK_FEATURE_NAMES),
        "training": {
            "learning_rate": float(learning_rate),
            "max_epochs": int(max_epochs),
            "l2": float(l2),
            "early_stopping_rounds": int(early_stopping_rounds),
            "epochs_run": len(training_trace),
            "selected_threshold": float(selected_threshold),
        },
        "counts": {
            "users": len(dataset_rows),
            "train_users": len(train_rows),
            "valid_users": len(valid_rows),
            "test_users": sum(1 for row in dataset_rows if row["split"] == "test"),
        },
        "baseline_metrics": baseline_manifest.get("metrics", {}),
        "reranker_metrics": metrics,
        "files": {
            "predictions_csv": str(predictions_path),
            "metrics_json": str(metrics_path),
            "weights_csv": str(weights_path),
            "threshold_search_csv": str(threshold_path),
            "summary_md": str(summary_path),
        },
    }
    write_json(manifest_path, manifest)
    summary_path.write_text(
        _render_reranker_summary(
            manifest,
            weight_rows,
            weight_rows,
        ),
        encoding="utf-8",
    )
    return manifest


def _extract_feature_values(prediction_row: dict[str, str], feature_row: dict[str, Any]) -> dict[str, float]:
    tweets_total = max(_float_value(feature_row.get("tweets_total")), 1.0)
    return {
        "community_bot_score": float(prediction_row.get("bot_score") or 0.0),
        "log1p_community_size": math.log1p(max(_float_value(prediction_row.get("community_size")), 0.0)),
        "log1p_followers_count": math.log1p(max(_float_value(feature_row.get("followers_count")), 0.0)),
        "log1p_following_count": math.log1p(max(_float_value(feature_row.get("following_count")), 0.0)),
        "log1p_tweets_total": math.log1p(max(_float_value(feature_row.get("tweets_total")), 0.0)),
        "log1p_triplet_tweet_count": math.log1p(max(_float_value(feature_row.get("triplet_tweet_count")), 0.0)),
        "log1p_post_type_tweet_count": math.log1p(max(_float_value(feature_row.get("post_type_tweet_count")), 0.0)),
        "log1p_following_in_degree": math.log1p(max(_float_value(feature_row.get("following_in_degree")), 0.0)),
        "log1p_following_out_degree": math.log1p(max(_float_value(feature_row.get("following_out_degree")), 0.0)),
        "verified": _float_value(feature_row.get("verified")),
        "can_triplet": _float_value(feature_row.get("can_triplet")),
        "can_post_type": _float_value(feature_row.get("can_post_type")),
        "can_time_feature": _float_value(feature_row.get("can_time_feature")),
        "can_network_feature": _float_value(feature_row.get("can_network_feature")),
        "can_full_pipeline": _float_value(feature_row.get("can_full_pipeline")),
        "triplet_document_present": _float_value(feature_row.get("triplet_document_present")),
        "triplet_incomplete_flag": _float_value(feature_row.get("triplet_incomplete_flag")),
        "post_type_incomplete_flag": _float_value(feature_row.get("post_type_incomplete_flag")),
        "tweets_with_created_at_ratio": _safe_ratio(feature_row.get("tweets_with_created_at"), tweets_total),
        "tweets_with_public_metrics_ratio": _safe_ratio(feature_row.get("tweets_with_public_metrics"), tweets_total),
        "tweets_with_references_ratio": _safe_ratio(feature_row.get("tweets_with_references"), tweets_total),
        "tweets_with_external_url_ratio": _safe_ratio(feature_row.get("tweets_with_external_url"), tweets_total),
        "post_type_coarse_ratio_original": _float_value(feature_row.get("post_type_coarse_ratio_original")),
        "post_type_coarse_ratio_retweet": _float_value(feature_row.get("post_type_coarse_ratio_retweet")),
        "post_type_coarse_ratio_comment_reply": _float_value(feature_row.get("post_type_coarse_ratio_comment_reply")),
        "post_type_coarse_ratio_link_share": _float_value(feature_row.get("post_type_coarse_ratio_link_share")),
    }


def _compute_feature_stats(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    for feature_name in RERANK_FEATURE_NAMES:
        values = [float(row["feature_values"].get(feature_name, 0.0)) for row in rows]
        mean = sum(values) / len(values) if values else 0.0
        variance = sum((value - mean) ** 2 for value in values) / len(values) if values else 0.0
        std = math.sqrt(variance)
        stats[feature_name] = {"mean": mean, "std": std}
    return stats


def _scale_features(feature_values: dict[str, float], feature_stats: dict[str, dict[str, float]]) -> list[float]:
    scaled = []
    for feature_name in RERANK_FEATURE_NAMES:
        value = float(feature_values.get(feature_name, 0.0))
        mean = feature_stats[feature_name]["mean"]
        std = feature_stats[feature_name]["std"]
        if std <= 0:
            scaled.append(0.0)
        else:
            scaled.append((value - mean) / std)
    return scaled


def _fit_logistic_regression(
    train_rows: list[dict[str, Any]],
    *,
    valid_rows: list[dict[str, Any]],
    learning_rate: float,
    max_epochs: int,
    l2: float,
    early_stopping_rounds: int,
) -> tuple[list[float], float, list[dict[str, Any]]]:
    weights = [0.0] * len(RERANK_FEATURE_NAMES)
    bias = 0.0
    best_weights = list(weights)
    best_bias = bias
    best_valid_loss = float("inf")
    epochs_without_improvement = 0
    trace: list[dict[str, Any]] = []

    for epoch in range(1, max(int(max_epochs), 1) + 1):
        grad_w = [0.0] * len(weights)
        grad_b = 0.0
        train_loss = 0.0
        for row in train_rows:
            target = 1.0 if row["label"] == "bot" else 0.0
            probability = _predict_probability(weights, bias, row["scaled_features"])
            error = probability - target
            grad_b += error
            for index, value in enumerate(row["scaled_features"]):
                grad_w[index] += error * value
            train_loss += _binary_log_loss(probability, target)
        sample_count = max(len(train_rows), 1)
        for index in range(len(weights)):
            grad_w[index] = grad_w[index] / sample_count + l2 * weights[index]
            weights[index] -= learning_rate * grad_w[index]
        bias -= learning_rate * (grad_b / sample_count)
        train_loss = train_loss / sample_count + (l2 / 2.0) * sum(weight * weight for weight in weights)

        valid_loss = _dataset_log_loss(valid_rows or train_rows, weights, bias)
        trace.append(
            {
                "epoch": epoch,
                "train_loss": round(train_loss, 8),
                "valid_loss": round(valid_loss, 8),
            }
        )
        if valid_loss + 1e-9 < best_valid_loss:
            best_valid_loss = valid_loss
            best_weights = list(weights)
            best_bias = bias
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= max(int(early_stopping_rounds), 1):
                break

    return best_weights, best_bias, trace


def _predict_probability(weights: list[float], bias: float, values: list[float]) -> float:
    logit = bias
    for weight, value in zip(weights, values):
        logit += weight * value
    if logit >= 0:
        exp_value = math.exp(-logit)
        return 1.0 / (1.0 + exp_value)
    exp_value = math.exp(logit)
    return exp_value / (1.0 + exp_value)


def _binary_log_loss(probability: float, target: float) -> float:
    probability = min(max(probability, 1e-8), 1.0 - 1e-8)
    return -(target * math.log(probability) + (1.0 - target) * math.log(1.0 - probability))


def _dataset_log_loss(rows: list[dict[str, Any]], weights: list[float], bias: float) -> float:
    if not rows:
        return 0.0
    return sum(
        _binary_log_loss(
            _predict_probability(weights, bias, row["scaled_features"]),
            1.0 if row["label"] == "bot" else 0.0,
        )
        for row in rows
    ) / len(rows)


def _select_threshold(dataset_rows: list[dict[str, Any]], threshold_values: list[float]) -> tuple[float, list[dict[str, Any]]]:
    validation_rows = [row for row in dataset_rows if row["split"] == "valid"]
    reference_rows = validation_rows if validation_rows else [row for row in dataset_rows if row["split"] == "train"]
    threshold_rows = []
    best_threshold = threshold_values[0]
    best_metrics: dict[str, Any] | None = None
    for threshold in threshold_values:
        rows = []
        for row in reference_rows:
            rows.append(
                {
                    "label": row["label"],
                    "split": row["split"],
                    "bot_score": row["reranker_bot_score"],
                    "predicted_label": "bot" if row["reranker_bot_score"] >= threshold else "human",
                }
            )
        metrics = _compute_metrics_by_split(rows)
        valid_metrics = metrics.get("valid") or metrics.get("train") or {}
        threshold_row = {
            "threshold": round(threshold, 8),
            "valid_accuracy": float(valid_metrics.get("accuracy", 0.0)),
            "valid_precision": float(valid_metrics.get("precision", 0.0)),
            "valid_recall": float(valid_metrics.get("recall", 0.0)),
            "valid_f1": float(valid_metrics.get("f1", 0.0)),
            "valid_auc": float(valid_metrics.get("auc", 0.0)),
            "valid_labeled_users": int(valid_metrics.get("labeled_users", 0)),
        }
        threshold_rows.append(threshold_row)
        if best_metrics is None or (
            threshold_row["valid_f1"],
            threshold_row["valid_auc"],
            -abs(threshold - 0.5),
        ) > (
            best_metrics["valid_f1"],
            best_metrics["valid_auc"],
            -abs(best_threshold - 0.5),
        ):
            best_threshold = threshold
            best_metrics = threshold_row
    return best_threshold, threshold_rows


def _build_prediction_output_rows(dataset_rows: list[dict[str, Any]], selected_threshold: float) -> list[dict[str, Any]]:
    rows = []
    for row in dataset_rows:
        rows.append(
            {
                "user_id": row["user_id"],
                "split": row["split"],
                "label": row["label"],
                "community_id": row["community_id"],
                "community_size": row["community_size"],
                "community_bot_score": round(row["community_bot_score"], 8),
                "baseline_predicted_label": row["baseline_predicted_label"],
                "reranker_bot_score": round(row["reranker_bot_score"], 8),
                "bot_score": round(row["reranker_bot_score"], 8),
                "predicted_label": "bot" if row["reranker_bot_score"] >= selected_threshold else "human",
                "selected_threshold": round(selected_threshold, 8),
            }
        )
    rows.sort(key=lambda row: row["user_id"])
    return rows


def _load_baseline_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _build_weight_rows(weights: list[float]) -> list[dict[str, Any]]:
    rows = []
    for feature_name, weight in zip(RERANK_FEATURE_NAMES, weights):
        rows.append(
            {
                "feature_name": feature_name,
                "weight": round(weight, 8),
                "direction": "bot" if weight > 0 else "human",
                "abs_weight": round(abs(weight), 8),
            }
        )
    rows.sort(key=lambda row: (-row["abs_weight"], row["feature_name"]))
    return rows


def _build_threshold_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: float(row["threshold"]))


def _render_reranker_summary(
    manifest: dict[str, Any],
    top_positive_rows: list[dict[str, Any]],
    top_negative_rows: list[dict[str, Any]],
) -> str:
    baseline_metrics = manifest["baseline_metrics"]
    reranker_metrics = manifest["reranker_metrics"]
    training = manifest["training"]
    lines = [
        "# Community Reranker Summary",
        "",
        "## Training",
        f"- Learning rate: {training['learning_rate']}",
        f"- Max epochs: {training['max_epochs']}",
        f"- L2: {training['l2']}",
        f"- Epochs run: {training['epochs_run']}",
        f"- Selected threshold: {training['selected_threshold']}",
        "",
        "## Test Metrics",
        f"- Baseline F1: {baseline_metrics.get('test', {}).get('f1', 0.0)}",
        f"- Baseline AUC: {baseline_metrics.get('test', {}).get('auc', 0.0)}",
        f"- Reranker F1: {reranker_metrics.get('test', {}).get('f1', 0.0)}",
        f"- Reranker AUC: {reranker_metrics.get('test', {}).get('auc', 0.0)}",
        "",
        "## Top Bot-Leaning Features",
    ]
    for row in [item for item in top_positive_rows if float(item["weight"]) > 0][:10]:
        lines.append(f"- {row['feature_name']}: {row['weight']}")
    lines.extend(["", "## Top Human-Leaning Features"])
    negative_rows = sorted(
        (row for row in top_negative_rows if float(row["weight"]) < 0),
        key=lambda row: float(row["weight"]),
    )[:10]
    for row in negative_rows:
        lines.append(f"- {row['feature_name']}: {row['weight']}")
    return "\n".join(lines) + "\n"


def _safe_ratio(value: Any, denominator: float) -> float:
    numerator = max(_float_value(value), 0.0)
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def _float_value(value: Any) -> float:
    if value in (None, ""):
        return 0.0
    return float(value)

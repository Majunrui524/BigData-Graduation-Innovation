"""Evaluate community assignments against split/label metadata."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .readers import read_csv_rows, write_csv, write_json

DEFAULT_EVAL_THRESHOLD = 0.5
DEFAULT_EVAL_SMOOTHING_ALPHA = 1.0


def evaluate_communities(
    sample_root: Path,
    communities_root: Path,
    output_root: Path,
    *,
    threshold: float = DEFAULT_EVAL_THRESHOLD,
    smoothing_alpha: float = DEFAULT_EVAL_SMOOTHING_ALPHA,
) -> dict[str, Any]:
    """Evaluate detected communities with split-aware label projection."""

    assignments = list(read_csv_rows(communities_root / "community_assignments.csv"))
    if not assignments:
        raise ValueError(f"No community assignments found under {communities_root}")

    split_by_user = {
        str(row.get("id") or row.get("user_id") or ""): str(row.get("split") or "")
        for row in read_csv_rows(sample_root / "split.csv")
        if row.get("id") or row.get("user_id")
    }
    label_by_user = {
        str(row.get("id") or row.get("user_id") or ""): str(row.get("label") or "")
        for row in read_csv_rows(sample_root / "label.csv")
        if row.get("id") or row.get("user_id")
    }

    global_prior = _compute_global_train_prior(assignments, split_by_user, label_by_user)
    community_profiles = _build_community_profiles(
        assignments,
        split_by_user=split_by_user,
        label_by_user=label_by_user,
        threshold=threshold,
        smoothing_alpha=smoothing_alpha,
        global_prior=global_prior,
    )
    prediction_rows = _build_prediction_rows(
        assignments,
        split_by_user=split_by_user,
        label_by_user=label_by_user,
        community_profiles=community_profiles,
    )
    metrics = _compute_metrics_by_split(prediction_rows)
    summary_rows = _build_community_summary_rows(community_profiles)

    output_root.mkdir(parents=True, exist_ok=True)
    predictions_path = output_root / "community_user_predictions.csv"
    community_scores_path = output_root / "community_scores.csv"
    metrics_path = output_root / "community_metrics.json"
    manifest_path = output_root / "community_eval_manifest.json"
    summary_path = output_root / "community_eval_summary.md"

    write_csv(
        predictions_path,
        [
            "user_id",
            "split",
            "label",
            "community_id",
            "community_size",
            "bot_score",
            "predicted_label",
            "score_source",
        ],
        prediction_rows,
    )
    write_csv(
        community_scores_path,
        [
            "community_id",
            "community_size",
            "train_human_count",
            "train_bot_count",
            "train_labeled_count",
            "all_human_count",
            "all_bot_count",
            "all_labeled_count",
            "bot_score",
            "predicted_label",
            "score_source",
        ],
        summary_rows,
    )
    write_json(metrics_path, metrics)

    manifest = {
        "sample_root": str(sample_root),
        "communities_root": str(communities_root),
        "output_root": str(output_root),
        "threshold": float(threshold),
        "smoothing_alpha": float(smoothing_alpha),
        "global_train_bot_prior": round(global_prior, 8),
        "counts": {
            "users": len(assignments),
            "communities": len(community_profiles),
            "prediction_rows": len(prediction_rows),
            "labeled_users": sum(1 for row in prediction_rows if row["label"] in {"bot", "human"}),
            "fallback_communities": sum(
                1 for profile in community_profiles.values() if profile["score_source"] == "global_train_prior"
            ),
        },
        "metrics": metrics,
        "files": {
            "predictions_csv": str(predictions_path),
            "community_scores_csv": str(community_scores_path),
            "metrics_json": str(metrics_path),
            "summary_md": str(summary_path),
        },
    }
    write_json(manifest_path, manifest)
    summary_path.write_text(_render_eval_summary(manifest), encoding="utf-8")
    return manifest


def _compute_global_train_prior(
    assignments: list[dict[str, str]],
    split_by_user: dict[str, str],
    label_by_user: dict[str, str],
) -> float:
    train_bot = 0
    train_human = 0
    for row in assignments:
        user_id = str(row.get("user_id") or "")
        if split_by_user.get(user_id) != "train":
            continue
        label = label_by_user.get(user_id, "")
        if label == "bot":
            train_bot += 1
        elif label == "human":
            train_human += 1
    labeled = train_bot + train_human
    if labeled == 0:
        return 0.5
    return train_bot / labeled


def _build_community_profiles(
    assignments: list[dict[str, str]],
    *,
    split_by_user: dict[str, str],
    label_by_user: dict[str, str],
    threshold: float,
    smoothing_alpha: float,
    global_prior: float,
) -> dict[str, dict[str, Any]]:
    grouped: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    for row in assignments:
        grouped[str(row.get("community_id") or "")].append(row)

    profiles: dict[str, dict[str, Any]] = {}
    for community_id, rows in grouped.items():
        train_human_count = 0
        train_bot_count = 0
        all_human_count = 0
        all_bot_count = 0
        for row in rows:
            user_id = str(row.get("user_id") or "")
            label = label_by_user.get(user_id, "")
            if label == "human":
                all_human_count += 1
            elif label == "bot":
                all_bot_count += 1
            if split_by_user.get(user_id) != "train":
                continue
            if label == "human":
                train_human_count += 1
            elif label == "bot":
                train_bot_count += 1

        train_labeled_count = train_human_count + train_bot_count
        all_labeled_count = all_human_count + all_bot_count
        if train_labeled_count > 0:
            bot_score = (train_bot_count + smoothing_alpha * global_prior) / (train_labeled_count + smoothing_alpha)
            score_source = "train_members"
        else:
            bot_score = global_prior
            score_source = "global_train_prior"
        profiles[community_id] = {
            "community_id": community_id,
            "community_size": len(rows),
            "train_human_count": train_human_count,
            "train_bot_count": train_bot_count,
            "train_labeled_count": train_labeled_count,
            "all_human_count": all_human_count,
            "all_bot_count": all_bot_count,
            "all_labeled_count": all_labeled_count,
            "bot_score": round(bot_score, 8),
            "predicted_label": "bot" if bot_score >= threshold else "human",
            "score_source": score_source,
        }
    return profiles


def _build_prediction_rows(
    assignments: list[dict[str, str]],
    *,
    split_by_user: dict[str, str],
    label_by_user: dict[str, str],
    community_profiles: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for assignment in assignments:
        user_id = str(assignment.get("user_id") or "")
        community_id = str(assignment.get("community_id") or "")
        profile = community_profiles[community_id]
        rows.append(
            {
                "user_id": user_id,
                "split": split_by_user.get(user_id, str(assignment.get("split") or "")),
                "label": label_by_user.get(user_id, str(assignment.get("label") or "")),
                "community_id": community_id,
                "community_size": int(profile["community_size"]),
                "bot_score": float(profile["bot_score"]),
                "predicted_label": str(profile["predicted_label"]),
                "score_source": str(profile["score_source"]),
            }
        )
    rows.sort(key=lambda row: row["user_id"])
    return rows


def _compute_metrics_by_split(prediction_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in prediction_rows:
        if row["label"] not in {"bot", "human"}:
            continue
        buckets["all"].append(row)
        split = str(row.get("split") or "")
        buckets[split].append(row)
    return {split: _compute_binary_metrics(rows) for split, rows in sorted(buckets.items())}


def _compute_binary_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "labeled_users": 0,
            "tp": 0,
            "fp": 0,
            "tn": 0,
            "fn": 0,
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "auc": 0.0,
            "predicted_bot_rate": 0.0,
        }

    tp = fp = tn = fn = 0
    positives = 0
    negatives = 0
    scores: list[tuple[float, int]] = []
    predicted_bot = 0
    for row in rows:
        actual = 1 if row["label"] == "bot" else 0
        predicted = 1 if row["predicted_label"] == "bot" else 0
        score = float(row["bot_score"])
        scores.append((score, actual))
        positives += actual
        negatives += 1 - actual
        predicted_bot += predicted
        if predicted == 1 and actual == 1:
            tp += 1
        elif predicted == 1 and actual == 0:
            fp += 1
        elif predicted == 0 and actual == 0:
            tn += 1
        else:
            fn += 1

    labeled_users = len(rows)
    accuracy = (tp + tn) / labeled_users if labeled_users else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    auc = _roc_auc_score(scores, positives=positives, negatives=negatives)
    return {
        "labeled_users": labeled_users,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": round(accuracy, 8),
        "precision": round(precision, 8),
        "recall": round(recall, 8),
        "f1": round(f1, 8),
        "auc": round(auc, 8),
        "predicted_bot_rate": round(predicted_bot / labeled_users, 8),
    }


def _roc_auc_score(scores: list[tuple[float, int]], *, positives: int, negatives: int) -> float:
    if positives == 0 or negatives == 0:
        return 0.0
    scores = sorted(scores, key=lambda item: item[0])
    rank_sum = 0.0
    index = 0
    total = len(scores)
    while index < total:
        end = index + 1
        while end < total and scores[end][0] == scores[index][0]:
            end += 1
        avg_rank = (index + 1 + end) / 2.0
        positive_count = sum(actual for _, actual in scores[index:end])
        rank_sum += positive_count * avg_rank
        index = end
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def _build_community_summary_rows(community_profiles: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        community_profiles[community_id]
        for community_id in sorted(
            community_profiles,
            key=lambda cid: (-int(community_profiles[cid]["community_size"]), cid),
        )
    ]


def _render_eval_summary(manifest: dict[str, Any]) -> str:
    metrics = manifest["metrics"]
    lines = [
        "# Community Evaluation Summary",
        "",
        "## Overall",
        f"- Users: {manifest['counts']['users']}",
        f"- Communities: {manifest['counts']['communities']}",
        f"- Labeled users: {manifest['counts']['labeled_users']}",
        f"- Fallback communities: {manifest['counts']['fallback_communities']}",
        f"- Threshold: {manifest['threshold']}",
        f"- Smoothing alpha: {manifest['smoothing_alpha']}",
        f"- Global train bot prior: {manifest['global_train_bot_prior']}",
        "",
        "## Metrics",
    ]
    for split in ("train", "valid", "test", "all"):
        if split not in metrics:
            continue
        bucket = metrics[split]
        lines.append(
            f"- {split}: acc={bucket['accuracy']}, precision={bucket['precision']}, "
            f"recall={bucket['recall']}, f1={bucket['f1']}, auc={bucket['auc']}, "
            f"labeled_users={bucket['labeled_users']}"
        )
    return "\n".join(lines) + "\n"

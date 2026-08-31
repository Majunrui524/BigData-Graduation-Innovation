"""Purity-oriented community evaluation and label projection."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .readers import read_csv_rows, write_csv, write_json

DEFAULT_PURITY_THRESHOLD = 0.5
DEFAULT_PURITY_SMOOTHING_ALPHA = 1.0


def evaluate_community_purity(
    sample_root: Path,
    communities_root: Path,
    output_root: Path,
    *,
    threshold: float = DEFAULT_PURITY_THRESHOLD,
    smoothing_alpha: float = DEFAULT_PURITY_SMOOTHING_ALPHA,
) -> dict[str, Any]:
    """Evaluate a detected community partition through purity and majority labels."""

    assignments = list(read_csv_rows(communities_root / "community_assignments.csv"))
    if not assignments:
        raise ValueError(f"No community assignments found under {communities_root}")

    community_manifest_path = communities_root / "community_manifest.json"
    community_manifest = _read_optional_json(community_manifest_path)
    method_name = (
        "Structural Entropy"
        if community_manifest.get("algorithm") == "structural_entropy"
        else "Weighted LPA"
        if community_manifest.get("algorithm") == "weighted_lpa"
        else "Community Purity Evaluation"
    )
    return evaluate_assignment_rows(
        sample_root,
        assignments,
        output_root,
        threshold=threshold,
        smoothing_alpha=smoothing_alpha,
        encoding_tree_path=(communities_root / "encoding_tree.json"),
        method_key=str(community_manifest.get("algorithm") or "community_partition"),
        method_name=method_name,
        model_family="grouping",
        graph_source=str(community_manifest.get("graph_root") or ""),
        selected_params=_build_selected_params(community_manifest, threshold, smoothing_alpha),
        source_root=communities_root,
    )


def evaluate_assignment_rows(
    sample_root: Path,
    assignments: list[dict[str, Any]],
    output_root: Path,
    *,
    threshold: float = DEFAULT_PURITY_THRESHOLD,
    smoothing_alpha: float = DEFAULT_PURITY_SMOOTHING_ALPHA,
    encoding_tree_path: Path | None = None,
    method_key: str = "community_partition",
    method_name: str = "Community Partition",
    model_family: str = "grouping",
    graph_source: str = "",
    selected_params: dict[str, Any] | None = None,
    source_root: Path | None = None,
) -> dict[str, Any]:
    """Evaluate arbitrary user-to-community assignments with purity outputs."""

    if not assignments:
        raise ValueError("No assignment rows were provided for purity evaluation")

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
    normalized_assignments = _normalize_assignments(assignments, split_by_user=split_by_user, label_by_user=label_by_user)

    depth_payload = _load_encoding_depths(encoding_tree_path, normalized_assignments)
    global_prior = _compute_global_train_prior(normalized_assignments, label_by_user=label_by_user)
    community_profiles = _build_community_profiles(
        normalized_assignments,
        label_by_user=label_by_user,
        threshold=threshold,
        smoothing_alpha=smoothing_alpha,
        global_prior=global_prior,
        community_depths=depth_payload["community_depths"],
    )
    prediction_rows = _build_prediction_rows(
        normalized_assignments,
        label_by_user=label_by_user,
        community_profiles=community_profiles,
    )
    metrics = _compute_metrics_by_split(prediction_rows)
    summary_rows = _build_sorted_summary_rows(community_profiles)
    global_purity = _compute_global_purity(summary_rows)

    output_root.mkdir(parents=True, exist_ok=True)
    predictions_path = output_root / "community_purity_user_predictions.csv"
    summary_path = output_root / "community_purity_summary.csv"
    metrics_path = output_root / "community_purity_metrics.json"
    manifest_path = output_root / "community_purity_manifest.json"
    markdown_path = output_root / "community_purity_summary.md"

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
            "label_source",
            "purity",
            "encoding_depth",
        ],
        prediction_rows,
    )
    write_csv(
        summary_path,
        [
            "community_id",
            "community_size",
            "train_human_count",
            "train_bot_count",
            "train_labeled_count",
            "all_human_count",
            "all_bot_count",
            "all_labeled_count",
            "bot_ratio",
            "purity",
            "bot_score",
            "predicted_label_by_train_majority",
            "label_source",
            "encoding_depth",
            "train_count",
            "valid_count",
            "test_count",
        ],
        summary_rows,
    )
    write_json(metrics_path, metrics)

    manifest = {
        "sample_root": str(sample_root),
        "source_root": str(source_root or ""),
        "output_root": str(output_root),
        "method_key": method_key,
        "method_name": method_name,
        "model_family": model_family,
        "graph_source": graph_source,
        "threshold": float(threshold),
        "smoothing_alpha": float(smoothing_alpha),
        "global_train_bot_prior": round(global_prior, 8),
        "global_purity": round(global_purity, 8),
        "selected_params": selected_params or {},
        "counts": {
            "users": len(normalized_assignments),
            "communities": len(community_profiles),
            "prediction_rows": len(prediction_rows),
            "labeled_users": sum(1 for row in prediction_rows if row["label"] in {"bot", "human"}),
            "fallback_communities": sum(
                1 for profile in community_profiles.values() if profile["label_source"] == "global_train_prior"
            ),
            "tie_break_communities": sum(
                1 for profile in community_profiles.values() if profile["label_source"] == "train_tie_break_bot_score"
            ),
            "communities_with_exact_encoding_depth": int(depth_payload["communities_with_exact_depth"]),
        },
        "metrics": metrics,
        "files": {
            "predictions_csv": str(predictions_path),
            "community_purity_csv": str(summary_path),
            "metrics_json": str(metrics_path),
            "summary_md": str(markdown_path),
        },
    }
    write_json(manifest_path, manifest)
    markdown_path.write_text(_render_purity_summary(manifest, summary_rows), encoding="utf-8")
    return manifest


def _build_selected_params(
    community_manifest: dict[str, Any],
    threshold: float,
    smoothing_alpha: float,
) -> dict[str, Any]:
    selected = {
        "threshold": float(threshold),
        "smoothing_alpha": float(smoothing_alpha),
    }
    for key in ("algorithm", "min_community_size", "mutual_support_bonus"):
        if key in community_manifest:
            selected[key] = community_manifest[key]
    return selected


def _normalize_assignments(
    assignments: list[dict[str, Any]],
    *,
    split_by_user: dict[str, str],
    label_by_user: dict[str, str],
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for row in assignments:
        user_id = str(row.get("user_id") or row.get("id") or "")
        community_id = str(row.get("community_id") or "")
        if not user_id or not community_id:
            continue
        normalized.append(
            {
                "user_id": user_id,
                "community_id": community_id,
                "split": split_by_user.get(user_id, str(row.get("split") or "")),
                "label": label_by_user.get(user_id, str(row.get("label") or "")),
            }
        )
    normalized.sort(key=lambda row: row["user_id"])
    return normalized


def _compute_global_train_prior(
    assignments: list[dict[str, Any]],
    *,
    label_by_user: dict[str, str],
) -> float:
    train_bot = 0
    train_human = 0
    for row in assignments:
        if str(row.get("split") or "") != "train":
            continue
        label = label_by_user.get(str(row.get("user_id") or ""), str(row.get("label") or ""))
        if label == "bot":
            train_bot += 1
        elif label == "human":
            train_human += 1
    labeled = train_bot + train_human
    if labeled == 0:
        return 0.5
    return train_bot / labeled


def _build_community_profiles(
    assignments: list[dict[str, Any]],
    *,
    label_by_user: dict[str, str],
    threshold: float,
    smoothing_alpha: float,
    global_prior: float,
    community_depths: dict[str, float],
) -> dict[str, dict[str, Any]]:
    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in assignments:
        grouped[str(row["community_id"])].append(row)

    profiles: dict[str, dict[str, Any]] = {}
    for community_id, rows in grouped.items():
        train_human_count = 0
        train_bot_count = 0
        all_human_count = 0
        all_bot_count = 0
        split_counter: Counter[str] = Counter()
        for row in rows:
            user_id = str(row.get("user_id") or "")
            label = label_by_user.get(user_id, str(row.get("label") or ""))
            split = str(row.get("split") or "")
            if split:
                split_counter[split] += 1
            if label == "human":
                all_human_count += 1
            elif label == "bot":
                all_bot_count += 1
            if split != "train":
                continue
            if label == "human":
                train_human_count += 1
            elif label == "bot":
                train_bot_count += 1

        community_size = len(rows)
        train_labeled_count = train_human_count + train_bot_count
        all_labeled_count = all_human_count + all_bot_count
        if train_labeled_count > 0:
            bot_score = (train_bot_count + smoothing_alpha * global_prior) / (train_labeled_count + smoothing_alpha)
            if train_bot_count > train_human_count:
                predicted_label = "bot"
                label_source = "train_majority"
            elif train_human_count > train_bot_count:
                predicted_label = "human"
                label_source = "train_majority"
            else:
                predicted_label = "bot" if bot_score >= threshold else "human"
                label_source = "train_tie_break_bot_score"
        else:
            bot_score = global_prior
            predicted_label = "bot" if bot_score >= threshold else "human"
            label_source = "global_train_prior"

        purity = max(all_human_count, all_bot_count) / community_size if community_size else 0.0
        bot_ratio = all_bot_count / all_labeled_count if all_labeled_count else 0.0
        profiles[community_id] = {
            "community_id": community_id,
            "community_size": community_size,
            "train_human_count": train_human_count,
            "train_bot_count": train_bot_count,
            "train_labeled_count": train_labeled_count,
            "all_human_count": all_human_count,
            "all_bot_count": all_bot_count,
            "all_labeled_count": all_labeled_count,
            "bot_ratio": round(bot_ratio, 8),
            "purity": round(purity, 8),
            "bot_score": round(bot_score, 8),
            "predicted_label_by_train_majority": predicted_label,
            "label_source": label_source,
            "encoding_depth": round(float(community_depths.get(community_id, 0.0)), 8),
            "train_count": int(split_counter.get("train", 0)),
            "valid_count": int(split_counter.get("valid", 0)),
            "test_count": int(split_counter.get("test", 0)),
        }
    return profiles


def _build_prediction_rows(
    assignments: list[dict[str, Any]],
    *,
    label_by_user: dict[str, str],
    community_profiles: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for assignment in assignments:
        community_id = str(assignment["community_id"])
        profile = community_profiles[community_id]
        user_id = str(assignment["user_id"])
        rows.append(
            {
                "user_id": user_id,
                "split": str(assignment.get("split") or ""),
                "label": label_by_user.get(user_id, str(assignment.get("label") or "")),
                "community_id": community_id,
                "community_size": int(profile["community_size"]),
                "bot_score": float(profile["bot_score"]),
                "predicted_label": str(profile["predicted_label_by_train_majority"]),
                "label_source": str(profile["label_source"]),
                "purity": float(profile["purity"]),
                "encoding_depth": float(profile["encoding_depth"]),
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
    ranked = sorted(scores, key=lambda item: item[0])
    rank_sum = 0.0
    index = 0
    total = len(ranked)
    while index < total:
        end = index + 1
        while end < total and ranked[end][0] == ranked[index][0]:
            end += 1
        avg_rank = (index + 1 + end) / 2.0
        positive_count = sum(actual for _, actual in ranked[index:end])
        rank_sum += positive_count * avg_rank
        index = end
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def _build_sorted_summary_rows(community_profiles: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        community_profiles[community_id]
        for community_id in sorted(
            community_profiles,
            key=lambda cid: (-int(community_profiles[cid]["community_size"]), cid),
        )
    ]


def _compute_global_purity(summary_rows: list[dict[str, Any]]) -> float:
    total_users = sum(int(row["community_size"]) for row in summary_rows)
    if total_users == 0:
        return 0.0
    weighted_sum = sum(float(row["community_size"]) * float(row["purity"]) for row in summary_rows)
    return weighted_sum / total_users


def _load_encoding_depths(
    encoding_tree_path: Path | None,
    assignments: list[dict[str, Any]],
) -> dict[str, Any]:
    if encoding_tree_path is None or not encoding_tree_path.exists():
        return {"community_depths": {}, "communities_with_exact_depth": 0}

    payload = _read_optional_json(encoding_tree_path)
    nodes = payload.get("nodes") if isinstance(payload, dict) else {}
    roots = payload.get("roots") if isinstance(payload, dict) else []
    if not isinstance(nodes, dict) or not isinstance(roots, list):
        return {"community_depths": {}, "communities_with_exact_depth": 0}

    user_to_community = {str(row["user_id"]): str(row["community_id"]) for row in assignments}
    leaf_depths: dict[str, int] = {}
    exact_depths: dict[str, list[int]] = defaultdict(list)

    def _visit(node_id: str, depth: int) -> tuple[str | None, list[str]]:
        payload = nodes.get(node_id, {})
        node_type = str(payload.get("type") or "")
        if node_type == "leaf":
            user_id = str(payload.get("user_id") or "")
            if user_id:
                leaf_depths[user_id] = depth
                return user_to_community.get(user_id), [user_id]
            return None, []
        children = payload.get("children") if isinstance(payload, dict) else []
        if not isinstance(children, list) or not children:
            return None, []

        child_labels: set[str] = set()
        descendants: list[str] = []
        for child_id in children:
            child_label, child_descendants = _visit(str(child_id), depth + 1)
            if child_label:
                child_labels.add(child_label)
            descendants.extend(child_descendants)
        if len(child_labels) == 1 and descendants:
            community_id = next(iter(child_labels))
            descendant_labels = {user_to_community.get(user_id) for user_id in descendants}
            if descendant_labels == {community_id}:
                exact_depths[community_id].append(depth)
                return community_id, descendants
        return None, descendants

    for root_id in roots:
        _visit(str(root_id), 1)

    members_by_community: defaultdict[str, list[str]] = defaultdict(list)
    for row in assignments:
        members_by_community[str(row["community_id"])].append(str(row["user_id"]))

    community_depths: dict[str, float] = {}
    exact_count = 0
    for community_id, members in members_by_community.items():
        member_depths = [leaf_depths[user_id] for user_id in members if user_id in leaf_depths]
        if exact_depths.get(community_id):
            exact_count += 1
        community_depths[community_id] = (
            sum(member_depths) / len(member_depths)
            if member_depths
            else float(min(exact_depths.get(community_id, [0])))
        )
    return {
        "community_depths": community_depths,
        "communities_with_exact_depth": exact_count,
    }


def _read_optional_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


def _render_purity_summary(manifest: dict[str, Any], summary_rows: list[dict[str, Any]]) -> str:
    metrics = manifest["metrics"]
    lines = [
        f"# {manifest['method_name']} Purity Summary",
        "",
        "## Overall",
        f"- Users: {manifest['counts']['users']}",
        f"- Communities: {manifest['counts']['communities']}",
        f"- Global purity: {manifest['global_purity']}",
        f"- Global train bot prior: {manifest['global_train_bot_prior']}",
        f"- Threshold: {manifest['threshold']}",
        f"- Smoothing alpha: {manifest['smoothing_alpha']}",
        f"- Fallback communities: {manifest['counts']['fallback_communities']}",
        f"- Exact encoding-depth communities: {manifest['counts']['communities_with_exact_encoding_depth']}",
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
    lines.extend(["", "## Representative Communities"])
    for row in summary_rows[:10]:
        lines.append(
            f"- {row['community_id']}: size={row['community_size']}, purity={row['purity']}, "
            f"bot_ratio={row['bot_ratio']}, predicted={row['predicted_label_by_train_majority']}, "
            f"encoding_depth={row['encoding_depth']}"
        )
    return "\n".join(lines) + "\n"

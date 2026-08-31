"""Error analysis for finalized community predictions."""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .readers import read_csv_rows, read_jsonl_records, write_csv, write_json

DEFAULT_ERROR_ANALYSIS_SPLIT = "test"
DEFAULT_ERROR_ANALYSIS_TOP_K = 100


def analyze_community_errors(
    sample_root: Path,
    best_root: Path,
    output_root: Path,
    *,
    focus_split: str = DEFAULT_ERROR_ANALYSIS_SPLIT,
    top_k: int = DEFAULT_ERROR_ANALYSIS_TOP_K,
) -> dict[str, Any]:
    """Analyze misclassifications and community purity from finalized predictions."""

    prediction_rows = list(read_csv_rows(best_root / "evaluation" / "community_user_predictions.csv"))
    community_score_rows = list(read_csv_rows(best_root / "evaluation" / "community_scores.csv"))
    feature_rows = {
        str(row.get("user_id") or ""): row
        for row in read_jsonl_records(sample_root / "analysis" / "user_features" / "user_feature_table.jsonl")
        if row.get("user_id")
    }
    if not prediction_rows:
        raise ValueError(f"No prediction rows found under {best_root / 'evaluation'}")

    threshold = _load_threshold(best_root / "best_run_manifest.json")
    enriched_rows = [
        _enrich_prediction_row(row, feature_rows.get(str(row.get("user_id") or ""), {}), threshold)
        for row in prediction_rows
    ]

    false_positive_rows = [row for row in enriched_rows if row["error_type"] == "false_positive"]
    false_negative_rows = [row for row in enriched_rows if row["error_type"] == "false_negative"]
    focus_rows = [row for row in enriched_rows if row["split"] == focus_split and row["label"] in {"bot", "human"}]
    focus_misclassified = [row for row in focus_rows if row["error_type"] != ""]

    focus_false_positive_rows = [row for row in focus_misclassified if row["error_type"] == "false_positive"]
    focus_false_negative_rows = [row for row in focus_misclassified if row["error_type"] == "false_negative"]

    false_positive_rows.sort(key=lambda row: (-row["score_margin_abs"], -row["bot_score"], row["user_id"]))
    false_negative_rows.sort(key=lambda row: (-row["score_margin_abs"], row["bot_score"], row["user_id"]))
    focus_false_positive_rows.sort(key=lambda row: (-row["score_margin_abs"], -row["bot_score"], row["user_id"]))
    focus_false_negative_rows.sort(key=lambda row: (-row["score_margin_abs"], row["bot_score"], row["user_id"]))

    uncertain_rows = [
        row
        for row in focus_rows
        if row["label"] in {"bot", "human"}
    ]
    uncertain_rows.sort(key=lambda row: (row["score_margin_abs"], row["user_id"]))

    community_summary_rows = _build_community_error_rows(
        community_score_rows,
        enriched_rows,
        focus_split=focus_split,
    )
    community_summary_rows.sort(
        key=lambda row: (-row["focus_error_count"], -row["focus_error_rate"], -row["community_size"], row["community_id"])
    )

    output_root.mkdir(parents=True, exist_ok=True)
    fp_path = output_root / "false_positives.csv"
    fn_path = output_root / "false_negatives.csv"
    focus_fp_path = output_root / f"{focus_split}_false_positives.csv"
    focus_fn_path = output_root / f"{focus_split}_false_negatives.csv"
    uncertain_path = output_root / f"{focus_split}_uncertain_predictions.csv"
    community_path = output_root / "community_error_summary.csv"
    manifest_path = output_root / "error_analysis_manifest.json"
    summary_path = output_root / "error_analysis_summary.md"

    _write_analysis_rows(fp_path, false_positive_rows[:top_k])
    _write_analysis_rows(fn_path, false_negative_rows[:top_k])
    _write_analysis_rows(focus_fp_path, focus_false_positive_rows[:top_k])
    _write_analysis_rows(focus_fn_path, focus_false_negative_rows[:top_k])
    _write_analysis_rows(uncertain_path, uncertain_rows[:top_k])
    write_csv(community_path, _community_error_fieldnames(), community_summary_rows[: max(top_k, 50)])

    manifest = {
        "sample_root": str(sample_root),
        "best_root": str(best_root),
        "output_root": str(output_root),
        "focus_split": focus_split,
        "threshold": threshold,
        "top_k": int(top_k),
        "counts": {
            "users": len(enriched_rows),
            "focus_users": len(focus_rows),
            "false_positives_all": len(false_positive_rows),
            "false_negatives_all": len(false_negative_rows),
            "false_positives_focus": len(focus_false_positive_rows),
            "false_negatives_focus": len(focus_false_negative_rows),
            "communities": len(community_summary_rows),
        },
        "feature_signals": _summarize_feature_signals(
            focus_false_positive_rows,
            focus_false_negative_rows,
        ),
        "files": {
            "false_positives_csv": str(fp_path),
            "false_negatives_csv": str(fn_path),
            f"{focus_split}_false_positives_csv": str(focus_fp_path),
            f"{focus_split}_false_negatives_csv": str(focus_fn_path),
            f"{focus_split}_uncertain_predictions_csv": str(uncertain_path),
            "community_error_summary_csv": str(community_path),
            "summary_md": str(summary_path),
        },
    }
    write_json(manifest_path, manifest)
    summary_path.write_text(
        _render_error_analysis_summary(
            manifest,
            focus_false_positive_rows[:10],
            focus_false_negative_rows[:10],
            community_summary_rows[:10],
        ),
        encoding="utf-8",
    )
    return manifest


def _load_threshold(path: Path) -> float:
    manifest = read_json_rowsafe(path)
    return float(manifest.get("selected_run", {}).get("threshold", 0.5))


def read_json_rowsafe(path: Path) -> dict[str, Any]:
    import json

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _enrich_prediction_row(
    row: dict[str, str],
    feature_row: dict[str, Any],
    threshold: float,
) -> dict[str, Any]:
    user_id = str(row.get("user_id") or "")
    split = str(row.get("split") or "")
    label = str(row.get("label") or "")
    predicted_label = str(row.get("predicted_label") or "")
    bot_score = float(row.get("bot_score") or 0.0)
    if label == "human" and predicted_label == "bot":
        error_type = "false_positive"
    elif label == "bot" and predicted_label == "human":
        error_type = "false_negative"
    else:
        error_type = ""
    return {
        "user_id": user_id,
        "split": split,
        "label": label,
        "predicted_label": predicted_label,
        "error_type": error_type,
        "community_id": str(row.get("community_id") or ""),
        "community_size": int(float(row.get("community_size") or 0)),
        "bot_score": round(bot_score, 8),
        "score_margin": round(bot_score - threshold, 8),
        "score_margin_abs": round(abs(bot_score - threshold), 8),
        "score_source": str(row.get("score_source") or ""),
        "username": str(feature_row.get("username") or ""),
        "name": str(feature_row.get("name") or ""),
        "description_excerpt": _excerpt(str(feature_row.get("description") or ""), 160),
        "followers_count": int(_float_value(feature_row.get("followers_count"))),
        "following_count": int(_float_value(feature_row.get("following_count"))),
        "tweets_total": int(_float_value(feature_row.get("tweets_total"))),
        "verified": int(_float_value(feature_row.get("verified"))),
        "can_triplet": int(_float_value(feature_row.get("can_triplet"))),
        "can_post_type": int(_float_value(feature_row.get("can_post_type"))),
        "can_time_feature": int(_float_value(feature_row.get("can_time_feature"))),
        "can_network_feature": int(_float_value(feature_row.get("can_network_feature"))),
        "can_full_pipeline": int(_float_value(feature_row.get("can_full_pipeline"))),
        "triplet_document_present": int(_float_value(feature_row.get("triplet_document_present"))),
        "triplet_tweet_count": int(_float_value(feature_row.get("triplet_tweet_count"))),
        "post_type_tweet_count": int(_float_value(feature_row.get("post_type_tweet_count"))),
        "triplet_incomplete_flag": int(_float_value(feature_row.get("triplet_incomplete_flag"))),
        "post_type_incomplete_flag": int(_float_value(feature_row.get("post_type_incomplete_flag"))),
    }


def _build_community_error_rows(
    community_score_rows: list[dict[str, str]],
    enriched_rows: list[dict[str, Any]],
    *,
    focus_split: str,
) -> list[dict[str, Any]]:
    by_community_all: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    by_community_focus: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in enriched_rows:
        community_id = str(row["community_id"])
        by_community_all[community_id].append(row)
        if row["split"] == focus_split:
            by_community_focus[community_id].append(row)

    rows = []
    for row in community_score_rows:
        community_id = str(row.get("community_id") or "")
        all_rows = by_community_all.get(community_id, [])
        focus_rows = by_community_focus.get(community_id, [])
        focus_error_count = sum(1 for item in focus_rows if item["error_type"])
        focus_size = len([item for item in focus_rows if item["label"] in {"bot", "human"}])
        rows.append(
            {
                "community_id": community_id,
                "community_size": int(float(row.get("community_size") or 0)),
                "bot_score": float(row.get("bot_score") or 0.0),
                "predicted_label": str(row.get("predicted_label") or ""),
                "all_human_count": int(float(row.get("all_human_count") or 0)),
                "all_bot_count": int(float(row.get("all_bot_count") or 0)),
                "all_labeled_count": int(float(row.get("all_labeled_count") or 0)),
                "focus_error_count": focus_error_count,
                "focus_size": focus_size,
                "focus_error_rate": round(focus_error_count / focus_size, 8) if focus_size else 0.0,
                "focus_false_positive_count": sum(1 for item in focus_rows if item["error_type"] == "false_positive"),
                "focus_false_negative_count": sum(1 for item in focus_rows if item["error_type"] == "false_negative"),
                "all_error_count": sum(1 for item in all_rows if item["error_type"]),
            }
        )
    return rows


def _summarize_feature_signals(
    false_positive_rows: list[dict[str, Any]],
    false_negative_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "false_positive": _aggregate_feature_rows(false_positive_rows),
        "false_negative": _aggregate_feature_rows(false_negative_rows),
    }


def _aggregate_feature_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "count": 0,
            "avg_bot_score": 0.0,
            "avg_followers_count": 0.0,
            "avg_tweets_total": 0.0,
            "full_pipeline_rate": 0.0,
            "triplet_available_rate": 0.0,
        }
    count = len(rows)
    return {
        "count": count,
        "avg_bot_score": round(sum(float(row["bot_score"]) for row in rows) / count, 8),
        "avg_followers_count": round(sum(int(row["followers_count"]) for row in rows) / count, 4),
        "avg_tweets_total": round(sum(int(row["tweets_total"]) for row in rows) / count, 4),
        "full_pipeline_rate": round(sum(int(row["can_full_pipeline"]) for row in rows) / count, 8),
        "triplet_available_rate": round(sum(int(row["can_triplet"]) for row in rows) / count, 8),
    }


def _write_analysis_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    write_csv(path, _analysis_fieldnames(), rows)


def _analysis_fieldnames() -> list[str]:
    return [
        "user_id",
        "split",
        "label",
        "predicted_label",
        "error_type",
        "community_id",
        "community_size",
        "bot_score",
        "score_margin",
        "score_margin_abs",
        "score_source",
        "username",
        "name",
        "description_excerpt",
        "followers_count",
        "following_count",
        "tweets_total",
        "verified",
        "can_triplet",
        "can_post_type",
        "can_time_feature",
        "can_network_feature",
        "can_full_pipeline",
        "triplet_document_present",
        "triplet_tweet_count",
        "post_type_tweet_count",
        "triplet_incomplete_flag",
        "post_type_incomplete_flag",
    ]


def _community_error_fieldnames() -> list[str]:
    return [
        "community_id",
        "community_size",
        "bot_score",
        "predicted_label",
        "all_human_count",
        "all_bot_count",
        "all_labeled_count",
        "focus_error_count",
        "focus_size",
        "focus_error_rate",
        "focus_false_positive_count",
        "focus_false_negative_count",
        "all_error_count",
    ]


def _render_error_analysis_summary(
    manifest: dict[str, Any],
    top_fp_rows: list[dict[str, Any]],
    top_fn_rows: list[dict[str, Any]],
    top_community_rows: list[dict[str, Any]],
) -> str:
    counts = manifest["counts"]
    lines = [
        "# Community Error Analysis Summary",
        "",
        "## Overall",
        f"- Focus split: {manifest['focus_split']}",
        f"- Threshold: {manifest['threshold']}",
        f"- Users: {counts['users']}",
        f"- Focus users: {counts['focus_users']}",
        f"- Focus false positives: {counts['false_positives_focus']}",
        f"- Focus false negatives: {counts['false_negatives_focus']}",
        "",
        "## Top False Positives",
    ]
    for row in top_fp_rows:
        lines.append(
            f"- {row['user_id']} ({row['community_id']}): score={row['bot_score']}, "
            f"followers={row['followers_count']}, tweets={row['tweets_total']}"
        )
    lines.extend(["", "## Top False Negatives"])
    for row in top_fn_rows:
        lines.append(
            f"- {row['user_id']} ({row['community_id']}): score={row['bot_score']}, "
            f"followers={row['followers_count']}, tweets={row['tweets_total']}"
        )
    lines.extend(["", "## Most Error-Prone Communities"])
    for row in top_community_rows:
        lines.append(
            f"- {row['community_id']}: size={row['community_size']}, bot_score={row['bot_score']}, "
            f"focus_error_rate={row['focus_error_rate']}, focus_error_count={row['focus_error_count']}"
        )
    return "\n".join(lines) + "\n"


def _excerpt(value: str, limit: int) -> str:
    text = " ".join(value.split())
    if len(text) <= limit:
        return text
    return text[: max(limit - 3, 0)] + "..."


def _float_value(value: Any) -> float:
    if value in (None, ""):
        return 0.0
    return float(value)

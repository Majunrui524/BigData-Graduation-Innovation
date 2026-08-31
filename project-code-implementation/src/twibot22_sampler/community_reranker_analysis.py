"""Compare baseline community predictions against reranker predictions."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .readers import read_csv_rows, read_jsonl_records, write_csv, write_json

DEFAULT_RERANKER_ANALYSIS_SPLIT = "test"
DEFAULT_RERANKER_ANALYSIS_TOP_K = 100


def analyze_community_reranker(
    sample_root: Path,
    best_root: Path,
    reranker_root: Path,
    output_root: Path,
    *,
    focus_split: str = DEFAULT_RERANKER_ANALYSIS_SPLIT,
    top_k: int = DEFAULT_RERANKER_ANALYSIS_TOP_K,
) -> dict[str, Any]:
    """Analyze how the reranker changes baseline community predictions."""

    baseline_rows = {
        str(row.get("user_id") or ""): row
        for row in read_csv_rows(best_root / "evaluation" / "community_user_predictions.csv")
        if row.get("user_id")
    }
    reranker_rows = {
        str(row.get("user_id") or ""): row
        for row in read_csv_rows(reranker_root / "reranker_predictions.csv")
        if row.get("user_id")
    }
    feature_rows = {
        str(row.get("user_id") or ""): row
        for row in read_jsonl_records(sample_root / "analysis" / "user_features" / "user_feature_table.jsonl")
        if row.get("user_id")
    }
    if not baseline_rows or not reranker_rows:
        raise ValueError("Expected both baseline and reranker prediction files to exist")

    compared_rows = []
    for user_id in sorted(set(baseline_rows) & set(reranker_rows)):
        base = baseline_rows[user_id]
        rerank = reranker_rows[user_id]
        feature_row = feature_rows.get(user_id, {})
        compared_rows.append(_build_compared_row(user_id, base, rerank, feature_row))

    focus_rows = [row for row in compared_rows if row["split"] == focus_split and row["label"] in {"bot", "human"}]
    changed_rows = [row for row in focus_rows if row["baseline_predicted_label"] != row["reranker_predicted_label"]]

    fixed_rows = [row for row in changed_rows if row["baseline_correct"] == 0 and row["reranker_correct"] == 1]
    regressed_rows = [row for row in changed_rows if row["baseline_correct"] == 1 and row["reranker_correct"] == 0]
    unchanged_wrong_rows = [row for row in focus_rows if row["baseline_correct"] == 0 and row["reranker_correct"] == 0]

    fixed_rows.sort(key=lambda row: (-row["reranker_confidence_margin"], row["user_id"]))
    regressed_rows.sort(key=lambda row: (-row["reranker_confidence_margin"], row["user_id"]))
    changed_rows.sort(key=lambda row: (-row["score_delta_abs"], row["user_id"]))
    unchanged_wrong_rows.sort(key=lambda row: (-row["baseline_confidence_margin"], row["user_id"]))

    community_rows = _build_community_change_rows(focus_rows)
    community_rows.sort(
        key=lambda row: (
            -row["net_gain"],
            -row["fixed_count"],
            row["community_id"],
        )
    )

    output_root.mkdir(parents=True, exist_ok=True)
    fixed_path = output_root / f"{focus_split}_fixed_cases.csv"
    regressed_path = output_root / f"{focus_split}_regressed_cases.csv"
    changed_path = output_root / f"{focus_split}_changed_predictions.csv"
    stubborn_path = output_root / f"{focus_split}_unchanged_errors.csv"
    community_path = output_root / "community_change_summary.csv"
    manifest_path = output_root / "reranker_comparison_manifest.json"
    summary_path = output_root / "reranker_comparison_summary.md"

    write_csv(fixed_path, _comparison_fieldnames(), fixed_rows[:top_k])
    write_csv(regressed_path, _comparison_fieldnames(), regressed_rows[:top_k])
    write_csv(changed_path, _comparison_fieldnames(), changed_rows[:top_k])
    write_csv(stubborn_path, _comparison_fieldnames(), unchanged_wrong_rows[:top_k])
    write_csv(community_path, _community_change_fieldnames(), community_rows[: max(top_k, 50)])

    baseline_confusion = _compute_confusion_summary(focus_rows, prefix="baseline")
    reranker_confusion = _compute_confusion_summary(focus_rows, prefix="reranker")
    manifest = {
        "sample_root": str(sample_root),
        "best_root": str(best_root),
        "reranker_root": str(reranker_root),
        "output_root": str(output_root),
        "focus_split": focus_split,
        "top_k": int(top_k),
        "counts": {
            "focus_users": len(focus_rows),
            "changed_predictions": len(changed_rows),
            "fixed_cases": len(fixed_rows),
            "regressed_cases": len(regressed_rows),
            "unchanged_errors": len(unchanged_wrong_rows),
            "communities": len(community_rows),
        },
        "baseline_confusion": baseline_confusion,
        "reranker_confusion": reranker_confusion,
        "files": {
            "fixed_cases_csv": str(fixed_path),
            "regressed_cases_csv": str(regressed_path),
            "changed_predictions_csv": str(changed_path),
            "unchanged_errors_csv": str(stubborn_path),
            "community_change_summary_csv": str(community_path),
            "summary_md": str(summary_path),
        },
    }
    write_json(manifest_path, manifest)
    summary_path.write_text(
        _render_comparison_summary(
            manifest,
            fixed_rows[:10],
            regressed_rows[:10],
            community_rows[:10],
        ),
        encoding="utf-8",
    )
    return manifest


def _build_compared_row(
    user_id: str,
    base: dict[str, str],
    rerank: dict[str, str],
    feature_row: dict[str, Any],
) -> dict[str, Any]:
    label = str(rerank.get("label") or base.get("label") or "")
    baseline_predicted_label = str(base.get("predicted_label") or "")
    reranker_predicted_label = str(rerank.get("predicted_label") or "")
    baseline_score = float(base.get("bot_score") or base.get("community_bot_score") or 0.0)
    reranker_score = float(rerank.get("reranker_bot_score") or rerank.get("bot_score") or 0.0)
    return {
        "user_id": user_id,
        "split": str(rerank.get("split") or base.get("split") or ""),
        "label": label,
        "community_id": str(rerank.get("community_id") or base.get("community_id") or ""),
        "community_size": int(float(rerank.get("community_size") or base.get("community_size") or 0.0)),
        "baseline_predicted_label": baseline_predicted_label,
        "reranker_predicted_label": reranker_predicted_label,
        "baseline_bot_score": round(baseline_score, 8),
        "reranker_bot_score": round(reranker_score, 8),
        "score_delta": round(reranker_score - baseline_score, 8),
        "score_delta_abs": round(abs(reranker_score - baseline_score), 8),
        "baseline_correct": int(label in {"bot", "human"} and baseline_predicted_label == label),
        "reranker_correct": int(label in {"bot", "human"} and reranker_predicted_label == label),
        "baseline_confidence_margin": round(abs(baseline_score - 0.5), 8),
        "reranker_confidence_margin": round(abs(reranker_score - 0.5), 8),
        "username": str(feature_row.get("username") or ""),
        "name": str(feature_row.get("name") or ""),
        "description_excerpt": _excerpt(str(feature_row.get("description") or ""), 160),
        "followers_count": int(_float_value(feature_row.get("followers_count"))),
        "following_count": int(_float_value(feature_row.get("following_count"))),
        "tweets_total": int(_float_value(feature_row.get("tweets_total"))),
        "verified": int(_float_value(feature_row.get("verified"))),
        "can_full_pipeline": int(_float_value(feature_row.get("can_full_pipeline"))),
        "can_triplet": int(_float_value(feature_row.get("can_triplet"))),
        "can_post_type": int(_float_value(feature_row.get("can_post_type"))),
    }


def _build_community_change_rows(focus_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in focus_rows:
        grouped[row["community_id"]].append(row)

    rows = []
    for community_id, items in grouped.items():
        changed_count = sum(1 for item in items if item["baseline_predicted_label"] != item["reranker_predicted_label"])
        fixed_count = sum(1 for item in items if item["baseline_correct"] == 0 and item["reranker_correct"] == 1)
        regressed_count = sum(1 for item in items if item["baseline_correct"] == 1 and item["reranker_correct"] == 0)
        baseline_error_count = sum(1 for item in items if item["baseline_correct"] == 0)
        reranker_error_count = sum(1 for item in items if item["reranker_correct"] == 0)
        rows.append(
            {
                "community_id": community_id,
                "community_size": items[0]["community_size"],
                "changed_count": changed_count,
                "fixed_count": fixed_count,
                "regressed_count": regressed_count,
                "net_gain": fixed_count - regressed_count,
                "baseline_error_count": baseline_error_count,
                "reranker_error_count": reranker_error_count,
                "baseline_error_rate": round(baseline_error_count / len(items), 8) if items else 0.0,
                "reranker_error_rate": round(reranker_error_count / len(items), 8) if items else 0.0,
            }
        )
    return rows


def _compute_confusion_summary(rows: list[dict[str, Any]], *, prefix: str) -> dict[str, int]:
    tp = fp = tn = fn = 0
    predicted_key = f"{prefix}_predicted_label"
    correct_key = f"{prefix}_correct"
    for row in rows:
        label = row["label"]
        predicted = row[predicted_key]
        if predicted == "bot" and label == "bot":
            tp += 1
        elif predicted == "bot" and label == "human":
            fp += 1
        elif predicted == "human" and label == "human":
            tn += 1
        elif predicted == "human" and label == "bot":
            fn += 1
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def _comparison_fieldnames() -> list[str]:
    return [
        "user_id",
        "split",
        "label",
        "community_id",
        "community_size",
        "baseline_predicted_label",
        "reranker_predicted_label",
        "baseline_bot_score",
        "reranker_bot_score",
        "score_delta",
        "score_delta_abs",
        "baseline_correct",
        "reranker_correct",
        "baseline_confidence_margin",
        "reranker_confidence_margin",
        "username",
        "name",
        "description_excerpt",
        "followers_count",
        "following_count",
        "tweets_total",
        "verified",
        "can_full_pipeline",
        "can_triplet",
        "can_post_type",
    ]


def _community_change_fieldnames() -> list[str]:
    return [
        "community_id",
        "community_size",
        "changed_count",
        "fixed_count",
        "regressed_count",
        "net_gain",
        "baseline_error_count",
        "reranker_error_count",
        "baseline_error_rate",
        "reranker_error_rate",
    ]


def _render_comparison_summary(
    manifest: dict[str, Any],
    fixed_rows: list[dict[str, Any]],
    regressed_rows: list[dict[str, Any]],
    community_rows: list[dict[str, Any]],
) -> str:
    counts = manifest["counts"]
    lines = [
        "# Reranker Comparison Summary",
        "",
        "## Overall",
        f"- Focus split: {manifest['focus_split']}",
        f"- Changed predictions: {counts['changed_predictions']}",
        f"- Fixed cases: {counts['fixed_cases']}",
        f"- Regressed cases: {counts['regressed_cases']}",
        f"- Unchanged errors: {counts['unchanged_errors']}",
        "",
        "## Baseline Confusion",
        f"- {manifest['baseline_confusion']}",
        "",
        "## Reranker Confusion",
        f"- {manifest['reranker_confusion']}",
        "",
        "## Top Fixed Cases",
    ]
    for row in fixed_rows:
        lines.append(
            f"- {row['user_id']} ({row['community_id']}): {row['baseline_predicted_label']} -> "
            f"{row['reranker_predicted_label']}, label={row['label']}, "
            f"score {row['baseline_bot_score']} -> {row['reranker_bot_score']}"
        )
    lines.extend(["", "## Top Regressed Cases"])
    for row in regressed_rows:
        lines.append(
            f"- {row['user_id']} ({row['community_id']}): {row['baseline_predicted_label']} -> "
            f"{row['reranker_predicted_label']}, label={row['label']}, "
            f"score {row['baseline_bot_score']} -> {row['reranker_bot_score']}"
        )
    lines.extend(["", "## Communities With Largest Net Gain"])
    for row in community_rows:
        lines.append(
            f"- {row['community_id']}: net_gain={row['net_gain']}, fixed={row['fixed_count']}, "
            f"regressed={row['regressed_count']}, baseline_error_rate={row['baseline_error_rate']}, "
            f"reranker_error_rate={row['reranker_error_rate']}"
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

"""Field-availability audit for sampled datasets."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .normalize import canonical_user_id, has_external_url, normalize_label, normalize_split
from .readers import read_csv_rows, read_jsonl_records, write_csv, write_json


def run_field_audit(
    sample_root: Path,
    output_root: Path,
    *,
    min_triplet_tweets: int = 8,
    min_time_tweets: int = 8,
    min_behavior_tweets: int = 1,
) -> dict[str, Any]:
    """Audit feature availability on an exported sampled dataset."""

    output_root.mkdir(parents=True, exist_ok=True)
    user_path = sample_root / "user.jsonl"
    tweet_path = sample_root / "tweet_0.jsonl"
    edge_path = sample_root / "edge.csv"
    split_path = sample_root / "split.csv"
    label_path = sample_root / "label.csv"

    split_map = {
        canonical_user_id(row.get("id") or row.get("user_id")): normalize_split(row.get("split"))
        for row in read_csv_rows(split_path)
        if (row.get("id") or row.get("user_id"))
    }
    label_map = {
        canonical_user_id(row.get("id") or row.get("user_id")): normalize_label(row.get("label"))
        for row in read_csv_rows(label_path)
        if (row.get("id") or row.get("user_id"))
    }

    per_user: dict[str, dict[str, Any]] = {}
    for record in read_jsonl_records(user_path):
        user_id = canonical_user_id(record.get("id"))
        if not user_id:
            continue
        metrics = record.get("public_metrics") if isinstance(record.get("public_metrics"), dict) else {}
        per_user[user_id] = {
            "user_id": user_id,
            "split": split_map.get(user_id, "missing"),
            "label": label_map.get(user_id, "missing"),
            "user_created_at_present": int(record.get("created_at") not in (None, "")),
            "description_present": int(bool((record.get("description") or "").strip())),
            "profile_url_present": int(record.get("url") not in (None, "")),
            "followers_count_present": int(metrics.get("followers_count") is not None),
            "following_count_present": int(metrics.get("following_count") is not None),
            "listed_count_present": int(metrics.get("listed_count") is not None),
            "user_tweet_count_present": int(metrics.get("tweet_count") is not None),
            "verified_present": int(record.get("verified") is not None),
            "verified_true": int(bool(record.get("verified"))),
            "tweets_total": 0,
            "tweets_with_text": 0,
            "tweets_with_created_at": 0,
            "tweets_with_public_metrics": 0,
            "tweets_with_like_count": 0,
            "tweets_with_reply_count": 0,
            "tweets_with_retweet_count": 0,
            "tweets_with_quote_count": 0,
            "tweets_with_references": 0,
            "tweets_with_external_url": 0,
            "tweets_with_lang": 0,
            "tweets_with_source": 0,
            "following_out_degree": 0,
            "following_in_degree": 0,
            "post_edge_count": 0,
        }

    language_counter: Counter[str] = Counter()
    for record in read_jsonl_records(tweet_path):
        author_id = canonical_user_id(record.get("author_id"))
        if not author_id:
            continue
        row = per_user.setdefault(
            author_id,
            {
                "user_id": author_id,
                "split": split_map.get(author_id, "missing"),
                "label": label_map.get(author_id, "missing"),
                "user_created_at_present": 0,
                "description_present": 0,
                "profile_url_present": 0,
                "followers_count_present": 0,
                "following_count_present": 0,
                "listed_count_present": 0,
                "user_tweet_count_present": 0,
                "verified_present": 0,
                "verified_true": 0,
                "tweets_total": 0,
                "tweets_with_text": 0,
                "tweets_with_created_at": 0,
                "tweets_with_public_metrics": 0,
                "tweets_with_like_count": 0,
                "tweets_with_reply_count": 0,
                "tweets_with_retweet_count": 0,
                "tweets_with_quote_count": 0,
                "tweets_with_references": 0,
                "tweets_with_external_url": 0,
                "tweets_with_lang": 0,
                "tweets_with_source": 0,
                "following_out_degree": 0,
                "following_in_degree": 0,
                "post_edge_count": 0,
            },
        )
        row["tweets_total"] += 1
        text = (record.get("text") or "").strip()
        metrics = record.get("public_metrics") if isinstance(record.get("public_metrics"), dict) else {}
        if text:
            row["tweets_with_text"] += 1
        if record.get("created_at") not in (None, ""):
            row["tweets_with_created_at"] += 1
        if metrics:
            row["tweets_with_public_metrics"] += 1
        if metrics.get("like_count") is not None:
            row["tweets_with_like_count"] += 1
        if metrics.get("reply_count") is not None:
            row["tweets_with_reply_count"] += 1
        if metrics.get("retweet_count") is not None:
            row["tweets_with_retweet_count"] += 1
        if metrics.get("quote_count") is not None:
            row["tweets_with_quote_count"] += 1
        if record.get("referenced_tweets"):
            row["tweets_with_references"] += 1
        if has_external_url(record):
            row["tweets_with_external_url"] += 1
        if record.get("lang") not in (None, ""):
            row["tweets_with_lang"] += 1
            language_counter[str(record["lang"])] += 1
        if record.get("source") not in (None, ""):
            row["tweets_with_source"] += 1

    for edge in read_csv_rows(edge_path):
        source_id = canonical_user_id(edge.get("source_id"))
        target_id = canonical_user_id(edge.get("target_id"))
        relation = str(edge.get("relation") or "")
        if relation == "following":
            if source_id in per_user:
                per_user[source_id]["following_out_degree"] += 1
            if target_id in per_user:
                per_user[target_id]["following_in_degree"] += 1
        elif relation == "post" and source_id in per_user:
            per_user[source_id]["post_edge_count"] += 1

    audit_rows = []
    for row in per_user.values():
        row["can_triplet"] = int(row["tweets_with_text"] >= min_triplet_tweets)
        row["can_post_type"] = int(row["tweets_with_text"] >= 1)
        row["can_time_feature"] = int(row["tweets_with_created_at"] >= min_time_tweets)
        row["can_behavior_feature"] = int(
            row["followers_count_present"]
            and row["following_count_present"]
            and row["user_tweet_count_present"]
            and row["tweets_with_public_metrics"] >= min_behavior_tweets
        )
        row["can_network_feature"] = int((row["following_in_degree"] + row["following_out_degree"]) > 0)
        row["can_full_pipeline"] = int(
            row["can_triplet"]
            and row["can_post_type"]
            and row["can_time_feature"]
            and row["can_behavior_feature"]
            and row["can_network_feature"]
        )
        audit_rows.append(row)

    audit_rows.sort(key=lambda item: item["user_id"])
    write_csv(output_root / "user_feature_availability.csv", _audit_fieldnames(), audit_rows)

    summary = _build_audit_summary(
        audit_rows,
        sample_root=sample_root,
        language_counter=language_counter,
        min_triplet_tweets=min_triplet_tweets,
        min_time_tweets=min_time_tweets,
        min_behavior_tweets=min_behavior_tweets,
    )
    write_json(output_root / "audit_summary.json", summary)
    (output_root / "audit_summary.md").write_text(_render_audit_markdown(summary), encoding="utf-8")
    return summary


def _build_audit_summary(
    rows: list[dict[str, Any]],
    *,
    sample_root: Path,
    language_counter: Counter[str],
    min_triplet_tweets: int,
    min_time_tweets: int,
    min_behavior_tweets: int,
) -> dict[str, Any]:
    total_users = len(rows)
    total_tweets = sum(int(row["tweets_total"]) for row in rows)
    label_counter = Counter(row["label"] for row in rows)
    split_counter = Counter(row["split"] for row in rows)
    counts = {
        "triplet_ready_users": sum(int(row["can_triplet"]) for row in rows),
        "post_type_ready_users": sum(int(row["can_post_type"]) for row in rows),
        "time_ready_users": sum(int(row["can_time_feature"]) for row in rows),
        "behavior_ready_users": sum(int(row["can_behavior_feature"]) for row in rows),
        "network_ready_users": sum(int(row["can_network_feature"]) for row in rows),
        "full_pipeline_ready_users": sum(int(row["can_full_pipeline"]) for row in rows),
        "users_with_external_urls": sum(int(row["tweets_with_external_url"] > 0) for row in rows),
        "users_with_referenced_tweets": sum(int(row["tweets_with_references"] > 0) for row in rows),
    }
    by_label = {
        label: {
            "users": count,
            "triplet_ready_users": sum(int(row["can_triplet"]) for row in rows if row["label"] == label),
            "post_type_ready_users": sum(int(row["can_post_type"]) for row in rows if row["label"] == label),
            "time_ready_users": sum(int(row["can_time_feature"]) for row in rows if row["label"] == label),
            "full_pipeline_ready_users": sum(int(row["can_full_pipeline"]) for row in rows if row["label"] == label),
        }
        for label, count in sorted(label_counter.items())
    }
    summary = {
        "sample_root": str(sample_root),
        "thresholds": {
            "min_triplet_tweets": min_triplet_tweets,
            "min_time_tweets": min_time_tweets,
            "min_behavior_tweets": min_behavior_tweets,
        },
        "overall": {
            "users": total_users,
            "tweets": total_tweets,
            "avg_tweets_per_user": round(total_tweets / total_users, 4) if total_users else 0.0,
        },
        "split_distribution": dict(sorted(split_counter.items())),
        "label_distribution": dict(sorted(label_counter.items())),
        "availability_counts": counts,
        "availability_rates": {
            key: round(value / total_users, 6) if total_users else 0.0 for key, value in counts.items()
        },
        "language_top10": language_counter.most_common(10),
        "by_label": by_label,
    }
    return summary


def _render_audit_markdown(summary: dict[str, Any]) -> str:
    overall = summary["overall"]
    counts = summary["availability_counts"]
    rates = summary["availability_rates"]
    lines = [
        "# Field Availability Audit",
        "",
        "## Overall",
        f"- Users: {overall['users']}",
        f"- Tweets: {overall['tweets']}",
        f"- Average tweets per user: {overall['avg_tweets_per_user']:.2f}",
        "",
        "## Split Distribution",
    ]
    lines.extend(f"- {key}: {value}" for key, value in summary["split_distribution"].items())
    lines.extend(["", "## Label Distribution"])
    lines.extend(f"- {key}: {value}" for key, value in summary["label_distribution"].items())
    lines.extend(["", "## Feature Readiness"])
    lines.extend(
        f"- {key}: {counts[key]} ({rates[key] * 100:.2f}%)"
        for key in (
            "triplet_ready_users",
            "post_type_ready_users",
            "time_ready_users",
            "behavior_ready_users",
            "network_ready_users",
            "full_pipeline_ready_users",
            "users_with_external_urls",
            "users_with_referenced_tweets",
        )
    )
    lines.extend(["", "## Top Languages"])
    lines.extend(f"- {lang}: {count}" for lang, count in summary["language_top10"])
    lines.extend(["", "## Readiness By Label"])
    for label, payload in summary["by_label"].items():
        lines.append(
            f"- {label}: users={payload['users']}, triplet_ready={payload['triplet_ready_users']}, "
            f"post_type_ready={payload['post_type_ready_users']}, time_ready={payload['time_ready_users']}, "
            f"full_pipeline_ready={payload['full_pipeline_ready_users']}"
        )
    return "\n".join(lines) + "\n"


def _audit_fieldnames() -> list[str]:
    return [
        "user_id",
        "split",
        "label",
        "user_created_at_present",
        "description_present",
        "profile_url_present",
        "followers_count_present",
        "following_count_present",
        "listed_count_present",
        "user_tweet_count_present",
        "verified_present",
        "verified_true",
        "tweets_total",
        "tweets_with_text",
        "tweets_with_created_at",
        "tweets_with_public_metrics",
        "tweets_with_like_count",
        "tweets_with_reply_count",
        "tweets_with_retweet_count",
        "tweets_with_quote_count",
        "tweets_with_references",
        "tweets_with_external_url",
        "tweets_with_lang",
        "tweets_with_source",
        "following_out_degree",
        "following_in_degree",
        "post_edge_count",
        "can_triplet",
        "can_post_type",
        "can_time_feature",
        "can_behavior_feature",
        "can_network_feature",
        "can_full_pipeline",
    ]


def _string_value(value: Any) -> str:
    if value in (None, ""):
        return ""
    return str(value)

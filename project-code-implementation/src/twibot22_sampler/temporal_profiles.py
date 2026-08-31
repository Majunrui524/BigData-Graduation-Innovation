"""Build user temporal posting profiles from sampled tweets."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from .normalize import canonical_user_id, parse_timestamp
from .readers import iter_json_records, read_csv_rows, read_jsonl_records, write_json, write_jsonl

DEFAULT_TEMPORAL_MIN_TWEETS = 8


def build_temporal_profiles(
    sample_root: Path,
    output_root: Path,
    *,
    min_time_tweets: int = DEFAULT_TEMPORAL_MIN_TWEETS,
) -> dict[str, Any]:
    """Build UTC-hour posting histograms for each sampled user."""

    output_root.mkdir(parents=True, exist_ok=True)

    user_ids = _load_user_ids(sample_root / "user.jsonl")
    tweet_author_map, post_counts = _load_post_edges(sample_root / "edge.csv")
    hour_counts: dict[str, list[int]] = {user_id: [0] * 24 for user_id in user_ids}
    created_at_counts: Counter[str] = Counter()

    for tweet_path in sorted(sample_root.glob("tweet_*.jsonl")) + sorted(sample_root.glob("tweet_*.json")):
        for record in iter_json_records(tweet_path):
            tweet_id = str(record.get("id") or record.get("tweet_id") or "")
            author_id = tweet_author_map.get(tweet_id, "")
            if not author_id:
                continue
            timestamp = parse_timestamp(record.get("created_at"))
            if timestamp is None:
                continue
            hour_counts.setdefault(author_id, [0] * 24)[timestamp.hour] += 1
            created_at_counts[author_id] += 1

    rows = []
    for user_id in user_ids:
        created_at_tweets = int(created_at_counts.get(user_id, 0))
        histogram = list(hour_counts.get(user_id, [0] * 24))
        if created_at_tweets > 0:
            distribution = [round(value / created_at_tweets, 8) for value in histogram]
        else:
            distribution = [0.0] * 24
        rows.append(
            {
                "user_id": user_id,
                "total_tweets": int(post_counts.get(user_id, 0)),
                "created_at_tweets": created_at_tweets,
                "temporal_ready": int(created_at_tweets >= max(int(min_time_tweets), 1)),
                "utc_hour_counts": histogram,
                "utc_hour_distribution": distribution,
            }
        )

    jsonl_path = output_root / "user_temporal_profiles.jsonl"
    manifest_path = output_root / "temporal_manifest.json"
    write_jsonl(jsonl_path, rows)

    manifest = {
        "sample_root": str(sample_root),
        "output_root": str(output_root),
        "min_time_tweets": max(int(min_time_tweets), 1),
        "counts": {
            "users": len(rows),
            "temporal_ready_users": sum(int(row["temporal_ready"]) for row in rows),
            "total_post_edges": sum(int(post_counts.get(user_id, 0)) for user_id in user_ids),
            "tweets_with_created_at": sum(int(row["created_at_tweets"]) for row in rows),
        },
        "files": {
            "profiles": str(jsonl_path),
        },
    }
    write_json(manifest_path, manifest)
    return manifest


def _load_user_ids(path: Path) -> list[str]:
    user_ids = []
    for record in read_jsonl_records(path):
        user_id = canonical_user_id(record.get("id") or record.get("user_id"))
        if user_id:
            user_ids.append(user_id)
    user_ids.sort()
    return user_ids


def _load_post_edges(path: Path) -> tuple[dict[str, str], Counter[str]]:
    tweet_author_map: dict[str, str] = {}
    post_counts: Counter[str] = Counter()
    for row in read_csv_rows(path):
        if str(row.get("relation") or "").strip().lower() != "post":
            continue
        author_id = canonical_user_id(row.get("source_id"))
        tweet_id = str(row.get("target_id") or "").strip()
        if not author_id or not tweet_id:
            continue
        tweet_author_map[tweet_id] = author_id
        post_counts[author_id] += 1
    return tweet_author_map, post_counts

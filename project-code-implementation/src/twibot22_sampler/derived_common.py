"""Shared helpers for derived feature generation on sampled tweets."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .normalize import canonical_user_id, parse_timestamp, stable_hexdigest
from .readers import read_jsonl_records


def select_tweets_for_derived_tasks(
    sample_root: Path,
    *,
    per_user_limit: int,
    min_user_tweets: int,
    max_users: int | None,
    max_tweets: int | None,
    seed: int,
    require_text: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select a bounded, deterministic tweet set for downstream processing."""

    tweet_path = sample_root / "tweet_0.jsonl"
    per_author_candidates: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    per_author_seen_counts: Counter[str] = Counter()

    for record in read_jsonl_records(tweet_path):
        author_id = canonical_user_id(record.get("author_id"))
        tweet_id = _as_string(record.get("id"))
        text = (record.get("text") or "").strip()
        if not author_id or not tweet_id:
            continue
        if require_text and not text:
            continue
        per_author_seen_counts[author_id] += 1
        rank = stable_hexdigest(seed, "derived", author_id, tweet_id)
        bucket = per_author_candidates[author_id]
        bucket.append((rank, record))
        bucket.sort(key=lambda item: item[0])
        if len(bucket) > per_user_limit:
            del bucket[per_user_limit:]

    eligible_authors = [
        author_id for author_id, count in per_author_seen_counts.items() if count >= min_user_tweets
    ]
    eligible_authors = sorted(eligible_authors, key=lambda author_id: stable_hexdigest(seed, "author", author_id))
    if max_users is not None:
        eligible_authors = eligible_authors[:max_users]
    selected_author_set = set(eligible_authors)

    selected_records: list[dict[str, Any]] = []
    for author_id in eligible_authors:
        ranked_records = per_author_candidates.get(author_id, [])
        ordered_records = sorted(
            (record for _rank, record in ranked_records),
            key=lambda record: (_timestamp_sort_key(record), _as_string(record.get("id"))),
        )
        selected_records.extend(ordered_records)

    selected_records.sort(
        key=lambda record: (
            _as_string(record.get("author_id")),
            _timestamp_sort_key(record),
            _as_string(record.get("id")),
        )
    )
    if max_tweets is not None:
        selected_records = sorted(
            selected_records,
            key=lambda record: stable_hexdigest(
                seed,
                "global-tweet",
                _as_string(record.get("author_id")),
                _as_string(record.get("id")),
            ),
        )[:max_tweets]
        selected_records.sort(
            key=lambda record: (
                _as_string(record.get("author_id")),
                _timestamp_sort_key(record),
                _as_string(record.get("id")),
            )
        )

    summary = {
        "per_user_limit": per_user_limit,
        "min_user_tweets": min_user_tweets,
        "max_users": max_users,
        "max_tweets": max_tweets,
        "eligible_user_count": len(selected_author_set),
        "selected_tweet_count": len(selected_records),
        "observed_author_count": len(per_author_seen_counts),
        "observed_tweet_count": sum(per_author_seen_counts.values()),
    }
    return selected_records, summary


def read_processed_ids(path: Path) -> set[str]:
    """Read already processed tweet ids from a JSONL output file."""

    if not path.exists():
        return set()
    return {
        _as_string(record.get("tweet_id"))
        for record in read_jsonl_records(path)
        if _as_string(record.get("tweet_id"))
    }


def _timestamp_sort_key(record: dict[str, Any]) -> tuple[int, str]:
    parsed = parse_timestamp(record.get("created_at"))
    if parsed is None:
        return (1, "")
    return (0, parsed.isoformat())


def _as_string(value: Any) -> str:
    if value in (None, ""):
        return ""
    return str(value)

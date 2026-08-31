"""Tweet candidate collection, sampling, and relation closure."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import POST_PREFILTER_MULTIPLIER, SamplingThresholds
from .normalize import (
    canonical_follow_edge,
    canonical_tweet_relation,
    extract_referenced_tweet_ids,
    normalize_user_id,
    parse_timestamp,
    select_evenly,
    stable_hexdigest,
)
from .readers import iter_json_records, read_csv_rows


def collect_post_candidates_and_user_edges(
    edge_path: Path,
    *,
    final_user_ids: set[str],
    seed_user_ids: set[str],
    thresholds: SamplingThresholds,
    seed: int,
) -> tuple[dict[str, list[str]], set[tuple[str, str, str]], dict[str, Any]]:
    """Second pass over edge.csv for post candidates and final user-user edges."""

    candidate_store: dict[str, list[tuple[str, str]]] = {}
    final_user_edges: set[tuple[str, str, str]] = set()
    candidate_counter = 0

    for row in read_csv_rows(edge_path):
        follow_edge = canonical_follow_edge(
            row.get("source_id", ""),
            row.get("target_id", ""),
            row.get("relation", ""),
        )
        if follow_edge is not None:
            follower_id, followed_id, relation = follow_edge
            if follower_id in final_user_ids and followed_id in final_user_ids:
                final_user_edges.add((follower_id, followed_id, relation))
            continue

        tweet_edge = canonical_tweet_relation(
            row.get("source_id", ""),
            row.get("target_id", ""),
            row.get("relation", ""),
        )
        if tweet_edge is None:
            continue
        source_id, target_id, relation = tweet_edge
        if relation != "post" or source_id not in final_user_ids:
            continue
        candidate_counter += 1
        quota = thresholds.seed_user_max_tweets if source_id in seed_user_ids else thresholds.context_user_max_tweets
        _offer_ranked_tweet(
            candidate_store,
            user_id=source_id,
            tweet_id=target_id,
            rank=stable_hexdigest(seed, "post-candidate", source_id, target_id),
            limit=quota * POST_PREFILTER_MULTIPLIER,
        )

    post_candidates = {
        user_id: [tweet_id for _rank, tweet_id in sorted(records)]
        for user_id, records in candidate_store.items()
    }
    summary = {
        "post_candidate_users": len(post_candidates),
        "post_candidate_edges_scanned": candidate_counter,
        "final_user_edge_count": len(final_user_edges),
    }
    return post_candidates, final_user_edges, summary


def load_tweet_records(tweet_paths: list[Path], target_tweet_ids: set[str]) -> dict[str, dict[str, Any]]:
    """Fetch a target tweet subset from tweet shards."""

    records: dict[str, dict[str, Any]] = {}
    if not target_tweet_ids:
        return records
    remaining = set(target_tweet_ids)
    for path in tweet_paths:
        if not remaining:
            break
        for record in iter_json_records(path):
            tweet_id = normalize_user_id(record)
            if tweet_id not in remaining:
                continue
            records[tweet_id] = record
            remaining.discard(tweet_id)
            if not remaining:
                break
    return records


def finalize_post_selection(
    post_candidates_by_user: dict[str, list[str]],
    *,
    tweet_records: dict[str, dict[str, Any]],
    seed_user_ids: set[str],
    thresholds: SamplingThresholds,
    seed: int,
) -> tuple[dict[str, dict[str, Any]], set[tuple[str, str, str]], dict[str, Any]]:
    """Choose final tweet records per user and emit post edges."""

    selected_records: dict[str, dict[str, Any]] = {}
    post_edges: set[tuple[str, str, str]] = set()
    sampled_counts: dict[str, int] = {}

    for user_id, tweet_ids in post_candidates_by_user.items():
        existing_records = [tweet_records[tweet_id] for tweet_id in tweet_ids if tweet_id in tweet_records]
        limit = thresholds.seed_user_max_tweets if user_id in seed_user_ids else thresholds.context_user_max_tweets
        selected = _sample_user_tweets(existing_records, limit=limit, seed=seed, user_id=user_id)
        sampled_counts[user_id] = len(selected)
        for record in selected:
            tweet_id = normalize_user_id(record)
            selected_records[tweet_id] = record
            post_edges.add((user_id, tweet_id, "post"))

    summary = {
        "selected_post_tweet_count": len(selected_records),
        "selected_post_edge_count": len(post_edges),
        "per_user_selected_counts": sampled_counts,
    }
    return selected_records, post_edges, summary


def expand_referenced_tweets(
    tweet_paths: list[Path],
    *,
    selected_tweet_records: dict[str, dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Fetch one-hop referenced tweets for the already selected records."""

    selected_ids = set(selected_tweet_records)
    reference_ids: set[str] = set()
    for record in selected_tweet_records.values():
        reference_ids.update(extract_referenced_tweet_ids(record))
    missing_reference_ids = reference_ids - selected_ids
    referenced_records = load_tweet_records(tweet_paths, missing_reference_ids)
    merged = dict(selected_tweet_records)
    merged.update(referenced_records)
    summary = {
        "referenced_tweet_candidates": len(reference_ids),
        "referenced_tweets_added": len(referenced_records),
        "final_tweet_count": len(merged),
    }
    return merged, summary


def collect_third_pass_edges(
    edge_path: Path,
    *,
    final_user_ids: set[str],
    final_tweet_ids: set[str],
) -> tuple[set[tuple[str, str, str]], dict[str, int]]:
    """Third pass over edge.csv for tweet-tweet and mention relations."""

    edges: set[tuple[str, str, str]] = set()
    relation_counts = {"retweet": 0, "quote": 0, "reply": 0, "mention": 0}

    for row in read_csv_rows(edge_path):
        canonical = canonical_tweet_relation(
            row.get("source_id", ""),
            row.get("target_id", ""),
            row.get("relation", ""),
        )
        if canonical is None:
            continue
        source_id, target_id, relation = canonical
        if relation == "post":
            continue
        if relation == "mention":
            if source_id in final_tweet_ids and target_id in final_user_ids:
                edges.add((source_id, target_id, relation))
        elif source_id in final_tweet_ids and target_id in final_tweet_ids:
            edges.add((source_id, target_id, relation))

    for _source_id, _target_id, relation in edges:
        relation_counts[relation] += 1
    return edges, relation_counts


def _offer_ranked_tweet(
    store: dict[str, list[tuple[str, str]]],
    *,
    user_id: str,
    tweet_id: str,
    rank: str,
    limit: int,
) -> None:
    entries = store.setdefault(user_id, [])
    entries.append((rank, tweet_id))
    entries.sort()
    if len(entries) > limit:
        del entries[limit:]


def _sample_user_tweets(
    records: list[dict[str, Any]],
    *,
    limit: int,
    seed: int,
    user_id: str,
) -> list[dict[str, Any]]:
    if limit <= 0 or not records:
        return []
    if len(records) <= limit:
        return list(records)
    parsed_records = []
    for record in records:
        parsed_records.append((parse_timestamp(record.get("created_at")), record))
    if all(timestamp is not None for timestamp, _record in parsed_records):
        ordered = [record for timestamp, record in sorted(parsed_records, key=lambda item: item[0])]
        return select_evenly(ordered, limit)
    ordered = sorted(
        records,
        key=lambda record: stable_hexdigest(seed, "tweet", user_id, normalize_user_id(record)),
    )
    return ordered[:limit]

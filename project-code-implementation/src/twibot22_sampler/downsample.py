"""Secondary downsampling from an already exported sample subset."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from . import config
from .export import build_sample_stats_markdown
from .normalize import (
    assign_bucket,
    canonical_tweet_relation,
    canonical_user_id,
    normalize_followers_count,
    normalize_following_count,
    normalize_label,
    normalize_split,
    normalize_verified_bucket,
    quantile_edges,
    stable_hexdigest,
)
from .readers import read_csv_rows, read_jsonl_records, write_csv, write_json, write_jsonl
from .user_sampling import allocate_largest_remainder

USER_EXPORT_FILENAMES = ("user.jsonl", "tweet_0.jsonl", "edge.csv", "split.csv", "label.csv")


def downsample_exported_sample(
    sample_root: Path,
    output_root: Path,
    *,
    target_users: int,
    seed: int = config.DEFAULT_SEED,
) -> dict[str, Any]:
    """Create a smaller derived sample from an existing exported sample."""

    _ensure_output_root_ready(output_root)

    user_records = _load_user_records(sample_root / "user.jsonl")
    split_map = {
        canonical_user_id(row.get("id") or row.get("user_id")): normalize_split(row.get("split"))
        for row in read_csv_rows(sample_root / "split.csv")
    }
    label_map = {
        canonical_user_id(row.get("id") or row.get("user_id")): normalize_label(row.get("label"))
        for row in read_csv_rows(sample_root / "label.csv")
    }

    authored_tweet_counts = _count_tweets_by_author(sample_root / "tweet_0.jsonl")
    profile_rows = _build_profile_rows(
        user_records=user_records,
        split_map=split_map,
        label_map=label_map,
        authored_tweet_counts=authored_tweet_counts,
    )
    selected_user_ids, selection_summary = select_downsample_users(
        profile_rows,
        target_users=target_users,
        seed=seed,
    )
    selected_user_set = set(selected_user_ids)

    candidate_tweets = _collect_tweets_for_users(sample_root / "tweet_0.jsonl", selected_user_set)
    user_edges, post_edges, kept_tweet_ids = _collect_primary_edges(
        sample_root / "edge.csv",
        selected_user_set,
        set(candidate_tweets),
    )
    kept_tweet_records = {tweet_id: candidate_tweets[tweet_id] for tweet_id in kept_tweet_ids if tweet_id in candidate_tweets}
    extra_edges, tweet_relation_counts = _collect_secondary_edges(
        sample_root / "edge.csv",
        selected_user_set,
        kept_tweet_ids,
    )

    selected_user_records = {user_id: user_records[user_id] for user_id in selected_user_ids if user_id in user_records}
    edge_rows = sorted(user_edges | post_edges | extra_edges, key=lambda item: (item[2], item[0], item[1]))
    split_rows = [{"id": user_id, "split": split_map.get(user_id, "missing")} for user_id in sorted(selected_user_records)]
    label_rows = [{"id": user_id, "label": label_map.get(user_id, "missing")} for user_id in sorted(selected_user_records)]

    output_root.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_root / "user.jsonl", [selected_user_records[user_id] for user_id in sorted(selected_user_records)])
    write_jsonl(output_root / "tweet_0.jsonl", [kept_tweet_records[tweet_id] for tweet_id in sorted(kept_tweet_records)])
    write_csv(
        output_root / "edge.csv",
        ["source_id", "target_id", "relation"],
        (
            {"source_id": source_id, "target_id": target_id, "relation": relation}
            for source_id, target_id, relation in edge_rows
        ),
    )
    write_csv(output_root / "split.csv", ["id", "split"], split_rows)
    write_csv(output_root / "label.csv", ["id", "label"], label_rows)
    for filename in config.EMPTY_EXPORT_FILENAMES:
        (output_root / filename).write_text("", encoding="utf-8")

    relation_distribution = Counter(relation for _source_id, _target_id, relation in edge_rows)
    verified_distribution = Counter(normalize_verified_bucket(selected_user_records[user_id]) for user_id in selected_user_records)
    split_distribution = Counter(row["split"] for row in split_rows)
    label_distribution = Counter(row["label"] for row in label_rows)
    tweet_counts_per_user = Counter(source_id for source_id, _target_id, relation in edge_rows if relation == "post")
    label_counts_by_split = _build_nested_label_counts(label_rows, split_rows)

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_sample_root": str(sample_root),
        "preset": "downsample-final",
        "seed": seed,
        "sample_thresholds": {},
        "relation_whitelist": list(config.OUTPUT_RELATIONS),
        "seed_sampling": {
            "requested_seed_count": min(max(target_users, 0), len(profile_rows)),
            "selected_seed_count": len(selected_user_records),
            "split_quotas": dict(split_distribution),
            "label_quotas": label_counts_by_split,
            "selected_seed_ids": sorted(selected_user_records),
            "selection_targets": selection_summary,
        },
        "context_expansion": {
            "candidate_count": 0,
            "context_user_count": 0,
            "final_user_count": len(selected_user_records),
            "hub_degree_cutoff_p95": 0,
        },
        "second_pass": {
            "post_candidate_users": len({canonical_user_id(record.get("author_id")) for record in kept_tweet_records.values()}),
            "post_candidate_edges_scanned": len(post_edges),
            "final_user_edge_count": len(user_edges),
        },
        "post_selection": {
            "selected_post_tweet_count": len(kept_tweet_records),
        },
        "reference_closure": {
            "referenced_tweet_candidates": 0,
            "referenced_tweets_added": 0,
            "final_tweet_count": len(kept_tweet_records),
        },
        "tweet_relations": tweet_relation_counts,
        "final_counts": {
            "users": len(selected_user_records),
            "tweets": len(kept_tweet_records),
            "edges": len(edge_rows),
            "seed_users": len(selected_user_records),
            "context_users": 0,
        },
        "distributions": {
            "split": dict(split_distribution),
            "label": dict(label_distribution),
            "verified": dict(verified_distribution),
            "relation": dict(relation_distribution),
        },
        "tweet_budget": {
            "limit": len(kept_tweet_records),
            "used": len(kept_tweet_records),
            "within_budget": True,
        },
        "derivation": {
            "mode": "downsample-final",
            "target_users_requested": target_users,
            "target_users_selected": len(selected_user_records),
        },
    }
    write_json(output_root / "sample_manifest.json", manifest)
    stats_markdown = build_sample_stats_markdown(
        manifest=manifest,
        tweet_counts_per_user=tweet_counts_per_user,
    )
    (output_root / "sample_stats.md").write_text(stats_markdown, encoding="utf-8")
    return manifest


def select_downsample_users(
    profile_rows: list[dict[str, Any]],
    *,
    target_users: int,
    seed: int,
) -> tuple[list[str], dict[str, Any]]:
    """Select users while preserving split and label proportions."""

    if not profile_rows:
        return [], {"split_targets": {}, "label_targets": {}}
    target = min(max(int(target_users), 0), len(profile_rows))
    split_groups = _group_rows(profile_rows, "split")
    split_targets = allocate_largest_remainder(target, {split: len(rows) for split, rows in split_groups.items()})
    label_targets: dict[str, dict[str, int]] = {}

    selected_ids: list[str] = []
    for split, split_rows in sorted(split_groups.items()):
        label_groups = _group_rows(split_rows, "label")
        split_label_targets = allocate_largest_remainder(
            split_targets.get(split, 0),
            {label: len(rows) for label, rows in label_groups.items()},
        )
        label_targets[split] = split_label_targets
        for label, label_rows in sorted(label_groups.items()):
            chosen_rows = _select_stratified_rows(
                label_rows,
                target=split_label_targets.get(label, 0),
                seed=seed,
                namespace=f"{split}:{label}",
            )
            selected_ids.extend(row["user_id"] for row in chosen_rows)

    ordered_ids = sorted(set(selected_ids), key=lambda user_id: stable_hexdigest(seed, "downsample-final", user_id))
    return ordered_ids, {
        "split_targets": split_targets,
        "label_targets": label_targets,
    }


def _load_user_records(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for record in read_jsonl_records(path):
        user_id = canonical_user_id(record.get("id"))
        if user_id:
            records[user_id] = record
    return records


def _count_tweets_by_author(path: Path) -> Counter[str]:
    counts: Counter[str] = Counter()
    for record in read_jsonl_records(path):
        author_id = canonical_user_id(record.get("author_id"))
        if author_id:
            counts[author_id] += 1
    return counts


def _build_profile_rows(
    *,
    user_records: dict[str, dict[str, Any]],
    split_map: dict[str, str],
    label_map: dict[str, str],
    authored_tweet_counts: Counter[str],
) -> list[dict[str, Any]]:
    rows = []
    for user_id, record in user_records.items():
        followers_count = normalize_followers_count(record)
        following_count = normalize_following_count(record)
        rows.append(
            {
                "user_id": user_id,
                "split": split_map.get(user_id, "missing"),
                "label": label_map.get(user_id, "missing"),
                "verified_bucket": normalize_verified_bucket(record),
                "followers_count": followers_count,
                "following_count": following_count,
                "tweet_count_hint": int(authored_tweet_counts.get(user_id, 0)),
                "degree_proxy": followers_count + following_count,
            }
        )
    return rows


def _select_stratified_rows(
    rows: list[dict[str, Any]],
    *,
    target: int,
    seed: int,
    namespace: str,
) -> list[dict[str, Any]]:
    if target <= 0 or not rows:
        return []
    working_rows = [dict(row) for row in rows]
    tweet_edges = quantile_edges([int(row["tweet_count_hint"]) for row in working_rows], buckets=4)
    follower_edges = quantile_edges([int(row["followers_count"]) for row in working_rows], buckets=4)
    following_edges = quantile_edges([int(row["following_count"]) for row in working_rows], buckets=4)
    for row in working_rows:
        row["stratum"] = "|".join(
            (
                assign_bucket(int(row["tweet_count_hint"]), tweet_edges),
                assign_bucket(int(row["followers_count"]), follower_edges),
                assign_bucket(int(row["following_count"]), following_edges),
                row.get("verified_bucket", "missing"),
            )
        )
    strata_groups = _group_rows(working_rows, "stratum")
    strata_targets = allocate_largest_remainder(target, {stratum: len(group) for stratum, group in strata_groups.items()})

    selected: list[dict[str, Any]] = []
    remainder: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    for stratum, group_rows in sorted(strata_groups.items()):
        ordered = sorted(group_rows, key=lambda row: stable_hexdigest(seed, namespace, stratum, row["user_id"]))
        chosen = ordered[: strata_targets.get(stratum, 0)]
        selected.extend(chosen)
        selected_ids.update(row["user_id"] for row in chosen)
        remainder.extend(ordered[strata_targets.get(stratum, 0) :])
    if len(selected) < target:
        fallback = sorted(
            (row for row in remainder if row["user_id"] not in selected_ids),
            key=lambda row: stable_hexdigest(seed, namespace, "fallback", row["user_id"]),
        )
        selected.extend(fallback[: target - len(selected)])
    return selected[:target]


def _collect_tweets_for_users(tweet_path: Path, selected_user_ids: set[str]) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for record in read_jsonl_records(tweet_path):
        tweet_id = str(record.get("id") or "")
        author_id = canonical_user_id(record.get("author_id"))
        if tweet_id and author_id in selected_user_ids:
            records[tweet_id] = record
    return records


def _collect_primary_edges(
    edge_path: Path,
    selected_user_ids: set[str],
    candidate_tweet_ids: set[str],
) -> tuple[set[tuple[str, str, str]], set[tuple[str, str, str]], set[str]]:
    user_edges: set[tuple[str, str, str]] = set()
    post_edges: set[tuple[str, str, str]] = set()
    kept_tweet_ids: set[str] = set()
    for row in read_csv_rows(edge_path):
        source_id = str(row.get("source_id") or "")
        target_id = str(row.get("target_id") or "")
        relation = str(row.get("relation") or "")
        if relation == "following":
            if source_id in selected_user_ids and target_id in selected_user_ids:
                user_edges.add((source_id, target_id, relation))
        elif relation == "post":
            if source_id in selected_user_ids and target_id in candidate_tweet_ids:
                post_edges.add((source_id, target_id, relation))
                kept_tweet_ids.add(target_id)
    return user_edges, post_edges, kept_tweet_ids


def _collect_secondary_edges(
    edge_path: Path,
    selected_user_ids: set[str],
    kept_tweet_ids: set[str],
) -> tuple[set[tuple[str, str, str]], dict[str, int]]:
    extra_edges: set[tuple[str, str, str]] = set()
    relation_counts: Counter[str] = Counter()
    for row in read_csv_rows(edge_path):
        source_id = str(row.get("source_id") or "")
        target_id = str(row.get("target_id") or "")
        relation = str(row.get("relation") or "")
        if relation in {"retweet", "quote", "reply"}:
            if source_id in kept_tweet_ids and target_id in kept_tweet_ids:
                extra_edges.add((source_id, target_id, relation))
                relation_counts[relation] += 1
        elif relation == "mention":
            if source_id in kept_tweet_ids and target_id in selected_user_ids:
                extra_edges.add((source_id, target_id, relation))
                relation_counts[relation] += 1
        else:
            canonical = canonical_tweet_relation(source_id, target_id, relation)
            if canonical is not None:
                canon_source, canon_target, canon_relation = canonical
                if canon_relation in {"retweet", "quote", "reply"}:
                    if canon_source in kept_tweet_ids and canon_target in kept_tweet_ids:
                        extra_edges.add(canonical)
                        relation_counts[canon_relation] += 1
                elif canon_relation == "mention":
                    if canon_source in kept_tweet_ids and canon_target in selected_user_ids:
                        extra_edges.add(canonical)
                        relation_counts[canon_relation] += 1
    return extra_edges, dict(relation_counts)


def _build_nested_label_counts(
    label_rows: list[dict[str, str]],
    split_rows: list[dict[str, str]],
) -> dict[str, dict[str, int]]:
    split_by_user = {row["id"]: row["split"] for row in split_rows}
    nested: dict[str, Counter[str]] = defaultdict(Counter)
    for row in label_rows:
        split = split_by_user.get(row["id"], "missing")
        nested[split][row["label"]] += 1
    return {split: dict(counter) for split, counter in sorted(nested.items())}


def _group_rows(rows: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(key, "missing"))].append(row)
    return dict(grouped)


def _ensure_output_root_ready(output_root: Path) -> None:
    if not output_root.exists():
        return
    for filename in USER_EXPORT_FILENAMES:
        if (output_root / filename).exists():
            raise FileExistsError(
                f"Output root already contains exported sample files: {output_root}. "
                "Use a new directory or remove the existing files first."
            )

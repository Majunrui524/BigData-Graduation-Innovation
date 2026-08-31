"""Export the sampled subset and supporting reports."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from . import config
from .normalize import normalize_label, normalize_split, normalize_user_id
from .readers import iter_json_records, resolve_user_path, write_csv, write_json, write_jsonl


def collect_user_records(data_root: Path, target_user_ids: set[str]) -> dict[str, dict[str, Any]]:
    """Fetch user records for the sampled user ids."""

    records: dict[str, dict[str, Any]] = {}
    if not target_user_ids:
        return records
    remaining = set(target_user_ids)
    for record in iter_json_records(resolve_user_path(data_root)):
        user_id = normalize_user_id(record)
        if user_id not in remaining:
            continue
        records[user_id] = record
        remaining.discard(user_id)
        if not remaining:
            break
    return records


def export_sample_dataset(
    output_root: Path,
    *,
    preset: str,
    seed: int,
    source_data_root: Path,
    user_records: dict[str, dict[str, Any]],
    tweet_records: dict[str, dict[str, Any]],
    user_edges: set[tuple[str, str, str]],
    post_edges: set[tuple[str, str, str]],
    extra_edges: set[tuple[str, str, str]],
    split_map: dict[str, str],
    label_map: dict[str, str],
    profile_by_user_id: dict[str, dict[str, Any]],
    seed_sampling_summary: dict[str, Any],
    context_summary: dict[str, Any],
    second_pass_summary: dict[str, Any],
    post_selection_summary: dict[str, Any],
    reference_summary: dict[str, Any],
    tweet_relation_counts: dict[str, int],
    thresholds: config.SamplingThresholds,
) -> dict[str, Any]:
    """Write the standardized sample subset to disk."""

    output_root.mkdir(parents=True, exist_ok=True)
    user_items = [user_records[user_id] for user_id in sorted(user_records)]
    tweet_items = [tweet_records[tweet_id] for tweet_id in sorted(tweet_records)]
    edge_rows = _sorted_edges(user_edges | post_edges | extra_edges)
    split_rows = _build_split_rows(user_records, split_map)
    label_rows = _build_label_rows(user_records, label_map)

    write_jsonl(output_root / "user.jsonl", user_items)
    write_jsonl(output_root / "tweet_0.jsonl", tweet_items)
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
    verified_distribution = Counter(
        profile_by_user_id.get(user_id, {}).get("verified_bucket", "missing") for user_id in user_records
    )
    split_distribution = Counter(row["split"] for row in split_rows)
    label_distribution = Counter(row["label"] for row in label_rows)
    tweet_counts_per_user = Counter(source_id for source_id, _target_id, relation in edge_rows if relation == "post")

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_data_root": str(source_data_root),
        "preset": preset,
        "seed": seed,
        "sample_thresholds": {
            "max_context_mutual": thresholds.max_context_mutual,
            "max_context_follower": thresholds.max_context_follower,
            "max_context_following": thresholds.max_context_following,
            "seed_user_max_tweets": thresholds.seed_user_max_tweets,
            "context_user_max_tweets": thresholds.context_user_max_tweets,
        },
        "relation_whitelist": list(config.OUTPUT_RELATIONS),
        "seed_sampling": {
            "requested_seed_count": seed_sampling_summary.get("requested_seed_count", 0),
            "selected_seed_count": seed_sampling_summary.get("selected_seed_count", 0),
            "split_quotas": seed_sampling_summary.get("split_quotas", {}),
            "label_quotas": seed_sampling_summary.get("label_quotas", {}),
            "pool_targets": seed_sampling_summary.get("pool_targets", {}),
            "deficit": seed_sampling_summary.get("deficit", {}),
        },
        "context_expansion": context_summary,
        "second_pass": second_pass_summary,
        "post_selection": post_selection_summary,
        "reference_closure": reference_summary,
        "tweet_relations": tweet_relation_counts,
        "final_counts": {
            "users": len(user_records),
            "tweets": len(tweet_records),
            "edges": len(edge_rows),
            "seed_users": seed_sampling_summary.get("selected_seed_count", 0),
            "context_users": context_summary.get("context_user_count", 0),
        },
        "distributions": {
            "split": dict(split_distribution),
            "label": dict(label_distribution),
            "verified": dict(verified_distribution),
            "relation": dict(relation_distribution),
        },
        "tweet_budget": {
            "limit": config.TWEET_BUDGETS[preset],
            "used": len(tweet_records),
            "within_budget": len(tweet_records) <= config.TWEET_BUDGETS[preset],
        },
    }
    write_json(output_root / "sample_manifest.json", manifest)
    stats_markdown = build_sample_stats_markdown(
        manifest=manifest,
        tweet_counts_per_user=tweet_counts_per_user,
    )
    (output_root / "sample_stats.md").write_text(stats_markdown, encoding="utf-8")
    return manifest


def build_sample_stats_markdown(
    *,
    manifest: dict[str, Any],
    tweet_counts_per_user: Counter[str],
) -> str:
    """Render the sample statistics report."""

    final_counts = manifest["final_counts"]
    distributions = manifest["distributions"]
    average_tweets = mean(tweet_counts_per_user.values()) if tweet_counts_per_user else 0.0
    min_tweets = min(tweet_counts_per_user.values()) if tweet_counts_per_user else 0
    max_tweets = max(tweet_counts_per_user.values()) if tweet_counts_per_user else 0

    lines = [
        "# Sample Statistics",
        "",
        "## Overall Counts",
        f"- Users: {final_counts['users']}",
        f"- Tweets: {final_counts['tweets']}",
        f"- Edges: {final_counts['edges']}",
        f"- Seed users: {final_counts['seed_users']}",
        f"- Context users: {final_counts['context_users']}",
        "",
        "## Split Distribution",
    ]
    lines.extend(f"- {split}: {count}" for split, count in sorted(distributions["split"].items()))
    lines.extend(["", "## Label Distribution"])
    lines.extend(f"- {label}: {count}" for label, count in sorted(distributions["label"].items()))
    lines.extend(["", "## Verified Distribution"])
    lines.extend(f"- {bucket}: {count}" for bucket, count in sorted(distributions["verified"].items()))
    lines.extend(
        [
            "",
            "## Tweet Per User",
            f"- Min posts retained: {min_tweets}",
            f"- Avg posts retained: {average_tweets:.2f}",
            f"- Max posts retained: {max_tweets}",
            "",
            "## Relation Distribution",
        ]
    )
    lines.extend(f"- {relation}: {count}" for relation, count in sorted(distributions["relation"].items()))
    lines.extend(
        [
            "",
            "## Context Expansion",
            f"- Candidate users seen: {manifest['context_expansion'].get('candidate_count', 0)}",
            f"- Context users kept: {manifest['context_expansion'].get('context_user_count', 0)}",
            f"- Final user count: {manifest['context_expansion'].get('final_user_count', 0)}",
            f"- Hub cutoff (p95 degree proxy): {manifest['context_expansion'].get('hub_degree_cutoff_p95', 0)}",
            "",
            "## Budget Check",
            f"- Tweet budget limit: {manifest['tweet_budget']['limit']}",
            f"- Tweet budget used: {manifest['tweet_budget']['used']}",
            f"- Within budget: {manifest['tweet_budget']['within_budget']}",
        ]
    )
    return "\n".join(lines) + "\n"


def _sorted_edges(edges: set[tuple[str, str, str]]) -> list[tuple[str, str, str]]:
    return sorted(edges, key=lambda item: (item[2], item[0], item[1]))


def _build_split_rows(user_records: dict[str, dict[str, Any]], split_map: dict[str, str]) -> list[dict[str, str]]:
    rows = []
    for user_id in sorted(user_records):
        rows.append({"id": user_id, "split": normalize_split(split_map.get(user_id))})
    return rows


def _build_label_rows(user_records: dict[str, dict[str, Any]], label_map: dict[str, str]) -> list[dict[str, str]]:
    rows = []
    for user_id in sorted(user_records):
        rows.append({"id": user_id, "label": normalize_label(label_map.get(user_id))})
    return rows

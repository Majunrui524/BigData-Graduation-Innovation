"""One-hop local graph expansion around selected seeds."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from .config import SamplingThresholds
from .normalize import canonical_follow_edge, percentile, stable_hexdigest
from .readers import read_csv_rows


def expand_context_users(
    edge_path: Path,
    *,
    seed_user_ids: list[str],
    profile_by_user_id: dict[str, dict[str, Any]],
    thresholds: SamplingThresholds,
    seed: int,
) -> tuple[set[str], dict[str, Any]]:
    """Expand one-hop context users around seeds using canonical follow edges."""

    seed_set = set(seed_user_ids)
    per_seed_candidate_flags: dict[str, dict[str, dict[str, bool]]] = defaultdict(
        lambda: defaultdict(lambda: {"seed_follows": False, "follows_seed": False})
    )
    candidate_seed_links: dict[str, set[str]] = defaultdict(set)

    for row in read_csv_rows(edge_path):
        canonical = canonical_follow_edge(
            row.get("source_id", ""),
            row.get("target_id", ""),
            row.get("relation", ""),
        )
        if canonical is None:
            continue
        follower_id, followed_id, _ = canonical
        if follower_id in seed_set and followed_id not in seed_set:
            flags = per_seed_candidate_flags[follower_id][followed_id]
            flags["seed_follows"] = True
            candidate_seed_links[followed_id].add(follower_id)
        if followed_id in seed_set and follower_id not in seed_set:
            flags = per_seed_candidate_flags[followed_id][follower_id]
            flags["follows_seed"] = True
            candidate_seed_links[follower_id].add(followed_id)

    candidate_degree_values = [
        int(profile_by_user_id.get(candidate_id, {}).get("degree_proxy", 0))
        for candidate_id in candidate_seed_links
    ]
    hub_cutoff = percentile(candidate_degree_values, 0.95) if candidate_degree_values else 0

    selected_context: set[str] = set()
    category_totals = {"mutual": 0, "follower": 0, "following": 0}

    for seed_user_id in seed_user_ids:
        buckets = {"mutual": [], "follower": [], "following": []}
        for candidate_id, flags in per_seed_candidate_flags.get(seed_user_id, {}).items():
            degree_proxy = int(profile_by_user_id.get(candidate_id, {}).get("degree_proxy", 0))
            if len(candidate_seed_links[candidate_id]) == 1 and degree_proxy > hub_cutoff:
                continue
            category = _classify_candidate(flags)
            if category is None:
                continue
            buckets[category].append(
                (
                    -_connection_count(flags),
                    degree_proxy,
                    stable_hexdigest(seed, "context", seed_user_id, category, candidate_id),
                    candidate_id,
                )
            )

        limits = {
            "mutual": thresholds.max_context_mutual,
            "follower": thresholds.max_context_follower,
            "following": thresholds.max_context_following,
        }
        for category, limit in limits.items():
            ordered = sorted(buckets[category])
            chosen = [candidate_id for *_unused, candidate_id in ordered[:limit]]
            selected_context.update(chosen)
            category_totals[category] += len(chosen)

    final_users = seed_set | selected_context
    summary = {
        "context_user_count": len(selected_context),
        "final_user_count": len(final_users),
        "hub_degree_cutoff_p95": hub_cutoff,
        "candidate_count": len(candidate_seed_links),
        "category_totals": category_totals,
    }
    return final_users, summary


def _classify_candidate(flags: dict[str, bool]) -> str | None:
    if flags["seed_follows"] and flags["follows_seed"]:
        return "mutual"
    if flags["follows_seed"]:
        return "follower"
    if flags["seed_follows"]:
        return "following"
    return None


def _connection_count(flags: dict[str, bool]) -> int:
    return int(flags["seed_follows"]) + int(flags["follows_seed"])

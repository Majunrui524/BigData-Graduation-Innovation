"""Seed-user selection and quota allocation."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

from . import config
from .normalize import assign_bucket, quantile_edges, stable_hexdigest


def allocate_largest_remainder(total: int, weights: dict[str, float | int]) -> dict[str, int]:
    """Allocate integer quotas with the largest-remainder method."""

    total = max(int(total), 0)
    keys = list(weights)
    if total == 0 or not keys:
        return {key: 0 for key in keys}
    positive = {key: float(max(weights[key], 0)) for key in keys}
    weight_sum = sum(positive.values())
    if weight_sum <= 0:
        return {key: 0 for key in keys}
    raw = {key: total * (value / weight_sum) for key, value in positive.items()}
    base = {key: math.floor(value) for key, value in raw.items()}
    remainder = total - sum(base.values())
    ordered = sorted(
        keys,
        key=lambda key: (
            -(raw[key] - base[key]),
            str(key),
        ),
    )
    for key in ordered[:remainder]:
        base[key] += 1
    return base


def select_seed_users(
    profile_rows: list[dict[str, Any]],
    *,
    preset: str,
    seed: int,
) -> tuple[list[str], dict[str, Any]]:
    """Select deterministic seed users according to the configured rules."""

    target_size = config.preset_size(preset)
    rows = [dict(row) for row in profile_rows if row.get("split") != "missing" and row.get("label") != "missing"]
    split_groups = _group_rows(rows, "split")
    split_quotas = allocate_largest_remainder(
        target_size,
        {split: len(group_rows) for split, group_rows in split_groups.items()},
    )

    selected_ids: list[str] = []
    selected_id_set: set[str] = set()
    summary: dict[str, Any] = {
        "preset": preset,
        "requested_seed_count": target_size,
        "split_quotas": split_quotas,
        "label_quotas": {},
        "pool_targets": {},
        "deficit": {},
    }

    for split, split_rows in sorted(split_groups.items()):
        label_groups = _group_rows(split_rows, "label")
        label_quota = allocate_largest_remainder(
            split_quotas.get(split, 0),
            {label: len(group_rows) for label, group_rows in label_groups.items()},
        )
        summary["label_quotas"][split] = label_quota
        summary["pool_targets"][split] = {}
        summary["deficit"][split] = {}
        for label, rows_in_group in sorted(label_groups.items()):
            target = label_quota.get(label, 0)
            selected, group_summary = _select_split_label_rows(
                rows_in_group,
                target=target,
                seed=seed,
                split=split,
                label=label,
            )
            summary["pool_targets"][split][label] = group_summary["pool_targets"]
            summary["deficit"][split][label] = group_summary["deficit"]
            for user_id in selected:
                if user_id not in selected_id_set:
                    selected_id_set.add(user_id)
                    selected_ids.append(user_id)
    ordered_ids = sorted(selected_ids, key=lambda user_id: stable_hexdigest(seed, "seed", user_id))
    summary["selected_seed_count"] = len(ordered_ids)
    summary["selected_seed_ids"] = ordered_ids
    return ordered_ids, summary


def _select_split_label_rows(
    rows: list[dict[str, Any]],
    *,
    target: int,
    seed: int,
    split: str,
    label: str,
) -> tuple[list[str], dict[str, Any]]:
    rows = [dict(row) for row in rows]
    if target <= 0 or not rows:
        return [], {"pool_targets": {"primary": 0, "sparse": 0}, "deficit": 0}

    _attach_strata(rows)
    primary_rows = [row for row in rows if row.get("primary_pool")]
    sparse_rows = [row for row in rows if row.get("sparse_pool")]
    pool_targets = allocate_largest_remainder(
        target,
        {
            "primary": config.POOL_SPLIT["primary"],
            "sparse": config.POOL_SPLIT["sparse"],
        },
    )

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()

    for pool_name, pool_rows in (("primary", primary_rows), ("sparse", sparse_rows)):
        chosen = _select_from_pool(
            pool_rows,
            target=pool_targets[pool_name],
            seed=seed,
            namespace=f"{split}:{label}:{pool_name}",
        )
        for row in chosen:
            if row["user_id"] not in selected_ids:
                selected.append(row)
                selected_ids.add(row["user_id"])

    if len(selected) < target:
        fallback = sorted(
            (row for row in rows if row["user_id"] not in selected_ids),
            key=lambda row: stable_hexdigest(seed, "fallback", split, label, row["user_id"]),
        )
        for row in fallback[: target - len(selected)]:
            selected.append(row)
            selected_ids.add(row["user_id"])

    ordered_ids = [
        row["user_id"]
        for row in sorted(selected, key=lambda row: stable_hexdigest(seed, "selected", split, label, row["user_id"]))
    ]
    return ordered_ids, {
        "pool_targets": pool_targets,
        "deficit": max(target - len(ordered_ids), 0),
    }


def _attach_strata(rows: list[dict[str, Any]]) -> None:
    tweet_edges = quantile_edges([int(row["tweet_count_hint"]) for row in rows], buckets=4)
    follower_edges = quantile_edges([int(row["followers_count"]) for row in rows], buckets=4)
    following_edges = quantile_edges([int(row["following_count"]) for row in rows], buckets=4)
    for row in rows:
        row["stratum"] = "|".join(
            (
                assign_bucket(int(row["tweet_count_hint"]), tweet_edges),
                assign_bucket(int(row["followers_count"]), follower_edges),
                assign_bucket(int(row["following_count"]), following_edges),
                row.get("verified_bucket", "missing"),
            )
        )


def _select_from_pool(
    rows: list[dict[str, Any]],
    *,
    target: int,
    seed: int,
    namespace: str,
) -> list[dict[str, Any]]:
    if target <= 0 or not rows:
        return []
    grouped = _group_rows(rows, "stratum")
    quotas = allocate_largest_remainder(target, {stratum: len(group_rows) for stratum, group_rows in grouped.items()})
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    remainder_rows: list[dict[str, Any]] = []

    for stratum, stratum_rows in sorted(grouped.items()):
        ordered = sorted(
            stratum_rows,
            key=lambda row: stable_hexdigest(seed, namespace, stratum, row["user_id"]),
        )
        quota = quotas.get(stratum, 0)
        chosen = ordered[:quota]
        selected.extend(chosen)
        selected_ids.update(row["user_id"] for row in chosen)
        remainder_rows.extend(ordered[quota:])

    if len(selected) < target:
        ordered_remainder = sorted(
            (row for row in remainder_rows if row["user_id"] not in selected_ids),
            key=lambda row: stable_hexdigest(seed, namespace, "remainder", row["user_id"]),
        )
        selected.extend(ordered_remainder[: target - len(selected)])
    return selected[:target]


def _group_rows(rows: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(key, "missing"))].append(row)
    return dict(grouped)

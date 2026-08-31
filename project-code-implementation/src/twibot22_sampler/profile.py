"""Build the user-level sampling profile."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from . import config
from .normalize import (
    normalize_followers_count,
    normalize_following_count,
    normalize_label,
    normalize_split,
    normalize_tweet_count_hint,
    normalize_user_id,
    normalize_verified_bucket,
)
from .readers import (
    iter_json_records,
    read_label_map,
    read_split_map,
    resolve_label_path,
    resolve_split_path,
    resolve_user_path,
    write_csv,
)


def build_user_profile(data_root: Path, work_root: Path) -> Path:
    """Generate a CSV profile used by the sampler."""

    user_path = resolve_user_path(data_root)
    split_map = read_split_map(resolve_split_path(data_root))
    label_map = read_label_map(resolve_label_path(data_root))
    output_path = work_root / "profile" / "users_profile.csv"

    rows = (
        _profile_row(record, split_map=split_map, label_map=label_map)
        for record in iter_json_records(user_path)
    )
    write_csv(output_path, config.USER_PROFILE_COLUMNS, rows)
    return output_path


def _profile_row(
    record: dict[str, Any],
    *,
    split_map: dict[str, str],
    label_map: dict[str, str],
) -> dict[str, Any]:
    user_id = normalize_user_id(record)
    followers_count = normalize_followers_count(record)
    following_count = normalize_following_count(record)
    tweet_count_hint = normalize_tweet_count_hint(record)
    degree_proxy = max(followers_count, 0) + max(following_count, 0)
    primary_pool = tweet_count_hint >= config.PRIMARY_POOL_MIN_TWEETS
    sparse_pool = config.SPARSE_POOL_MIN_TWEETS <= tweet_count_hint < config.PRIMARY_POOL_MIN_TWEETS
    return {
        "user_id": user_id,
        "split": normalize_split(split_map.get(user_id)),
        "label": normalize_label(label_map.get(user_id)),
        "verified_bucket": normalize_verified_bucket(record),
        "followers_count": followers_count,
        "following_count": following_count,
        "tweet_count_hint": tweet_count_hint,
        "degree_proxy": degree_proxy,
        "primary_pool": "1" if primary_pool else "0",
        "sparse_pool": "1" if sparse_pool else "0",
    }


def load_profile_rows(profile_path: Path) -> list[dict[str, Any]]:
    """Load the generated profile CSV into memory for quota calculations."""

    rows: list[dict[str, Any]] = []
    with profile_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                {
                    "user_id": str(row["user_id"]),
                    "split": normalize_split(row.get("split")),
                    "label": normalize_label(row.get("label")),
                    "verified_bucket": row.get("verified_bucket", "missing") or "missing",
                    "followers_count": int(row.get("followers_count", 0)),
                    "following_count": int(row.get("following_count", 0)),
                    "tweet_count_hint": int(row.get("tweet_count_hint", 0)),
                    "degree_proxy": int(row.get("degree_proxy", 0)),
                    "primary_pool": row.get("primary_pool", "0") == "1",
                    "sparse_pool": row.get("sparse_pool", "0") == "1",
                }
            )
    return rows

"""Normalization helpers shared across the pipeline."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import urlparse
from typing import Any, Iterable, Sequence

from . import config


def stable_hexdigest(seed: int, *parts: object) -> str:
    """Return a stable digest for deterministic ranking across platforms."""

    payload = "|".join([str(seed), *(str(part) for part in parts)])
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def to_int(value: Any, default: int = 0) -> int:
    """Coerce a value into an integer without raising."""

    if value is None:
        return default
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return default
        try:
            return int(float(stripped.replace(",", "")))
        except ValueError:
            return default
    return default


def get_nested(mapping: dict[str, Any], *path: str) -> Any:
    """Safely traverse nested dictionaries."""

    current: Any = mapping
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def first_present(mapping: dict[str, Any], paths: Sequence[Sequence[str]]) -> Any:
    """Return the first non-None value for the provided path candidates."""

    for path in paths:
        value = get_nested(mapping, *path)
        if value is not None:
            return value
    return None


def normalize_user_id(record: dict[str, Any], fallback: str | None = None) -> str:
    """Extract a string user id."""

    value = first_present(
        record,
        (
            ("id",),
            ("user_id",),
            ("id_str",),
            ("rest_id",),
        ),
    )
    value = fallback if value is None else value
    if value is None:
        raise ValueError("Record does not contain a user id")
    return str(value)


def canonical_user_id(value: Any) -> str:
    """Normalize user identifiers to the exported sample form `u<digits>` when possible."""

    if value in (None, ""):
        return ""
    text = str(value).strip()
    if not text:
        return ""
    lowered = text.lower()
    if lowered.startswith("u") and text[1:].isdigit():
        return f"u{text[1:]}"
    if text.isdigit():
        return f"u{text}"
    return text


def normalize_followers_count(record: dict[str, Any]) -> int:
    """Extract follower count from supported aliases."""

    value = first_present(
        record,
        (
            ("followers_count",),
            ("public_metrics", "followers_count"),
        ),
    )
    return max(to_int(value, 0), 0)


def normalize_following_count(record: dict[str, Any]) -> int:
    """Extract following count from supported aliases."""

    value = first_present(
        record,
        (
            ("friends_count",),
            ("following_count",),
            ("public_metrics", "following_count"),
        ),
    )
    return max(to_int(value, 0), 0)


def normalize_tweet_count_hint(record: dict[str, Any]) -> int:
    """Extract user-level tweet count hint from supported aliases."""

    value = first_present(
        record,
        (
            ("statuses_count",),
            ("tweet_count",),
            ("public_metrics", "tweet_count"),
        ),
    )
    return max(to_int(value, 0), 0)


def normalize_verified_bucket(record: dict[str, Any]) -> str:
    """Reduce verification into true/false/missing."""

    value = first_present(record, (("verified",),))
    if value is None:
        return "missing"
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return "true"
        if lowered in {"false", "0", "no"}:
            return "false"
        return "missing"
    return "true" if bool(value) else "false"


def normalize_label(value: Any) -> str:
    """Normalize label values to bot/human/missing."""

    if value is None:
        return "missing"
    lowered = str(value).strip().lower()
    if lowered in {"bot", "1", "true"}:
        return "bot"
    if lowered in {"human", "0", "false"}:
        return "human"
    return lowered or "missing"


def normalize_split(value: Any) -> str:
    """Normalize split values to train/valid/test or missing."""

    if value is None:
        return "missing"
    lowered = str(value).strip().lower()
    alias = {
        "dev": "valid",
        "val": "valid",
        "validation": "valid",
    }
    return alias.get(lowered, lowered or "missing")


def canonical_follow_edge(
    source_id: str, target_id: str, relation: Any
) -> tuple[str, str, str] | None:
    """Return follower -> followed for supported user-user relations."""

    rel = str(relation).strip().lower()
    if rel not in config.USER_USER_RELATION_ALIASES:
        return None
    if rel == "following":
        return str(source_id), str(target_id), "following"
    return str(target_id), str(source_id), "following"


def canonical_tweet_relation(
    source_id: str, target_id: str, relation: Any
) -> tuple[str, str, str] | None:
    """Return canonical tweet-edge tuples for supported relations."""

    rel = str(relation).strip().lower()
    if rel in config.POST_RELATION_ALIASES:
        return str(source_id), str(target_id), "post"
    if rel in config.TWEET_TWEET_RELATION_ALIASES:
        return str(source_id), str(target_id), config.TWEET_TWEET_RELATION_ALIASES[rel]
    if rel in config.MENTION_RELATION_ALIASES:
        return str(source_id), str(target_id), "mention"
    return None


def parse_timestamp(value: Any) -> datetime | None:
    """Parse mixed Twitter-style timestamp formats."""

    if value in (None, ""):
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = parsedate_to_datetime(text)
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except (TypeError, ValueError, IndexError):
        pass
    cleaned = text.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(cleaned)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def percentile(values: Iterable[int], ratio: float) -> int:
    """Compute a simple percentile with rank interpolation."""

    ordered = sorted(int(v) for v in values)
    if not ordered:
        return 0
    if len(ordered) == 1:
        return ordered[0]
    clamped = min(max(ratio, 0.0), 1.0)
    position = clamped * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return int(round(ordered[lower] + (ordered[upper] - ordered[lower]) * fraction))


def quantile_edges(values: Sequence[int], buckets: int = 4) -> list[int]:
    """Return monotonic quantile cut points for bucket assignment."""

    if not values:
        return []
    return [percentile(values, idx / buckets) for idx in range(1, buckets)]


def assign_bucket(value: int | None, edges: Sequence[int]) -> str:
    """Map a value to a quantile bucket label."""

    if value is None:
        return "missing"
    for index, edge in enumerate(edges):
        if value <= edge:
            return f"q{index + 1}"
    return f"q{len(edges) + 1}"


def select_evenly(items: Sequence[Any], limit: int) -> list[Any]:
    """Choose limit items evenly across an ordered sequence."""

    if limit <= 0 or not items:
        return []
    if len(items) <= limit:
        return list(items)
    if limit == 1:
        return [items[0]]
    indices = []
    last_index = len(items) - 1
    for slot in range(limit):
        index = round(slot * last_index / (limit - 1))
        indices.append(index)
    return [items[index] for index in indices]


def extract_referenced_tweet_ids(record: dict[str, Any]) -> set[str]:
    """Extract one-hop referenced tweet ids from tweet records."""

    referenced = record.get("referenced_tweets")
    result: set[str] = set()
    if isinstance(referenced, list):
        for item in referenced:
            if isinstance(item, dict) and item.get("id") is not None:
                result.add(str(item["id"]))
            elif item is not None:
                result.add(str(item))
    elif isinstance(referenced, dict):
        for key in ("id", "tweet_id", "target_id"):
            value = referenced.get(key)
            if value is not None:
                result.add(str(value))
    return result


def extract_url_entries(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract URL entries from a tweet/user entities object."""

    entities = record.get("entities")
    if not isinstance(entities, dict):
        return []
    urls = entities.get("urls")
    if not isinstance(urls, list):
        return []
    return [item for item in urls if isinstance(item, dict)]


def has_external_url(record: dict[str, Any]) -> bool:
    """Return whether a record contains an external non-Twitter URL."""

    internal_domains = {
        "twitter.com",
        "www.twitter.com",
        "x.com",
        "www.x.com",
        "pic.twitter.com",
        "t.co",
    }
    for item in extract_url_entries(record):
        candidate = item.get("expanded_url") or item.get("display_url") or item.get("url")
        if not candidate:
            continue
        parsed = urlparse(str(candidate))
        host = parsed.netloc.lower()
        if not host and "://" not in str(candidate):
            host = str(candidate).split("/")[0].lower()
        if host and host not in internal_domains:
            return True
    return False

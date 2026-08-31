"""LLM-backed post-type classification for sampled tweets."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any, Iterator

from .derived_common import read_processed_ids, select_tweets_for_derived_tasks
from .llm_client import OpenAICompatibleClient
from .normalize import canonical_user_id, extract_url_entries, has_external_url
from .readers import read_jsonl_records, write_json, write_jsonl

POST_TYPE_PROMPT_VERSION = "post-type-v1"
POST_TYPE_COARSE_TYPES = ("original", "retweet", "comment_reply", "link_share")
POST_TYPE_DETAIL_TYPES = ("original", "retweet", "reply", "quote_comment", "link_share", "other")

POST_TYPE_SYSTEM_PROMPT = """You classify tweets into a small posting-type taxonomy.
Return JSON only.
Allowed coarse_type values: original, retweet, comment_reply, link_share.
Allowed detail_type values: original, retweet, reply, quote_comment, link_share, other.
Use the metadata provided in the prompt together with tweet text."""


def run_post_type_classification(
    sample_root: Path,
    output_root: Path,
    *,
    mode: str,
    client: OpenAICompatibleClient | None,
    seed: int = 42,
    per_user_limit: int = 20,
    min_user_tweets: int = 1,
    max_users: int | None = None,
    max_tweets: int | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Classify tweet post types and aggregate user-level distributions."""

    output_root.mkdir(parents=True, exist_ok=True)
    tweet_output_path = output_root / "tweet_post_types.jsonl"
    error_output_path = output_root / "tweet_post_type_errors.jsonl"
    user_output_path = output_root / "user_post_type_distribution.jsonl"

    if overwrite:
        for path in (tweet_output_path, error_output_path, user_output_path):
            if path.exists():
                path.unlink()

    selected_tweets, selection_summary = select_tweets_for_derived_tasks(
        sample_root,
        per_user_limit=per_user_limit,
        min_user_tweets=min_user_tweets,
        max_users=max_users,
        max_tweets=max_tweets,
        seed=seed,
        require_text=True,
    )
    already_processed = read_processed_ids(tweet_output_path)

    heuristic_count = 0
    llm_count = 0
    skipped_count = 0
    error_count = 0
    llm_pending_records = []
    with tweet_output_path.open("a", encoding="utf-8") as success_handle, error_output_path.open(
        "a", encoding="utf-8"
    ) as error_handle:
        for record in selected_tweets:
            tweet_id = str(record.get("id"))
            author_id = canonical_user_id(record.get("author_id"))
            if tweet_id in already_processed:
                skipped_count += 1
                continue
            try:
                heuristic = classify_post_type_heuristic(record) if mode in {"heuristic", "hybrid"} else None
                if heuristic is not None:
                    normalized = heuristic
                    source = "heuristic"
                    heuristic_count += 1
                else:
                    if mode == "heuristic":
                        normalized = {
                            "coarse_type": "original",
                            "detail_type": "other",
                            "confidence": None,
                            "reason": "heuristic_fallback",
                        }
                        source = "heuristic_fallback"
                        heuristic_count += 1
                    else:
                        llm_pending_records.append(record)
                        continue
                payload = {
                    "tweet_id": tweet_id,
                    "author_id": author_id,
                    "created_at": record.get("created_at"),
                    "lang": record.get("lang"),
                    "prompt_version": POST_TYPE_PROMPT_VERSION,
                    "mode": mode,
                    "model": client.settings.model if client is not None else None,
                    "source": source,
                    "coarse_type": normalized["coarse_type"],
                    "detail_type": normalized["detail_type"],
                    "confidence": normalized["confidence"],
                    "reason": normalized["reason"],
                }
                success_handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True))
                success_handle.write("\n")
            except Exception as exc:  # pragma: no cover - network/error path
                error_payload = {
                    "tweet_id": tweet_id,
                    "author_id": author_id,
                    "created_at": record.get("created_at"),
                    "lang": record.get("lang"),
                    "prompt_version": POST_TYPE_PROMPT_VERSION,
                    "mode": mode,
                    "model": client.settings.model if client is not None else None,
                    "status": "error",
                    "error": str(exc),
                }
                error_handle.write(json.dumps(error_payload, ensure_ascii=False, sort_keys=True))
                error_handle.write("\n")
                error_count += 1

        if llm_pending_records and client is None:
            raise ValueError("LLM client is required for llm or hybrid mode when heuristics are insufficient")

        for outcome in _iter_post_type_outcomes(llm_pending_records, mode, client):
            if outcome["status"] == "ok":
                payload = outcome["payload"]
                success_handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True))
                success_handle.write("\n")
                llm_count += 1
            else:
                error_payload = outcome["payload"]
                error_handle.write(json.dumps(error_payload, ensure_ascii=False, sort_keys=True))
                error_handle.write("\n")
                error_count += 1

    user_distribution = build_user_post_type_distribution(tweet_output_path)
    write_jsonl(user_output_path, user_distribution)
    manifest = {
        "sample_root": str(sample_root),
        "output_root": str(output_root),
        "prompt_version": POST_TYPE_PROMPT_VERSION,
        "mode": mode,
        "model": client.settings.model if client is not None else None,
        "concurrency": client.settings.concurrency if client is not None else 1,
        "requests_per_minute": client.settings.requests_per_minute if client is not None else None,
        "selection": selection_summary,
        "heuristic_count": heuristic_count,
        "llm_count": llm_count,
        "skipped_count": skipped_count,
        "error_count": error_count,
        "user_distribution_count": len(user_distribution),
        "files": {
            "tweet_post_types": str(tweet_output_path),
            "tweet_post_type_errors": str(error_output_path),
            "user_post_type_distribution": str(user_output_path),
        },
    }
    write_json(output_root / "run_manifest.json", manifest)
    return manifest


def _iter_post_type_outcomes(
    records: list[dict[str, Any]],
    mode: str,
    client: OpenAICompatibleClient | None,
) -> Iterator[dict[str, Any]]:
    if not records:
        return
    assert client is not None
    if client.settings.concurrency <= 1:
        for record in records:
            yield _run_single_post_type_record(record, mode, client)
        return
    with ThreadPoolExecutor(max_workers=client.settings.concurrency) as executor:
        yield from executor.map(partial(_run_single_post_type_record, mode=mode, client=client), records)


def _run_single_post_type_record(
    record: dict[str, Any],
    mode: str,
    client: OpenAICompatibleClient,
) -> dict[str, Any]:
    tweet_id = str(record.get("id"))
    author_id = canonical_user_id(record.get("author_id"))
    try:
        response = client.chat_json(
            system_prompt=POST_TYPE_SYSTEM_PROMPT,
            user_prompt=build_post_type_user_prompt(record),
            max_tokens=350,
        )
        normalized = normalize_post_type_response(response["parsed"])
        return {
            "status": "ok",
            "payload": {
                "tweet_id": tweet_id,
                "author_id": author_id,
                "created_at": record.get("created_at"),
                "lang": record.get("lang"),
                "prompt_version": POST_TYPE_PROMPT_VERSION,
                "mode": mode,
                "model": client.settings.model,
                "source": "llm",
                "coarse_type": normalized["coarse_type"],
                "detail_type": normalized["detail_type"],
                "confidence": normalized["confidence"],
                "reason": normalized["reason"],
            },
        }
    except Exception as exc:  # pragma: no cover - network/error path
        return {
            "status": "error",
            "payload": {
                "tweet_id": tweet_id,
                "author_id": author_id,
                "created_at": record.get("created_at"),
                "lang": record.get("lang"),
                "prompt_version": POST_TYPE_PROMPT_VERSION,
                "mode": mode,
                "model": client.settings.model,
                "status": "error",
                "error": str(exc),
            },
        }


def classify_post_type_heuristic(record: dict[str, Any]) -> dict[str, Any] | None:
    """Return a confident rule-based classification when the metadata is unambiguous."""

    text = str(record.get("text") or "").strip()
    referenced = record.get("referenced_tweets")
    referenced_types = set()
    if isinstance(referenced, list):
        referenced_types = {
            str(item.get("type") or "").strip().lower() for item in referenced if isinstance(item, dict)
        }
    if "retweeted" in referenced_types or text.startswith("RT @"):
        return {
            "coarse_type": "retweet",
            "detail_type": "retweet",
            "confidence": 0.99,
            "reason": "retweeted_metadata_or_rt_prefix",
        }
    if "replied_to" in referenced_types or record.get("in_reply_to_user_id") not in (None, ""):
        return {
            "coarse_type": "comment_reply",
            "detail_type": "reply",
            "confidence": 0.99,
            "reason": "reply_metadata",
        }
    if "quoted" in referenced_types:
        return {
            "coarse_type": "comment_reply",
            "detail_type": "quote_comment",
            "confidence": 0.95,
            "reason": "quoted_metadata",
        }
    if has_external_url(record):
        return {
            "coarse_type": "link_share",
            "detail_type": "link_share",
            "confidence": 0.9,
            "reason": "external_url_present",
        }
    return None


def build_post_type_user_prompt(record: dict[str, Any]) -> str:
    """Build the user prompt for post-type classification."""

    text = str(record.get("text") or "").strip()
    referenced = record.get("referenced_tweets") or []
    referenced_types = []
    if isinstance(referenced, list):
        referenced_types = [str(item.get("type") or "") for item in referenced if isinstance(item, dict)]
    return (
        "Classify this tweet into the posting-type taxonomy and return JSON with this schema:\n"
        '{"coarse_type":"original|retweet|comment_reply|link_share","detail_type":"original|retweet|reply|quote_comment|link_share|other","confidence":0.0,"reason":"..."}\n'
        f"Has external URL: {has_external_url(record)}\n"
        f"Referenced tweet types: {referenced_types}\n"
        f"In reply to user: {record.get('in_reply_to_user_id') is not None}\n"
        f"URL count: {len(extract_url_entries(record))}\n"
        f"Tweet:\n{text}"
    )


def normalize_post_type_response(payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize loosely structured LLM JSON into stable coarse/detail labels."""

    coarse = str(payload.get("coarse_type") or payload.get("label") or "").strip().lower()
    detail = str(payload.get("detail_type") or payload.get("subtype") or "").strip().lower()
    if coarse not in POST_TYPE_COARSE_TYPES:
        coarse = "original"
    if detail not in POST_TYPE_DETAIL_TYPES:
        detail = "original" if coarse == "original" else "other"
    confidence = payload.get("confidence")
    try:
        confidence_value = float(confidence) if confidence is not None else None
    except (TypeError, ValueError):
        confidence_value = None
    reason = str(payload.get("reason") or "").strip()
    return {
        "coarse_type": coarse,
        "detail_type": detail,
        "confidence": confidence_value,
        "reason": reason,
    }


def build_user_post_type_distribution(tweet_output_path: Path) -> list[dict[str, Any]]:
    """Aggregate tweet-level classifications into user-level distributions."""

    per_user: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "author_id": "",
            "tweet_count": 0,
            "coarse_counts": Counter(),
            "detail_counts": Counter(),
        }
    )
    for record in read_jsonl_records(tweet_output_path):
        author_id = canonical_user_id(record.get("author_id"))
        if not author_id:
            continue
        coarse = str(record.get("coarse_type") or "")
        detail = str(record.get("detail_type") or "")
        row = per_user[author_id]
        row["author_id"] = author_id
        row["tweet_count"] += 1
        if coarse:
            row["coarse_counts"][coarse] += 1
        if detail:
            row["detail_counts"][detail] += 1

    output = []
    for author_id, row in sorted(per_user.items()):
        coarse_counts = {label: row["coarse_counts"].get(label, 0) for label in POST_TYPE_COARSE_TYPES}
        detail_counts = {label: row["detail_counts"].get(label, 0) for label in POST_TYPE_DETAIL_TYPES}
        tweet_count = row["tweet_count"]
        coarse_distribution = {
            label: round(coarse_counts[label] / tweet_count, 6) if tweet_count else 0.0 for label in POST_TYPE_COARSE_TYPES
        }
        output.append(
            {
                "author_id": author_id,
                "tweet_count": tweet_count,
                "coarse_counts": coarse_counts,
                "coarse_distribution": coarse_distribution,
                "detail_counts": detail_counts,
            }
        )
    return output

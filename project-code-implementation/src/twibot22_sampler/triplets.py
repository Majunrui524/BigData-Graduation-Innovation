"""LLM-driven triplet compression for sampled tweets."""

from __future__ import annotations

import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any, Iterator

from .derived_common import read_processed_ids, select_tweets_for_derived_tasks
from .normalize import canonical_user_id
from .llm_client import OpenAICompatibleClient
from .readers import read_jsonl_records, write_json, write_jsonl

TRIPLET_PROMPT_VERSION = "triplet-v1"

TRIPLET_SYSTEM_PROMPT = """You compress tweets into concise semantic triplets.
Return JSON only.
Extract the core semantic content without hashtags, tracking links, or filler.
If the tweet has no clear semantic relation, return an empty triplets list but still provide a short compressed_text."""


def run_triplet_extraction(
    sample_root: Path,
    output_root: Path,
    *,
    client: OpenAICompatibleClient,
    seed: int = 42,
    per_user_limit: int = 20,
    min_user_tweets: int = 8,
    max_users: int | None = None,
    max_tweets: int | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Extract LLM triplets and aggregate them to user-level documents."""

    output_root.mkdir(parents=True, exist_ok=True)
    tweet_output_path = output_root / "tweet_triplets.jsonl"
    error_output_path = output_root / "tweet_triplet_errors.jsonl"
    user_output_path = output_root / "user_triplet_documents.jsonl"

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

    processed_count = 0
    skipped_count = 0
    error_count = 0
    pending_records = []
    with tweet_output_path.open("a", encoding="utf-8") as success_handle, error_output_path.open(
        "a", encoding="utf-8"
    ) as error_handle:
        for record in selected_tweets:
            tweet_id = str(record.get("id"))
            if tweet_id in already_processed:
                skipped_count += 1
                continue
            pending_records.append(record)

        for outcome in _iter_triplet_outcomes(pending_records, client):
            if outcome["status"] == "ok":
                payload = outcome["payload"]
                success_handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True))
                success_handle.write("\n")
                processed_count += 1
            else:
                error_payload = outcome["payload"]
                error_handle.write(json.dumps(error_payload, ensure_ascii=False, sort_keys=True))
                error_handle.write("\n")
                error_count += 1

    user_documents = build_user_triplet_documents(tweet_output_path)
    write_jsonl(user_output_path, user_documents)
    manifest = {
        "sample_root": str(sample_root),
        "output_root": str(output_root),
        "prompt_version": TRIPLET_PROMPT_VERSION,
        "model": client.settings.model,
        "selection": selection_summary,
        "concurrency": client.settings.concurrency,
        "requests_per_minute": client.settings.requests_per_minute,
        "processed_count": processed_count,
        "skipped_count": skipped_count,
        "error_count": error_count,
        "user_document_count": len(user_documents),
        "files": {
            "tweet_triplets": str(tweet_output_path),
            "tweet_triplet_errors": str(error_output_path),
            "user_triplet_documents": str(user_output_path),
        },
    }
    write_json(output_root / "run_manifest.json", manifest)
    return manifest


def _iter_triplet_outcomes(
    records: list[dict[str, Any]],
    client: OpenAICompatibleClient,
) -> Iterator[dict[str, Any]]:
    if not records:
        return
    if client.settings.concurrency <= 1:
        for record in records:
            yield _run_single_triplet_record(record, client)
        return
    with ThreadPoolExecutor(max_workers=client.settings.concurrency) as executor:
        yield from executor.map(partial(_run_single_triplet_record, client=client), records)


def _run_single_triplet_record(record: dict[str, Any], client: OpenAICompatibleClient) -> dict[str, Any]:
    tweet_id = str(record.get("id"))
    author_id = canonical_user_id(record.get("author_id"))
    prompt = build_triplet_user_prompt(record)
    try:
        response = client.chat_json(
            system_prompt=TRIPLET_SYSTEM_PROMPT,
            user_prompt=prompt,
            max_tokens=700,
        )
        normalized = normalize_triplet_response(response["parsed"])
        return {
            "status": "ok",
            "payload": {
                "tweet_id": tweet_id,
                "author_id": author_id,
                "created_at": record.get("created_at"),
                "lang": record.get("lang"),
                "prompt_version": TRIPLET_PROMPT_VERSION,
                "model": client.settings.model,
                "status": "ok",
                "compressed_text": normalized["compressed_text"],
                "triplets": normalized["triplets"],
                "triplet_text": normalized["triplet_text"],
                "confidence": normalized["confidence"],
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
                "prompt_version": TRIPLET_PROMPT_VERSION,
                "model": client.settings.model,
                "status": "error",
                "error": str(exc),
            },
        }


def build_triplet_user_prompt(record: dict[str, Any]) -> str:
    """Build the user prompt for triplet extraction."""

    text = str(record.get("text") or "").strip()
    lang = str(record.get("lang") or "unknown")
    return (
        "Extract semantic triplets from the tweet below and return JSON with this schema:\n"
        '{"compressed_text": "...", "triplets": [{"subject": "...", "predicate": "...", "object": "..."}], "confidence": 0.0}\n'
        "Keep at most 5 triplets. Use concise normalized wording.\n"
        f"Language: {lang}\n"
        f"Tweet:\n{text}"
    )


def normalize_triplet_response(payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize loosely structured LLM JSON into a stable schema."""

    compressed_text = _first_non_empty(
        payload.get("compressed_text"),
        payload.get("summary"),
        payload.get("compressed"),
        payload.get("normalized_text"),
    )
    raw_triplets = payload.get("triplets")
    if raw_triplets is None:
        raw_triplets = payload.get("relations") or payload.get("tuples") or []
    triplets = []
    if isinstance(raw_triplets, list):
        for item in raw_triplets:
            normalized = _normalize_triplet_item(item)
            if normalized is not None:
                triplets.append(normalized)
    if not compressed_text:
        compressed_text = "; ".join(
            f"{triplet['subject']} | {triplet['predicate']} | {triplet['object']}" for triplet in triplets
        ).strip()
    triplet_text = compressed_text or " ; ".join(
        f"{triplet['subject']} | {triplet['predicate']} | {triplet['object']}" for triplet in triplets
    )
    confidence = payload.get("confidence")
    try:
        confidence_value = float(confidence) if confidence is not None else None
    except (TypeError, ValueError):
        confidence_value = None
    return {
        "compressed_text": compressed_text,
        "triplets": triplets,
        "triplet_text": triplet_text,
        "confidence": confidence_value,
    }


def build_user_triplet_documents(tweet_output_path: Path) -> list[dict[str, Any]]:
    """Aggregate tweet-level triplet outputs into user-level documents."""

    per_user: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "author_id": "",
            "tweet_count": 0,
            "triplet_count": 0,
            "triplet_document_parts": [],
            "tweet_ids": [],
        }
    )
    for record in read_jsonl_records(tweet_output_path):
        author_id = canonical_user_id(record.get("author_id"))
        if not author_id:
            continue
        row = per_user[author_id]
        row["author_id"] = author_id
        row["tweet_count"] += 1
        row["triplet_count"] += len(record.get("triplets") or [])
        row["tweet_ids"].append(str(record.get("tweet_id") or ""))
        triplet_text = str(record.get("triplet_text") or "").strip()
        if triplet_text:
            row["triplet_document_parts"].append(triplet_text)

    documents = []
    for author_id, row in sorted(per_user.items()):
        documents.append(
            {
                "author_id": author_id,
                "tweet_count": row["tweet_count"],
                "triplet_count": row["triplet_count"],
                "tweet_ids": [tweet_id for tweet_id in row["tweet_ids"] if tweet_id],
                "triplet_document": "\n".join(row["triplet_document_parts"]).strip(),
            }
        )
    return documents


def _normalize_triplet_item(item: Any) -> dict[str, str] | None:
    if isinstance(item, dict):
        subject = _first_non_empty(item.get("subject"), item.get("subj"), item.get("s"))
        predicate = _first_non_empty(item.get("predicate"), item.get("relation"), item.get("p"))
        obj = _first_non_empty(item.get("object"), item.get("obj"), item.get("o"))
        if subject and predicate and obj:
            return {"subject": subject, "predicate": predicate, "object": obj}
        return None
    if isinstance(item, (list, tuple)) and len(item) >= 3:
        subject, predicate, obj = (str(item[0]).strip(), str(item[1]).strip(), str(item[2]).strip())
        if subject and predicate and obj:
            return {"subject": subject, "predicate": predicate, "object": obj}
    return None


def _first_non_empty(*values: Any) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""

"""Build a unified user-level feature table from sampled and derived outputs."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from . import config
from .derived_common import select_tweets_for_derived_tasks
from .normalize import (
    canonical_user_id,
    normalize_followers_count,
    normalize_following_count,
    normalize_label,
    normalize_split,
    normalize_tweet_count_hint,
    normalize_verified_bucket,
)
from .readers import read_csv_rows, read_jsonl_records, read_manifest, write_csv, write_json, write_jsonl

POST_TYPE_COARSE_TYPES = ("original", "retweet", "comment_reply", "link_share")
POST_TYPE_DETAIL_TYPES = ("original", "retweet", "reply", "quote_comment", "link_share", "other")


def build_user_feature_table(
    sample_root: Path,
    output_root: Path,
    *,
    triplet_seed: int = config.DEFAULT_SEED,
    post_type_seed: int = config.DEFAULT_SEED,
) -> dict[str, Any]:
    """Assemble a unified user-level feature table for downstream modeling."""

    output_root.mkdir(parents=True, exist_ok=True)

    user_records = _load_user_records(sample_root / "user.jsonl")
    split_map = {
        canonical_user_id(row.get("id") or row.get("user_id")): normalize_split(row.get("split"))
        for row in read_csv_rows(sample_root / "split.csv")
    }
    label_map = {
        canonical_user_id(row.get("id") or row.get("user_id")): normalize_label(row.get("label"))
        for row in read_csv_rows(sample_root / "label.csv")
    }

    audit_rows = _load_audit_rows(sample_root / "analysis" / "field_audit" / "user_feature_availability.csv")
    post_type_root = sample_root / "derived" / "post_types"
    triplet_root = sample_root / "derived" / "triplets"
    post_type_manifest = read_manifest(post_type_root / "run_manifest.json")
    triplet_manifest = read_manifest(triplet_root / "run_manifest.json")

    post_type_rows = _load_post_type_rows(post_type_root / "user_post_type_distribution.jsonl")
    triplet_rows = _load_triplet_rows(triplet_root / "user_triplet_documents.jsonl")

    post_type_expected_counts = _expected_tweet_counts(
        sample_root,
        manifest=post_type_manifest,
        seed=post_type_seed,
    )
    triplet_expected_counts = _expected_tweet_counts(
        sample_root,
        manifest=triplet_manifest,
        seed=triplet_seed,
    )
    post_type_unresolved_errors = _load_unresolved_error_counts(
        success_path=post_type_root / "tweet_post_types.jsonl",
        error_path=post_type_root / "tweet_post_type_errors.jsonl",
    )
    triplet_unresolved_errors = _load_unresolved_error_counts(
        success_path=triplet_root / "tweet_triplets.jsonl",
        error_path=triplet_root / "tweet_triplet_errors.jsonl",
    )

    rows = []
    for user_id in sorted(user_records):
        user_record = user_records[user_id]
        audit_row = audit_rows.get(user_id, {})
        post_type_row = post_type_rows.get(user_id, {})
        triplet_row = triplet_rows.get(user_id, {})

        post_type_tweet_count = int(post_type_row.get("tweet_count", 0) or 0)
        expected_post_type_tweet_count = int(post_type_expected_counts.get(user_id, 0))
        unresolved_post_type_errors = int(post_type_unresolved_errors.get(user_id, 0))
        post_type_missing_tweet_count = max(
            expected_post_type_tweet_count - post_type_tweet_count - unresolved_post_type_errors,
            0,
        )
        triplet_tweet_count = int(triplet_row.get("tweet_count", 0) or 0)
        expected_triplet_tweet_count = int(triplet_expected_counts.get(user_id, 0))
        unresolved_triplet_errors = int(triplet_unresolved_errors.get(user_id, 0))
        triplet_missing_tweet_count = max(
            expected_triplet_tweet_count - triplet_tweet_count - unresolved_triplet_errors,
            0,
        )

        row = {
            "user_id": user_id,
            "split": split_map.get(user_id, "missing"),
            "label": label_map.get(user_id, "missing"),
            "username": str(user_record.get("username") or ""),
            "name": str(user_record.get("name") or ""),
            "created_at": str(user_record.get("created_at") or ""),
            "description": str(user_record.get("description") or ""),
            "description_present": int(bool(str(user_record.get("description") or "").strip())),
            "profile_url": str(user_record.get("url") or ""),
            "profile_url_present": int(bool(str(user_record.get("url") or "").strip())),
            "verified": int(bool(user_record.get("verified"))),
            "verified_bucket": normalize_verified_bucket(user_record),
            "followers_count": normalize_followers_count(user_record),
            "following_count": normalize_following_count(user_record),
            "account_tweet_count": normalize_tweet_count_hint(user_record),
            "tweets_total": _int_value(audit_row.get("tweets_total")),
            "tweets_with_text": _int_value(audit_row.get("tweets_with_text")),
            "tweets_with_created_at": _int_value(audit_row.get("tweets_with_created_at")),
            "tweets_with_public_metrics": _int_value(audit_row.get("tweets_with_public_metrics")),
            "tweets_with_references": _int_value(audit_row.get("tweets_with_references")),
            "tweets_with_external_url": _int_value(audit_row.get("tweets_with_external_url")),
            "following_out_degree": _int_value(audit_row.get("following_out_degree")),
            "following_in_degree": _int_value(audit_row.get("following_in_degree")),
            "post_edge_count": _int_value(audit_row.get("post_edge_count")),
            "can_triplet": _int_value(audit_row.get("can_triplet")),
            "can_post_type": _int_value(audit_row.get("can_post_type")),
            "can_time_feature": _int_value(audit_row.get("can_time_feature")),
            "can_behavior_feature": _int_value(audit_row.get("can_behavior_feature")),
            "can_network_feature": _int_value(audit_row.get("can_network_feature")),
            "can_full_pipeline": _int_value(audit_row.get("can_full_pipeline")),
            "post_type_tweet_count": post_type_tweet_count,
            "post_type_expected_tweet_count": expected_post_type_tweet_count,
            "post_type_unresolved_error_count": unresolved_post_type_errors,
            "post_type_missing_tweet_count": post_type_missing_tweet_count,
            "post_type_incomplete_flag": int(
                expected_post_type_tweet_count > 0
                and (post_type_tweet_count < expected_post_type_tweet_count or unresolved_post_type_errors > 0)
            ),
            "triplet_tweet_count": triplet_tweet_count,
            "triplet_expected_tweet_count": expected_triplet_tweet_count,
            "triplet_unresolved_error_count": unresolved_triplet_errors,
            "triplet_missing_tweet_count": triplet_missing_tweet_count,
            "triplet_incomplete_flag": int(
                expected_triplet_tweet_count > 0
                and (triplet_tweet_count < expected_triplet_tweet_count or unresolved_triplet_errors > 0)
            ),
            "triplet_count": _int_value(triplet_row.get("triplet_count")),
            "triplet_document": str(triplet_row.get("triplet_document") or ""),
            "triplet_document_present": int(bool(str(triplet_row.get("triplet_document") or "").strip())),
            "triplet_document_length": len(str(triplet_row.get("triplet_document") or "")),
        }

        tweet_count = max(post_type_tweet_count, 1)
        coarse_counts = post_type_row.get("coarse_counts") if isinstance(post_type_row.get("coarse_counts"), dict) else {}
        detail_counts = post_type_row.get("detail_counts") if isinstance(post_type_row.get("detail_counts"), dict) else {}
        for label in POST_TYPE_COARSE_TYPES:
            count = _int_value(coarse_counts.get(label))
            row[f"post_type_coarse_count_{label}"] = count
            row[f"post_type_coarse_ratio_{label}"] = round(count / tweet_count, 6) if post_type_tweet_count else 0.0
        for label in POST_TYPE_DETAIL_TYPES:
            count = _int_value(detail_counts.get(label))
            row[f"post_type_detail_count_{label}"] = count
            row[f"post_type_detail_ratio_{label}"] = round(count / tweet_count, 6) if post_type_tweet_count else 0.0
        rows.append(row)

    csv_path = output_root / "user_feature_table.csv"
    jsonl_path = output_root / "user_feature_table.jsonl"
    manifest_path = output_root / "feature_table_manifest.json"
    summary_path = output_root / "feature_table_summary.md"

    write_csv(csv_path, _feature_table_fieldnames(), rows)
    write_jsonl(jsonl_path, rows)

    manifest = {
        "sample_root": str(sample_root),
        "output_root": str(output_root),
        "triplet_seed": triplet_seed,
        "post_type_seed": post_type_seed,
        "input_files": {
            "user_jsonl": str(sample_root / "user.jsonl"),
            "split_csv": str(sample_root / "split.csv"),
            "label_csv": str(sample_root / "label.csv"),
            "audit_csv": str(sample_root / "analysis" / "field_audit" / "user_feature_availability.csv"),
            "post_type_distribution": str(post_type_root / "user_post_type_distribution.jsonl"),
            "triplet_documents": str(triplet_root / "user_triplet_documents.jsonl"),
        },
        "counts": {
            "users": len(rows),
            "post_type_users": sum(1 for row in rows if int(row["post_type_tweet_count"]) > 0),
            "triplet_users": sum(1 for row in rows if int(row["triplet_tweet_count"]) > 0),
            "post_type_incomplete_users": sum(1 for row in rows if int(row["post_type_incomplete_flag"]) > 0),
            "triplet_incomplete_users": sum(1 for row in rows if int(row["triplet_incomplete_flag"]) > 0),
            "full_pipeline_users": sum(
                1
                for row in rows
                if int(row["can_full_pipeline"]) > 0
                and int(row["post_type_tweet_count"]) > 0
                and int(row["triplet_tweet_count"]) > 0
            ),
        },
        "totals": {
            "post_type_expected_tweets": sum(post_type_expected_counts.values()),
            "post_type_resolved_tweets": sum(int(row["post_type_tweet_count"]) for row in rows),
            "post_type_unresolved_errors": sum(post_type_unresolved_errors.values()),
            "triplet_expected_tweets": sum(triplet_expected_counts.values()),
            "triplet_resolved_tweets": sum(int(row["triplet_tweet_count"]) for row in rows),
            "triplet_unresolved_errors": sum(triplet_unresolved_errors.values()),
        },
        "files": {
            "csv": str(csv_path),
            "jsonl": str(jsonl_path),
            "summary": str(summary_path),
        },
    }
    write_json(manifest_path, manifest)
    summary_path.write_text(_render_feature_summary(rows, manifest), encoding="utf-8")
    return manifest


def _load_user_records(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for record in read_jsonl_records(path):
        user_id = canonical_user_id(record.get("id"))
        if user_id:
            records[user_id] = record
    return records


def _load_audit_rows(path: Path) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    for row in read_csv_rows(path):
        user_id = canonical_user_id(row.get("user_id"))
        if user_id:
            rows[user_id] = row
    return rows


def _load_post_type_rows(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for record in read_jsonl_records(path):
        user_id = canonical_user_id(record.get("author_id"))
        if user_id:
            rows[user_id] = record
    return rows


def _load_triplet_rows(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for record in read_jsonl_records(path):
        user_id = canonical_user_id(record.get("author_id"))
        if user_id:
            rows[user_id] = record
    return rows


def _expected_tweet_counts(
    sample_root: Path,
    *,
    manifest: dict[str, Any],
    seed: int,
) -> Counter[str]:
    selection = manifest.get("selection", {})
    records, _summary = select_tweets_for_derived_tasks(
        sample_root,
        per_user_limit=int(selection.get("per_user_limit", 0) or 0),
        min_user_tweets=int(selection.get("min_user_tweets", 0) or 0),
        max_users=_none_if_null(selection.get("max_users")),
        max_tweets=_none_if_null(selection.get("max_tweets")),
        seed=seed,
        require_text=True,
    )
    counts: Counter[str] = Counter()
    for record in records:
        counts[canonical_user_id(record.get("author_id"))] += 1
    return counts


def _load_unresolved_error_counts(
    *,
    success_path: Path,
    error_path: Path,
) -> Counter[str]:
    success_ids = {
        str(record.get("tweet_id") or "")
        for record in read_jsonl_records(success_path)
        if str(record.get("tweet_id") or "")
    }
    unresolved: Counter[str] = Counter()
    if not error_path.exists():
        return unresolved
    seen_error_ids: set[str] = set()
    for record in read_jsonl_records(error_path):
        tweet_id = str(record.get("tweet_id") or "")
        if not tweet_id or tweet_id in success_ids or tweet_id in seen_error_ids:
            continue
        seen_error_ids.add(tweet_id)
        unresolved[canonical_user_id(record.get("author_id"))] += 1
    return unresolved


def _none_if_null(value: Any) -> int | None:
    if value in (None, "", "null"):
        return None
    return int(value)


def _int_value(value: Any) -> int:
    if value in (None, ""):
        return 0
    return int(value)


def _feature_table_fieldnames() -> list[str]:
    fieldnames = [
        "user_id",
        "split",
        "label",
        "username",
        "name",
        "created_at",
        "description",
        "description_present",
        "profile_url",
        "profile_url_present",
        "verified",
        "verified_bucket",
        "followers_count",
        "following_count",
        "account_tweet_count",
        "tweets_total",
        "tweets_with_text",
        "tweets_with_created_at",
        "tweets_with_public_metrics",
        "tweets_with_references",
        "tweets_with_external_url",
        "following_out_degree",
        "following_in_degree",
        "post_edge_count",
        "can_triplet",
        "can_post_type",
        "can_time_feature",
        "can_behavior_feature",
        "can_network_feature",
        "can_full_pipeline",
        "post_type_tweet_count",
        "post_type_expected_tweet_count",
        "post_type_unresolved_error_count",
        "post_type_missing_tweet_count",
        "post_type_incomplete_flag",
        "triplet_tweet_count",
        "triplet_expected_tweet_count",
        "triplet_unresolved_error_count",
        "triplet_missing_tweet_count",
        "triplet_incomplete_flag",
        "triplet_count",
        "triplet_document_present",
        "triplet_document_length",
        "triplet_document",
    ]
    for label in POST_TYPE_COARSE_TYPES:
        fieldnames.append(f"post_type_coarse_count_{label}")
        fieldnames.append(f"post_type_coarse_ratio_{label}")
    for label in POST_TYPE_DETAIL_TYPES:
        fieldnames.append(f"post_type_detail_count_{label}")
        fieldnames.append(f"post_type_detail_ratio_{label}")
    return fieldnames


def _render_feature_summary(rows: list[dict[str, Any]], manifest: dict[str, Any]) -> str:
    counts = manifest["counts"]
    totals = manifest["totals"]
    lines = [
        "# User Feature Table Summary",
        "",
        "## Overall",
        f"- Users: {counts['users']}",
        f"- Users with post-type features: {counts['post_type_users']}",
        f"- Users with triplet features: {counts['triplet_users']}",
        f"- Users with complete post-type coverage: {counts['post_type_users'] - counts['post_type_incomplete_users']}",
        f"- Users with complete triplet coverage: {counts['triplet_users'] - counts['triplet_incomplete_users']}",
        f"- Users ready for full downstream pipeline: {counts['full_pipeline_users']}",
        "",
        "## Tweet Coverage",
        f"- Post-type expected tweets: {totals['post_type_expected_tweets']}",
        f"- Post-type resolved tweets: {totals['post_type_resolved_tweets']}",
        f"- Post-type unresolved errors: {totals['post_type_unresolved_errors']}",
        f"- Triplet expected tweets: {totals['triplet_expected_tweets']}",
        f"- Triplet resolved tweets: {totals['triplet_resolved_tweets']}",
        f"- Triplet unresolved errors: {totals['triplet_unresolved_errors']}",
    ]
    split_counter = Counter(row["split"] for row in rows)
    label_counter = Counter(row["label"] for row in rows)
    lines.extend(["", "## Split Distribution"])
    lines.extend(f"- {split}: {count}" for split, count in sorted(split_counter.items()))
    lines.extend(["", "## Label Distribution"])
    lines.extend(f"- {label}: {count}" for label, count in sorted(label_counter.items()))
    return "\n".join(lines) + "\n"

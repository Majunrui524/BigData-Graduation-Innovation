"""Build text embeddings and fused user vectors from the user feature table."""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .llm_client import OpenAICompatibleClient
from .readers import read_jsonl_records, write_json, write_jsonl

DEFAULT_TEXT_FIELD = "triplet_document"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"

LOG_COUNT_FIELDS = (
    "followers_count",
    "following_count",
    "account_tweet_count",
    "tweets_total",
    "following_in_degree",
    "following_out_degree",
    "post_edge_count",
    "post_type_tweet_count",
    "triplet_tweet_count",
    "triplet_count",
)
RATIO_FIELDS = (
    "tweets_with_created_at_ratio",
    "tweets_with_public_metrics_ratio",
    "tweets_with_references_ratio",
    "tweets_with_external_url_ratio",
    "post_type_coarse_ratio_original",
    "post_type_coarse_ratio_retweet",
    "post_type_coarse_ratio_comment_reply",
    "post_type_coarse_ratio_link_share",
    "post_type_detail_ratio_original",
    "post_type_detail_ratio_retweet",
    "post_type_detail_ratio_reply",
    "post_type_detail_ratio_quote_comment",
    "post_type_detail_ratio_link_share",
    "post_type_detail_ratio_other",
)
FLAG_FIELDS = (
    "verified",
    "description_present",
    "profile_url_present",
    "can_time_feature",
    "can_behavior_feature",
    "can_network_feature",
    "can_full_pipeline",
    "post_type_incomplete_flag",
    "triplet_incomplete_flag",
)


def build_user_vectors(
    feature_root: Path,
    output_root: Path,
    *,
    client: OpenAICompatibleClient,
    text_field: str = DEFAULT_TEXT_FIELD,
    fallback_to_description: bool = True,
    batch_size: int = 64,
    max_users: int | None = None,
) -> dict[str, Any]:
    """Build embeddings, numeric vectors, and fused vectors for downstream modeling."""

    rows = list(read_jsonl_records(feature_root / "user_feature_table.jsonl"))
    rows.sort(key=lambda row: str(row.get("user_id") or ""))
    if max_users is not None:
        rows = rows[:max_users]
    output_root.mkdir(parents=True, exist_ok=True)

    text_jobs: list[dict[str, Any]] = []
    source_counter: Counter[str] = Counter()
    for row in rows:
        text, source = _choose_text(row, text_field=text_field, fallback_to_description=fallback_to_description)
        if text:
            text_jobs.append(
                {
                    "user_id": str(row.get("user_id") or ""),
                    "text": text,
                    "text_source": source,
                    "text_length": len(text),
                }
            )
            source_counter[source] += 1
        else:
            source_counter["missing"] += 1

    embedding_by_user: dict[str, dict[str, Any]] = {}
    embedding_dim = 0
    for batch in _batched(text_jobs, batch_size):
        texts = [item["text"] for item in batch]
        vectors = client.embed_texts(texts)
        if len(vectors) != len(batch):
            raise ValueError("Embedding response count did not match request count")
        if vectors and embedding_dim == 0:
            embedding_dim = len(vectors[0])
        for item, vector in zip(batch, vectors):
            embedding_dim = max(embedding_dim, len(vector))
            embedding_by_user[item["user_id"]] = {
                "user_id": item["user_id"],
                "text_source": item["text_source"],
                "text_length": item["text_length"],
                "embedding": vector,
            }

    numeric_feature_names = _numeric_feature_names()
    raw_numeric_rows = [_build_raw_numeric_features(row) for row in rows]
    numeric_stats = _compute_numeric_stats(raw_numeric_rows, numeric_feature_names)
    numeric_rows: list[dict[str, Any]] = []
    fused_rows: list[dict[str, Any]] = []
    embedding_rows: list[dict[str, Any]] = []

    zero_embedding = [0.0] * embedding_dim
    for row, raw_numeric in zip(rows, raw_numeric_rows):
        user_id = str(row.get("user_id") or "")
        embedding_record = embedding_by_user.get(user_id)
        if embedding_record is None:
            embedding_record = {
                "user_id": user_id,
                "text_source": "missing",
                "text_length": 0,
                "embedding": list(zero_embedding),
            }
        elif len(embedding_record["embedding"]) < embedding_dim:
            embedding_record = {
                **embedding_record,
                "embedding": embedding_record["embedding"] + [0.0] * (embedding_dim - len(embedding_record["embedding"])),
            }

        numeric_vector = [
            _scale_feature(raw_numeric.get(name, 0.0), numeric_stats[name]["mean"], numeric_stats[name]["std"])
            for name in numeric_feature_names
        ]
        embedding_rows.append(
            {
                "user_id": user_id,
                "text_source": embedding_record["text_source"],
                "text_length": embedding_record["text_length"],
                "embedding_dim": embedding_dim,
                "embedding": embedding_record["embedding"],
            }
        )
        numeric_rows.append(
            {
                "user_id": user_id,
                "numeric_feature_names": list(numeric_feature_names),
                "numeric_features": raw_numeric,
                "numeric_vector": numeric_vector,
            }
        )
        fused_rows.append(
            {
                "user_id": user_id,
                "text_source": embedding_record["text_source"],
                "embedding_dim": embedding_dim,
                "numeric_dim": len(numeric_feature_names),
                "fused_dim": embedding_dim + len(numeric_feature_names),
                "fused_vector": list(embedding_record["embedding"]) + numeric_vector,
            }
        )

    embedding_path = output_root / "user_embedding_vectors.jsonl"
    numeric_path = output_root / "user_numeric_vectors.jsonl"
    fused_path = output_root / "user_fused_vectors.jsonl"
    manifest_path = output_root / "vector_manifest.json"
    summary_path = output_root / "vector_summary.md"

    write_jsonl(embedding_path, embedding_rows)
    write_jsonl(numeric_path, numeric_rows)
    write_jsonl(fused_path, fused_rows)

    manifest = {
        "feature_root": str(feature_root),
        "output_root": str(output_root),
        "embedding_model": client.settings.model,
        "batch_size": batch_size,
        "text_field": text_field,
        "fallback_to_description": fallback_to_description,
        "embedding_dim": embedding_dim,
        "numeric_dim": len(numeric_feature_names),
        "fused_dim": embedding_dim + len(numeric_feature_names),
        "counts": {
            "users": len(rows),
            "embedded_users": sum(1 for row in embedding_rows if row["text_source"] != "missing"),
            "missing_text_users": sum(1 for row in embedding_rows if row["text_source"] == "missing"),
        },
        "text_source_distribution": dict(source_counter),
        "numeric_feature_names": list(numeric_feature_names),
        "numeric_stats": numeric_stats,
        "files": {
            "embedding_vectors": str(embedding_path),
            "numeric_vectors": str(numeric_path),
            "fused_vectors": str(fused_path),
            "summary": str(summary_path),
        },
    }
    write_json(manifest_path, manifest)
    summary_path.write_text(_render_vector_summary(manifest), encoding="utf-8")
    return manifest


def _choose_text(
    row: dict[str, Any],
    *,
    text_field: str,
    fallback_to_description: bool,
) -> tuple[str, str]:
    primary = str(row.get(text_field) or "").strip()
    if primary:
        return primary, text_field
    if fallback_to_description:
        fallback = str(row.get("description") or "").strip()
        if fallback:
            return fallback, "description_fallback"
    return "", "missing"


def _numeric_feature_names() -> tuple[str, ...]:
    names = [f"log1p_{field}" for field in LOG_COUNT_FIELDS]
    names.extend(RATIO_FIELDS)
    names.extend(FLAG_FIELDS)
    return tuple(names)


def _build_raw_numeric_features(row: dict[str, Any]) -> dict[str, float]:
    tweets_total = max(_float_value(row.get("tweets_total")), 1.0)
    features: dict[str, float] = {}
    for field in LOG_COUNT_FIELDS:
        features[f"log1p_{field}"] = math.log1p(max(_float_value(row.get(field)), 0.0))
    features["tweets_with_created_at_ratio"] = _safe_ratio(row.get("tweets_with_created_at"), tweets_total)
    features["tweets_with_public_metrics_ratio"] = _safe_ratio(row.get("tweets_with_public_metrics"), tweets_total)
    features["tweets_with_references_ratio"] = _safe_ratio(row.get("tweets_with_references"), tweets_total)
    features["tweets_with_external_url_ratio"] = _safe_ratio(row.get("tweets_with_external_url"), tweets_total)
    for field in (
        "post_type_coarse_ratio_original",
        "post_type_coarse_ratio_retweet",
        "post_type_coarse_ratio_comment_reply",
        "post_type_coarse_ratio_link_share",
        "post_type_detail_ratio_original",
        "post_type_detail_ratio_retweet",
        "post_type_detail_ratio_reply",
        "post_type_detail_ratio_quote_comment",
        "post_type_detail_ratio_link_share",
        "post_type_detail_ratio_other",
    ):
        features[field] = _float_value(row.get(field))
    for field in FLAG_FIELDS:
        features[field] = _float_value(row.get(field))
    return features


def _compute_numeric_stats(
    raw_rows: list[dict[str, float]],
    feature_names: Iterable[str],
) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    for name in feature_names:
        values = [float(row.get(name, 0.0)) for row in raw_rows]
        if not values:
            stats[name] = {"mean": 0.0, "std": 1.0}
            continue
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        std = math.sqrt(variance)
        stats[name] = {"mean": round(mean, 8), "std": round(std, 8)}
    return stats


def _scale_feature(value: float, mean: float, std: float) -> float:
    if std <= 0:
        return 0.0
    return round((float(value) - mean) / std, 8)


def _safe_ratio(value: Any, denominator: float) -> float:
    numerator = max(_float_value(value), 0.0)
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 8)


def _float_value(value: Any) -> float:
    if value in (None, ""):
        return 0.0
    return float(value)


def _batched(items: list[dict[str, Any]], size: int) -> Iterable[list[dict[str, Any]]]:
    batch_size = max(int(size), 1)
    for index in range(0, len(items), batch_size):
        yield items[index : index + batch_size]


def _render_vector_summary(manifest: dict[str, Any]) -> str:
    counts = manifest["counts"]
    lines = [
        "# User Vector Summary",
        "",
        "## Overall",
        f"- Users: {counts['users']}",
        f"- Embedded users: {counts['embedded_users']}",
        f"- Missing-text users: {counts['missing_text_users']}",
        f"- Embedding model: {manifest['embedding_model']}",
        f"- Embedding dim: {manifest['embedding_dim']}",
        f"- Numeric dim: {manifest['numeric_dim']}",
        f"- Fused dim: {manifest['fused_dim']}",
        "",
        "## Text Sources",
    ]
    lines.extend(f"- {source}: {count}" for source, count in sorted(manifest["text_source_distribution"].items()))
    return "\n".join(lines) + "\n"

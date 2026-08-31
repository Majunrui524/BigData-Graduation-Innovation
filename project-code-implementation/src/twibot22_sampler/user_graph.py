"""Build user similarity graphs via early or late fusion."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .channel_similarity import (
    jaccard_similarity,
    js_similarity,
    normalized_cosine_similarity,
    dtw_average_distance,
    weighted_available_average,
)
from .normalize import canonical_user_id
from .readers import read_csv_rows, read_jsonl_records, write_csv, write_json

DEFAULT_GRAPH_K = 10
DEFAULT_GRAPH_METRIC = "cosine"
DEFAULT_GRAPH_CHUNK_SIZE = 512
DEFAULT_GRAPH_BACKEND = "auto"
DEFAULT_GRAPH_SYMMETRIZE = "union_max"
DEFAULT_GRAPH_FUSION_MODE = "late"
DEFAULT_GRAPH_CANDIDATE_K = 100
DEFAULT_GRAPH_LAMBDA_CONTENT = 0.25
DEFAULT_GRAPH_LAMBDA_BEHAVIOR = 0.25
DEFAULT_GRAPH_LAMBDA_TEMPORAL = 0.25
DEFAULT_GRAPH_LAMBDA_NETWORK = 0.25
MAX_PYTHON_BACKEND_USERS = 1200

BEHAVIOR_DISTRIBUTION_FIELDS = (
    "post_type_coarse_ratio_original",
    "post_type_coarse_ratio_retweet",
    "post_type_coarse_ratio_comment_reply",
    "post_type_coarse_ratio_link_share",
)
BEHAVIOR_SCALAR_FIELDS = (
    "followers_count",
    "following_count",
    "tweets_total",
    "post_type_tweet_count",
    "triplet_tweet_count",
    "verified",
    "tweets_with_created_at_ratio",
    "tweets_with_public_metrics_ratio",
    "tweets_with_references_ratio",
    "tweets_with_external_url_ratio",
)

try:  # pragma: no cover - exercised indirectly when numpy is installed
    import numpy as _np
except ImportError:  # pragma: no cover - fallback path is covered in tests
    _np = None


def build_user_graph(
    source_root: Path,
    output_root: Path,
    *,
    k: int = DEFAULT_GRAPH_K,
    metric: str = DEFAULT_GRAPH_METRIC,
    min_similarity: float = 0.0,
    backend: str = DEFAULT_GRAPH_BACKEND,
    symmetrize: str = DEFAULT_GRAPH_SYMMETRIZE,
    chunk_size: int = DEFAULT_GRAPH_CHUNK_SIZE,
    max_users: int | None = None,
    fusion_mode: str = DEFAULT_GRAPH_FUSION_MODE,
    vector_root: Path | None = None,
    feature_root: Path | None = None,
    temporal_root: Path | None = None,
    candidate_k: int = DEFAULT_GRAPH_CANDIDATE_K,
    lambda_content: float = DEFAULT_GRAPH_LAMBDA_CONTENT,
    lambda_behavior: float = DEFAULT_GRAPH_LAMBDA_BEHAVIOR,
    lambda_temporal: float = DEFAULT_GRAPH_LAMBDA_TEMPORAL,
    lambda_network: float = DEFAULT_GRAPH_LAMBDA_NETWORK,
) -> dict[str, Any]:
    """Build a kNN user graph via early fused vectors or late channel fusion."""

    if metric != "cosine":
        raise ValueError(f"Unsupported metric: {metric}")
    if symmetrize not in {"union_max", "mutual_max", "directed"}:
        raise ValueError(f"Unsupported symmetrize mode: {symmetrize}")
    if fusion_mode not in {"early", "late"}:
        raise ValueError(f"Unsupported fusion_mode: {fusion_mode}")

    if fusion_mode == "early":
        vector_root = vector_root or source_root
        return _build_user_graph_early(
            vector_root,
            output_root,
            k=k,
            metric=metric,
            min_similarity=min_similarity,
            backend=backend,
            symmetrize=symmetrize,
            chunk_size=chunk_size,
            max_users=max_users,
        )

    sample_root = source_root
    vector_root = vector_root or (sample_root / "analysis" / "user_vectors")
    feature_root = feature_root or (sample_root / "analysis" / "user_features")
    temporal_root = temporal_root or (sample_root / "analysis" / "temporal_profiles")
    return _build_user_graph_late(
        sample_root,
        output_root,
        vector_root=vector_root,
        feature_root=feature_root,
        temporal_root=temporal_root,
        k=k,
        min_similarity=min_similarity,
        backend=backend,
        symmetrize=symmetrize,
        chunk_size=chunk_size,
        max_users=max_users,
        candidate_k=candidate_k,
        lambda_content=lambda_content,
        lambda_behavior=lambda_behavior,
        lambda_temporal=lambda_temporal,
        lambda_network=lambda_network,
    )


def _build_user_graph_early(
    vector_root: Path,
    output_root: Path,
    *,
    k: int,
    metric: str,
    min_similarity: float,
    backend: str,
    symmetrize: str,
    chunk_size: int,
    max_users: int | None,
) -> dict[str, Any]:
    rows = list(read_jsonl_records(vector_root / "user_fused_vectors.jsonl"))
    rows.sort(key=lambda row: str(row.get("user_id") or ""))
    if max_users is not None:
        rows = rows[: max(int(max_users), 0)]
    if not rows:
        raise ValueError(f"No fused vector rows found under {vector_root}")

    user_ids = [str(row.get("user_id") or "") for row in rows]
    vectors = [list(_float_vector(row.get("fused_vector"))) for row in rows]
    vector_dim = len(vectors[0]) if vectors else 0
    if any(len(vector) != vector_dim for vector in vectors):
        raise ValueError("Fused vectors do not share a consistent dimension")

    resolved_backend = _resolve_backend(backend, len(user_ids))
    if resolved_backend == "numpy":
        directed_edges, zero_norm_users = _build_directed_edges_numpy(
            user_ids,
            vectors,
            k=k,
            min_similarity=min_similarity,
            chunk_size=chunk_size,
        )
    else:
        directed_edges, zero_norm_users = _build_directed_edges_python(
            user_ids,
            vectors,
            k=k,
            min_similarity=min_similarity,
        )

    undirected_edges = _symmetrize_edges(directed_edges, mode=symmetrize)
    degree_summary = _compute_degree_summary(user_ids, directed_edges, undirected_edges)

    output_root.mkdir(parents=True, exist_ok=True)
    directed_path = output_root / "user_knn_directed_edges.csv"
    undirected_path = output_root / "user_knn_edges.csv"
    manifest_path = output_root / "graph_manifest.json"
    summary_path = output_root / "graph_summary.md"

    write_csv(
        directed_path,
        ["source_user_id", "target_user_id", "similarity", "rank"],
        directed_edges,
    )
    write_csv(
        undirected_path,
        ["source_user_id", "target_user_id", "weight", "support"],
        undirected_edges,
    )

    manifest = {
        "fusion_mode": "early",
        "vector_root": str(vector_root),
        "output_root": str(output_root),
        "backend": resolved_backend,
        "metric": metric,
        "k": max(int(k), 0),
        "min_similarity": float(min_similarity),
        "symmetrize": symmetrize,
        "chunk_size": max(int(chunk_size), 1),
        "vector_dim": vector_dim,
        "candidate_k": None,
        "channel_weights": {},
        "counts": {
            "users": len(user_ids),
            "directed_edges": len(directed_edges),
            "undirected_edges": len(undirected_edges),
            "zero_norm_users": zero_norm_users,
        },
        "degree_summary": degree_summary,
        "files": {
            "directed_edges": str(directed_path),
            "undirected_edges": str(undirected_path),
            "summary": str(summary_path),
        },
    }
    write_json(manifest_path, manifest)
    summary_path.write_text(_render_graph_summary(manifest), encoding="utf-8")
    return manifest


def _build_user_graph_late(
    sample_root: Path,
    output_root: Path,
    *,
    vector_root: Path,
    feature_root: Path,
    temporal_root: Path,
    k: int,
    min_similarity: float,
    backend: str,
    symmetrize: str,
    chunk_size: int,
    max_users: int | None,
    candidate_k: int,
    lambda_content: float,
    lambda_behavior: float,
    lambda_temporal: float,
    lambda_network: float,
) -> dict[str, Any]:
    user_ids = _load_sample_user_ids(sample_root / "user.jsonl")
    if max_users is not None:
        user_ids = user_ids[: max(int(max_users), 0)]
    if not user_ids:
        raise ValueError(f"No users found under {sample_root / 'user.jsonl'}")
    user_id_set = set(user_ids)

    resolved_backend = _resolve_backend(backend, len(user_ids))
    embedding_rows = _load_embedding_rows(vector_root / "user_embedding_vectors.jsonl", user_id_set)
    feature_rows = _load_feature_rows(feature_root / "user_feature_table.jsonl", user_id_set)
    temporal_rows = _load_temporal_rows(temporal_root / "user_temporal_profiles.jsonl", user_id_set)
    network_rows = _load_network_rows(sample_root / "edge.csv", user_id_set)

    content_vectors = {
        user_id: row["embedding"]
        for user_id, row in embedding_rows.items()
        if row["text_source"] != "missing" and _vector_has_signal(row["embedding"])
    }
    content_normed = {user_id: _normalize_vector(vector) for user_id, vector in content_vectors.items()}
    behavior_bundle = _build_behavior_bundle(user_ids, feature_rows)
    temporal_bundle = _build_temporal_bundle(user_ids, temporal_rows)
    network_bundle = _build_network_bundle(user_ids, network_rows)

    candidate_pairs = _build_candidate_pairs(
        user_ids,
        content_normed,
        behavior_bundle["scalar_normed"],
        network_bundle["out_neighbors"],
        backend=resolved_backend,
        chunk_size=chunk_size,
        candidate_k=candidate_k,
    )

    weights = {
        "content": max(float(lambda_content), 0.0),
        "behavior": max(float(lambda_behavior), 0.0),
        "temporal": max(float(lambda_temporal), 0.0),
        "network": max(float(lambda_network), 0.0),
    }

    directed_edges: list[dict[str, Any]] = []
    directed_detailed: list[dict[str, Any]] = []
    channel_directed_counts: Counter[str] = Counter()
    score_cache: dict[tuple[str, str], dict[str, float | None]] = {}
    neighbor_count = min(max(int(k), 0), max(len(user_ids) - 1, 0))

    for source_user_id in user_ids:
        scored_rows: list[dict[str, Any]] = []
        for target_user_id in sorted(candidate_pairs.get(source_user_id, set())):
            if source_user_id == target_user_id:
                continue
            pair_key = tuple(sorted((source_user_id, target_user_id)))
            if pair_key not in score_cache:
                score_cache[pair_key] = _compute_pair_scores(
                    source_user_id=pair_key[0],
                    target_user_id=pair_key[1],
                    content_normed=content_normed,
                    behavior_bundle=behavior_bundle,
                    temporal_bundle=temporal_bundle,
                    network_bundle=network_bundle,
                    weights=weights,
                )
            payload = dict(score_cache[pair_key])
            fused_weight = payload.get("fused_weight")
            if fused_weight is None or float(fused_weight) < float(min_similarity):
                continue
            scored_rows.append(
                {
                    "source_user_id": source_user_id,
                    "target_user_id": target_user_id,
                    **payload,
                }
            )

        scored_rows.sort(
            key=lambda row: (-float(row["fused_weight"]), str(row["target_user_id"])),
        )
        for rank, row in enumerate(scored_rows[:neighbor_count], start=1):
            directed_edges.append(
                {
                    "source_user_id": row["source_user_id"],
                    "target_user_id": row["target_user_id"],
                    "similarity": round(float(row["fused_weight"]), 8),
                    "rank": rank,
                }
            )
            directed_detailed.append(
                {
                    "source_user_id": row["source_user_id"],
                    "target_user_id": row["target_user_id"],
                    "content_similarity": _round_or_blank(row.get("content_similarity")),
                    "behavior_similarity": _round_or_blank(row.get("behavior_similarity")),
                    "temporal_similarity": _round_or_blank(row.get("temporal_similarity")),
                    "network_similarity": _round_or_blank(row.get("network_similarity")),
                    "available_weight_sum": round(float(row["available_weight_sum"]), 8),
                    "fused_weight": round(float(row["fused_weight"]), 8),
                    "rank": rank,
                }
            )
            for channel_name in ("content", "behavior", "temporal", "network"):
                if row.get(f"{channel_name}_similarity") is not None:
                    channel_directed_counts[channel_name] += 1

    undirected_edges = _symmetrize_edges(directed_edges, mode=symmetrize)
    undirected_detailed = _symmetrize_detailed_edges(directed_detailed, mode=symmetrize)
    degree_summary = _compute_degree_summary(user_ids, directed_edges, undirected_edges)

    output_root.mkdir(parents=True, exist_ok=True)
    directed_path = output_root / "user_knn_directed_edges.csv"
    undirected_path = output_root / "user_knn_edges.csv"
    detailed_path = output_root / "user_knn_edges_detailed.csv"
    manifest_path = output_root / "graph_manifest.json"
    summary_path = output_root / "graph_summary.md"

    write_csv(
        directed_path,
        ["source_user_id", "target_user_id", "similarity", "rank"],
        directed_edges,
    )
    write_csv(
        undirected_path,
        ["source_user_id", "target_user_id", "weight", "support"],
        undirected_edges,
    )
    write_csv(
        detailed_path,
        [
            "source_user_id",
            "target_user_id",
            "content_similarity",
            "behavior_similarity",
            "temporal_similarity",
            "network_similarity",
            "available_weight_sum",
            "fused_weight",
            "support",
        ],
        undirected_detailed,
    )

    directed_total = len(directed_edges)
    manifest = {
        "fusion_mode": "late",
        "sample_root": str(sample_root),
        "vector_root": str(vector_root),
        "feature_root": str(feature_root),
        "temporal_root": str(temporal_root),
        "output_root": str(output_root),
        "backend": resolved_backend,
        "metric": "late_fusion",
        "k": max(int(k), 0),
        "candidate_k": max(int(candidate_k), 0),
        "min_similarity": float(min_similarity),
        "symmetrize": symmetrize,
        "chunk_size": max(int(chunk_size), 1),
        "channel_weights": weights,
        "feature_dims": {
            "content_embedding_dim": embedding_rows[next(iter(embedding_rows))]["embedding_dim"] if embedding_rows else 0,
            "behavior_scalar_dim": len(BEHAVIOR_SCALAR_FIELDS),
            "behavior_distribution_dim": len(BEHAVIOR_DISTRIBUTION_FIELDS),
            "temporal_dim": 24,
            "network_degree_dim": 2,
        },
        "counts": {
            "users": len(user_ids),
            "candidate_directed_pairs": sum(len(targets) for targets in candidate_pairs.values()),
            "directed_edges": len(directed_edges),
            "undirected_edges": len(undirected_edges),
            "zero_norm_users": 0,
        },
        "channel_edge_coverage": {
            channel_name: {
                "directed_edges": int(channel_directed_counts.get(channel_name, 0)),
                "coverage": round(
                    channel_directed_counts.get(channel_name, 0) / directed_total,
                    8,
                )
                if directed_total
                else 0.0,
            }
            for channel_name in ("content", "behavior", "temporal", "network")
        },
        "input_coverage": {
            "content_users": len(content_normed),
            "behavior_users": sum(1 for user_id in user_ids if user_id in behavior_bundle["scalar_normed"]),
            "behavior_distribution_users": sum(1 for user_id in user_ids if user_id in behavior_bundle["distribution"]),
            "temporal_ready_users": sum(1 for user_id in user_ids if temporal_bundle["ready"].get(user_id, False)),
            "network_users": len(network_bundle["out_neighbors"]),
        },
        "degree_summary": degree_summary,
        "files": {
            "directed_edges": str(directed_path),
            "undirected_edges": str(undirected_path),
            "detailed_edges": str(detailed_path),
            "summary": str(summary_path),
        },
    }
    write_json(manifest_path, manifest)
    summary_path.write_text(_render_graph_summary(manifest), encoding="utf-8")
    return manifest


def _load_sample_user_ids(path: Path) -> list[str]:
    user_ids = []
    for row in read_jsonl_records(path):
        user_id = canonical_user_id(row.get("id") or row.get("user_id"))
        if user_id:
            user_ids.append(user_id)
    user_ids.sort()
    return user_ids


def _load_embedding_rows(path: Path, user_ids: set[str]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for row in read_jsonl_records(path):
        user_id = canonical_user_id(row.get("user_id"))
        if user_id and user_id in user_ids:
            rows[user_id] = {
                "user_id": user_id,
                "text_source": str(row.get("text_source") or "missing"),
                "embedding_dim": int(row.get("embedding_dim") or len(row.get("embedding") or [])),
                "embedding": [float(value) for value in row.get("embedding") or []],
            }
    return rows


def _load_feature_rows(path: Path, user_ids: set[str]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for row in read_jsonl_records(path):
        user_id = canonical_user_id(row.get("user_id"))
        if user_id and user_id in user_ids:
            rows[user_id] = row
    return rows


def _load_temporal_rows(path: Path, user_ids: set[str]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for row in read_jsonl_records(path):
        user_id = canonical_user_id(row.get("user_id"))
        if user_id and user_id in user_ids:
            rows[user_id] = row
    return rows


def _load_network_rows(path: Path, user_ids: set[str]) -> dict[str, Any]:
    out_neighbors: dict[str, set[str]] = {user_id: set() for user_id in user_ids}
    in_degree: Counter[str] = Counter()
    out_degree: Counter[str] = Counter()
    for row in read_csv_rows(path):
        if str(row.get("relation") or "").strip().lower() != "following":
            continue
        source_user_id = canonical_user_id(row.get("source_id"))
        target_user_id = canonical_user_id(row.get("target_id"))
        if source_user_id not in user_ids or target_user_id not in user_ids or source_user_id == target_user_id:
            continue
        out_neighbors[source_user_id].add(target_user_id)
        out_degree[source_user_id] += 1
        in_degree[target_user_id] += 1
    return {
        "out_neighbors": out_neighbors,
        "in_degree": in_degree,
        "out_degree": out_degree,
    }


def _build_behavior_bundle(
    user_ids: list[str],
    feature_rows: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    raw_scalar_rows: dict[str, list[float]] = {}
    distribution_rows: dict[str, list[float]] = {}

    for user_id in user_ids:
        row = feature_rows.get(user_id, {})
        raw_scalar_rows[user_id] = _build_behavior_scalar_vector(row)
        distribution = _build_behavior_distribution_vector(row)
        if distribution is not None:
            distribution_rows[user_id] = distribution

    means: list[float] = []
    stds: list[float] = []
    for column_index in range(len(BEHAVIOR_SCALAR_FIELDS)):
        column_values = [raw_scalar_rows[user_id][column_index] for user_id in user_ids]
        mean_value = sum(column_values) / len(column_values) if column_values else 0.0
        variance = (
            sum((value - mean_value) ** 2 for value in column_values) / len(column_values)
            if column_values
            else 0.0
        )
        std_value = math.sqrt(variance)
        means.append(mean_value)
        stds.append(std_value if std_value > 1e-9 else 1.0)

    standardized_rows = {
        user_id: [
            (value - means[index]) / stds[index]
            for index, value in enumerate(values)
        ]
        for user_id, values in raw_scalar_rows.items()
    }
    scalar_normed = {
        user_id: _normalize_vector(vector)
        for user_id, vector in standardized_rows.items()
        if _vector_has_signal(vector)
    }
    return {
        "raw_scalar": raw_scalar_rows,
        "scalar": standardized_rows,
        "scalar_normed": scalar_normed,
        "distribution": distribution_rows,
    }


def _build_behavior_scalar_vector(row: dict[str, Any]) -> list[float]:
    return [
        math.log1p(max(_to_float(row.get("followers_count")), 0.0)),
        math.log1p(max(_to_float(row.get("following_count")), 0.0)),
        math.log1p(max(_to_float(row.get("tweets_total")), 0.0)),
        math.log1p(max(_to_float(row.get("post_type_tweet_count")), 0.0)),
        math.log1p(max(_to_float(row.get("triplet_tweet_count")), 0.0)),
        1.0 if _to_float(row.get("verified")) > 0.0 else 0.0,
        _to_float(row.get("tweets_with_created_at_ratio")),
        _to_float(row.get("tweets_with_public_metrics_ratio")),
        _to_float(row.get("tweets_with_references_ratio")),
        _to_float(row.get("tweets_with_external_url_ratio")),
    ]


def _build_behavior_distribution_vector(row: dict[str, Any]) -> list[float] | None:
    distribution = [_to_float(row.get(field)) for field in BEHAVIOR_DISTRIBUTION_FIELDS]
    if sum(max(value, 0.0) for value in distribution) <= 0.0:
        return None
    return distribution


def _build_temporal_bundle(
    user_ids: list[str],
    temporal_rows: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    distributions: dict[str, list[float]] = {}
    ready: dict[str, bool] = {}
    for user_id in user_ids:
        row = temporal_rows.get(user_id, {})
        distribution = row.get("utc_hour_distribution") or []
        distributions[user_id] = [float(value) for value in distribution] if isinstance(distribution, list) else [0.0] * 24
        ready[user_id] = bool(int(row.get("temporal_ready") or 0))
    return {
        "distribution": distributions,
        "ready": ready,
    }


def _build_network_bundle(
    user_ids: list[str],
    network_rows: dict[str, Any],
) -> dict[str, Any]:
    out_neighbors = network_rows["out_neighbors"]
    in_degree = network_rows["in_degree"]
    out_degree = network_rows["out_degree"]
    degree_vectors = {
        user_id: [
            math.log1p(max(float(in_degree.get(user_id, 0)), 0.0)),
            math.log1p(max(float(out_degree.get(user_id, 0)), 0.0)),
        ]
        for user_id in user_ids
    }
    degree_normed = {
        user_id: _normalize_vector(vector)
        for user_id, vector in degree_vectors.items()
        if _vector_has_signal(vector)
    }
    return {
        "out_neighbors": out_neighbors,
        "in_degree": in_degree,
        "out_degree": out_degree,
        "degree_vectors": degree_vectors,
        "degree_normed": degree_normed,
    }


def _build_candidate_pairs(
    user_ids: list[str],
    content_normed: dict[str, list[float]],
    behavior_scalar_normed: dict[str, list[float]],
    out_neighbors: dict[str, set[str]],
    *,
    backend: str,
    chunk_size: int,
    candidate_k: int,
) -> dict[str, set[str]]:
    candidate_pairs: dict[str, set[str]] = {user_id: set() for user_id in user_ids}

    for pair_map in (
        _top_candidate_pairs(content_normed, candidate_k=candidate_k, backend=backend, chunk_size=chunk_size),
        _top_candidate_pairs(behavior_scalar_normed, candidate_k=candidate_k, backend=backend, chunk_size=chunk_size),
    ):
        for source_user_id, targets in pair_map.items():
            candidate_pairs.setdefault(source_user_id, set()).update(targets)

    for source_user_id, neighbors in out_neighbors.items():
        for target_user_id in neighbors:
            if source_user_id == target_user_id:
                continue
            candidate_pairs.setdefault(source_user_id, set()).add(target_user_id)
            candidate_pairs.setdefault(target_user_id, set()).add(source_user_id)

    return candidate_pairs


def _top_candidate_pairs(
    vector_map: dict[str, list[float]],
    *,
    candidate_k: int,
    backend: str,
    chunk_size: int,
) -> dict[str, set[str]]:
    user_ids = sorted(vector_map)
    if not user_ids or candidate_k <= 0:
        return {}
    vectors = [vector_map[user_id] for user_id in user_ids]
    if backend == "numpy":
        return _top_candidate_pairs_numpy(user_ids, vectors, candidate_k=candidate_k, chunk_size=chunk_size)
    return _top_candidate_pairs_python(user_ids, vectors, candidate_k=candidate_k)


def _top_candidate_pairs_numpy(
    user_ids: list[str],
    vectors: list[list[float]],
    *,
    candidate_k: int,
    chunk_size: int,
) -> dict[str, set[str]]:
    assert _np is not None

    matrix = _np.asarray(vectors, dtype=_np.float32)
    user_count = len(user_ids)
    neighbor_count = min(max(int(candidate_k), 0), max(user_count - 1, 0))
    pairs: dict[str, set[str]] = {user_id: set() for user_id in user_ids}
    if neighbor_count == 0:
        return pairs

    effective_chunk = max(int(chunk_size), 1)
    for start in range(0, user_count, effective_chunk):
        end = min(start + effective_chunk, user_count)
        similarities = matrix[start:end] @ matrix.T
        row_indices = _np.arange(end - start)
        similarities[row_indices, _np.arange(start, end)] = -_np.inf

        top_indices = _np.argpartition(similarities, -neighbor_count, axis=1)[:, -neighbor_count:]
        top_values = _np.take_along_axis(similarities, top_indices, axis=1)
        order = _np.argsort(-top_values, axis=1)
        top_indices = _np.take_along_axis(top_indices, order, axis=1)

        for local_row, source_index in enumerate(range(start, end)):
            source_user_id = user_ids[source_index]
            for neighbor_index in top_indices[local_row]:
                target_user_id = user_ids[int(neighbor_index)]
                if source_user_id != target_user_id:
                    pairs[source_user_id].add(target_user_id)
    return pairs


def _top_candidate_pairs_python(
    user_ids: list[str],
    vectors: list[list[float]],
    *,
    candidate_k: int,
) -> dict[str, set[str]]:
    neighbor_count = min(max(int(candidate_k), 0), max(len(user_ids) - 1, 0))
    pairs: dict[str, set[str]] = {user_id: set() for user_id in user_ids}
    if neighbor_count == 0:
        return pairs

    for source_index, source_user_id in enumerate(user_ids):
        scored_neighbors: list[tuple[float, str]] = []
        for target_index, target_user_id in enumerate(user_ids):
            if source_index == target_index:
                continue
            similarity = sum(left * right for left, right in zip(vectors[source_index], vectors[target_index]))
            scored_neighbors.append((similarity, target_user_id))
        scored_neighbors.sort(key=lambda item: (-item[0], item[1]))
        pairs[source_user_id].update(target for _score, target in scored_neighbors[:neighbor_count])
    return pairs


def _compute_pair_scores(
    *,
    source_user_id: str,
    target_user_id: str,
    content_normed: dict[str, list[float]],
    behavior_bundle: dict[str, Any],
    temporal_bundle: dict[str, Any],
    network_bundle: dict[str, Any],
    weights: dict[str, float],
) -> dict[str, float | None]:
    content_similarity = normalized_cosine_similarity(
        content_normed.get(source_user_id, []),
        content_normed.get(target_user_id, []),
    ) if source_user_id in content_normed and target_user_id in content_normed else None

    behavior_distribution_similarity = None
    if source_user_id in behavior_bundle["distribution"] and target_user_id in behavior_bundle["distribution"]:
        behavior_distribution_similarity = js_similarity(
            behavior_bundle["distribution"][source_user_id],
            behavior_bundle["distribution"][target_user_id],
        )
    behavior_scalar_similarity = None
    if source_user_id in behavior_bundle["scalar_normed"] and target_user_id in behavior_bundle["scalar_normed"]:
        behavior_scalar_similarity = normalized_cosine_similarity(
            behavior_bundle["scalar_normed"][source_user_id],
            behavior_bundle["scalar_normed"][target_user_id],
        )
    behavior_similarity = _combine_channel_scores(
        behavior_distribution_similarity,
        behavior_scalar_similarity,
        primary_weight=0.5,
        secondary_weight=0.5,
    )

    temporal_similarity = None
    if temporal_bundle["ready"].get(source_user_id, False) and temporal_bundle["ready"].get(target_user_id, False):
        temporal_distance = dtw_average_distance(
            temporal_bundle["distribution"].get(source_user_id, [0.0] * 24),
            temporal_bundle["distribution"].get(target_user_id, [0.0] * 24),
        )
        if temporal_distance is not None:
            temporal_similarity = 1.0 / (1.0 + temporal_distance)

    network_jaccard_similarity = jaccard_similarity(
        network_bundle["out_neighbors"].get(source_user_id, set()),
        network_bundle["out_neighbors"].get(target_user_id, set()),
    )
    network_degree_similarity = None
    if source_user_id in network_bundle["degree_normed"] and target_user_id in network_bundle["degree_normed"]:
        network_degree_similarity = normalized_cosine_similarity(
            network_bundle["degree_normed"][source_user_id],
            network_bundle["degree_normed"][target_user_id],
        )
    network_similarity = _combine_channel_scores(
        network_jaccard_similarity,
        network_degree_similarity,
        primary_weight=0.7,
        secondary_weight=0.3,
    )

    fused_weight, available_weight_sum = weighted_available_average(
        {
            "content": content_similarity,
            "behavior": behavior_similarity,
            "temporal": temporal_similarity,
            "network": network_similarity,
        },
        weights,
    )
    return {
        "content_similarity": content_similarity,
        "behavior_similarity": behavior_similarity,
        "temporal_similarity": temporal_similarity,
        "network_similarity": network_similarity,
        "available_weight_sum": available_weight_sum,
        "fused_weight": fused_weight,
    }


def _combine_channel_scores(
    primary_score: float | None,
    secondary_score: float | None,
    *,
    primary_weight: float,
    secondary_weight: float,
) -> float | None:
    if primary_score is not None and secondary_score is not None:
        return (
            primary_weight * float(primary_score) + secondary_weight * float(secondary_score)
        ) / (primary_weight + secondary_weight)
    if primary_score is not None:
        return float(primary_score)
    if secondary_score is not None:
        return float(secondary_score)
    return None


def _resolve_backend(requested_backend: str, user_count: int) -> str:
    backend = (requested_backend or "auto").strip().lower()
    if backend == "auto":
        if _np is not None:
            return "numpy"
        if user_count > MAX_PYTHON_BACKEND_USERS:
            raise RuntimeError(
                "numpy is required for build-user-graph above "
                f"{MAX_PYTHON_BACKEND_USERS} users. Install numpy and rerun."
            )
        return "python"
    if backend == "numpy":
        if _np is None:
            raise RuntimeError("Requested numpy backend, but numpy is not installed.")
        return "numpy"
    if backend == "python":
        return "python"
    raise ValueError(f"Unsupported backend: {requested_backend}")


def _build_directed_edges_numpy(
    user_ids: list[str],
    vectors: list[list[float]],
    *,
    k: int,
    min_similarity: float,
    chunk_size: int,
) -> tuple[list[dict[str, Any]], int]:
    assert _np is not None

    matrix = _np.asarray(vectors, dtype=_np.float32)
    norms = _np.linalg.norm(matrix, axis=1, keepdims=True)
    zero_norm_mask = norms.squeeze(axis=1) <= 0
    norms[zero_norm_mask] = 1.0
    normalized = matrix / norms

    user_count = len(user_ids)
    neighbor_count = min(max(int(k), 0), max(user_count - 1, 0))
    directed_edges: list[dict[str, Any]] = []
    if neighbor_count == 0:
        return directed_edges, int(zero_norm_mask.sum())

    effective_chunk = max(int(chunk_size), 1)
    for start in range(0, user_count, effective_chunk):
        end = min(start + effective_chunk, user_count)
        similarities = normalized[start:end] @ normalized.T
        row_indices = _np.arange(end - start)
        similarities[row_indices, _np.arange(start, end)] = -_np.inf

        top_indices = _np.argpartition(similarities, -neighbor_count, axis=1)[:, -neighbor_count:]
        top_values = _np.take_along_axis(similarities, top_indices, axis=1)
        order = _np.argsort(-top_values, axis=1)
        top_indices = _np.take_along_axis(top_indices, order, axis=1)
        top_values = _np.take_along_axis(top_values, order, axis=1)

        for local_row, source_index in enumerate(range(start, end)):
            rank = 0
            for neighbor_index, similarity in zip(top_indices[local_row], top_values[local_row]):
                similarity_value = float(similarity)
                if not math.isfinite(similarity_value) or similarity_value < min_similarity:
                    continue
                rank += 1
                directed_edges.append(
                    {
                        "source_user_id": user_ids[source_index],
                        "target_user_id": user_ids[int(neighbor_index)],
                        "similarity": round(similarity_value, 8),
                        "rank": rank,
                    }
                )

    return directed_edges, int(zero_norm_mask.sum())


def _build_directed_edges_python(
    user_ids: list[str],
    vectors: list[list[float]],
    *,
    k: int,
    min_similarity: float,
) -> tuple[list[dict[str, Any]], int]:
    normalized_vectors: list[list[float]] = []
    zero_norm_users = 0
    for vector in vectors:
        norm = math.sqrt(sum(value * value for value in vector))
        if norm <= 0:
            zero_norm_users += 1
            normalized_vectors.append([0.0 for _ in vector])
        else:
            normalized_vectors.append([value / norm for value in vector])

    neighbor_count = min(max(int(k), 0), max(len(user_ids) - 1, 0))
    directed_edges: list[dict[str, Any]] = []
    if neighbor_count == 0:
        return directed_edges, zero_norm_users

    for source_index, source_vector in enumerate(normalized_vectors):
        scored_neighbors: list[tuple[float, str]] = []
        for target_index, target_vector in enumerate(normalized_vectors):
            if source_index == target_index:
                continue
            similarity = sum(left * right for left, right in zip(source_vector, target_vector))
            if similarity < min_similarity:
                continue
            scored_neighbors.append((similarity, user_ids[target_index]))
        scored_neighbors.sort(key=lambda item: (-item[0], item[1]))
        for rank, (similarity, target_user_id) in enumerate(scored_neighbors[:neighbor_count], start=1):
            directed_edges.append(
                {
                    "source_user_id": user_ids[source_index],
                    "target_user_id": target_user_id,
                    "similarity": round(float(similarity), 8),
                    "rank": rank,
                }
            )
    return directed_edges, zero_norm_users


def _symmetrize_edges(
    directed_edges: list[dict[str, Any]],
    *,
    mode: str,
) -> list[dict[str, Any]]:
    if mode == "directed":
        return [
            {
                "source_user_id": row["source_user_id"],
                "target_user_id": row["target_user_id"],
                "weight": row["similarity"],
                "support": 1,
            }
            for row in directed_edges
        ]

    aggregated: dict[tuple[str, str], dict[str, Any]] = {}
    for row in directed_edges:
        source_user_id = str(row["source_user_id"])
        target_user_id = str(row["target_user_id"])
        key = tuple(sorted((source_user_id, target_user_id)))
        payload = aggregated.setdefault(
            key,
            {
                "source_user_id": key[0],
                "target_user_id": key[1],
                "weight": float(row["similarity"]),
                "support": 0,
            },
        )
        payload["support"] += 1
        payload["weight"] = max(float(payload["weight"]), float(row["similarity"]))

    rows = []
    for key in sorted(aggregated):
        row = aggregated[key]
        if mode == "mutual_max" and row["support"] < 2:
            continue
        rows.append(
            {
                "source_user_id": row["source_user_id"],
                "target_user_id": row["target_user_id"],
                "weight": round(float(row["weight"]), 8),
                "support": int(row["support"]),
            }
        )
    return rows


def _symmetrize_detailed_edges(
    directed_edges: list[dict[str, Any]],
    *,
    mode: str,
) -> list[dict[str, Any]]:
    aggregated: dict[tuple[str, str], dict[str, Any]] = {}
    for row in directed_edges:
        source_user_id = str(row["source_user_id"])
        target_user_id = str(row["target_user_id"])
        key = tuple(sorted((source_user_id, target_user_id)))
        payload = aggregated.setdefault(
            key,
            {
                "source_user_id": key[0],
                "target_user_id": key[1],
                "content_similarity": row["content_similarity"],
                "behavior_similarity": row["behavior_similarity"],
                "temporal_similarity": row["temporal_similarity"],
                "network_similarity": row["network_similarity"],
                "available_weight_sum": float(row["available_weight_sum"]),
                "fused_weight": float(row["fused_weight"]),
                "support": 0,
            },
        )
        payload["support"] += 1
        payload["fused_weight"] = max(float(payload["fused_weight"]), float(row["fused_weight"]))
        payload["available_weight_sum"] = max(float(payload["available_weight_sum"]), float(row["available_weight_sum"]))

    rows = []
    for key in sorted(aggregated):
        row = aggregated[key]
        if mode == "mutual_max" and row["support"] < 2:
            continue
        rows.append(
            {
                "source_user_id": row["source_user_id"],
                "target_user_id": row["target_user_id"],
                "content_similarity": row["content_similarity"],
                "behavior_similarity": row["behavior_similarity"],
                "temporal_similarity": row["temporal_similarity"],
                "network_similarity": row["network_similarity"],
                "available_weight_sum": round(float(row["available_weight_sum"]), 8),
                "fused_weight": round(float(row["fused_weight"]), 8),
                "support": int(row["support"]),
            }
        )
    return rows


def _compute_degree_summary(
    user_ids: list[str],
    directed_edges: list[dict[str, Any]],
    undirected_edges: list[dict[str, Any]],
) -> dict[str, Any]:
    directed_out_degree: defaultdict[str, int] = defaultdict(int)
    undirected_degree: defaultdict[str, int] = defaultdict(int)

    for row in directed_edges:
        directed_out_degree[str(row["source_user_id"])] += 1
    for row in undirected_edges:
        source_user_id = str(row["source_user_id"])
        target_user_id = str(row["target_user_id"])
        undirected_degree[source_user_id] += 1
        undirected_degree[target_user_id] += 1

    user_count = len(user_ids)
    if user_count == 0:
        return {
            "avg_directed_out_degree": 0.0,
            "avg_undirected_degree": 0.0,
            "isolated_undirected_users": 0,
        }

    directed_values = [directed_out_degree.get(user_id, 0) for user_id in user_ids]
    undirected_values = [undirected_degree.get(user_id, 0) for user_id in user_ids]
    return {
        "avg_directed_out_degree": round(sum(directed_values) / user_count, 8),
        "avg_undirected_degree": round(sum(undirected_values) / user_count, 8),
        "isolated_undirected_users": sum(1 for value in undirected_values if value == 0),
        "max_undirected_degree": max(undirected_values) if undirected_values else 0,
    }


def _float_vector(values: Any) -> list[float]:
    if not isinstance(values, list):
        raise ValueError("Expected fused_vector to be a list")
    return [float(value) for value in values]


def _normalize_vector(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm <= 0.0:
        return [0.0 for _ in vector]
    return [value / norm for value in vector]


def _vector_has_signal(vector: list[float]) -> bool:
    return any(abs(value) > 1e-12 for value in vector)


def _to_float(value: Any) -> float:
    if value in (None, ""):
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _round_or_blank(value: float | None) -> float | str:
    if value is None:
        return ""
    return round(float(value), 8)


def _render_graph_summary(manifest: dict[str, Any]) -> str:
    counts = manifest["counts"]
    degree_summary = manifest["degree_summary"]
    lines = [
        "# User Graph Summary",
        "",
        "## Overall",
        f"- Fusion mode: {manifest.get('fusion_mode', 'early')}",
        f"- Users: {counts['users']}",
        f"- Directed edges: {counts['directed_edges']}",
        f"- Undirected edges: {counts['undirected_edges']}",
        f"- Backend: {manifest['backend']}",
        f"- Metric: {manifest['metric']}",
        f"- k: {manifest['k']}",
        f"- Min similarity: {manifest['min_similarity']}",
    ]
    if manifest.get("fusion_mode") == "late":
        lines.extend(
            [
                f"- Candidate k: {manifest['candidate_k']}",
                f"- Channel weights: {manifest['channel_weights']}",
                "",
                "## Channel Coverage",
            ]
        )
        for channel_name, payload in manifest.get("channel_edge_coverage", {}).items():
            lines.append(
                f"- {channel_name}: directed_edges={payload['directed_edges']}, coverage={payload['coverage']}"
            )
    else:
        lines.extend(
            [
                f"- Vector dim: {manifest['vector_dim']}",
                f"- Zero-norm users: {counts['zero_norm_users']}",
            ]
        )
    lines.extend(
        [
            "",
            "## Degree",
            f"- Avg directed out-degree: {degree_summary['avg_directed_out_degree']}",
            f"- Avg undirected degree: {degree_summary['avg_undirected_degree']}",
            f"- Isolated undirected users: {degree_summary['isolated_undirected_users']}",
            f"- Max undirected degree: {degree_summary['max_undirected_degree']}",
        ]
    )
    return "\n".join(lines) + "\n"

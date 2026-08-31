"""Structural summaries over detected 10k communities."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .readers import read_csv_rows, write_csv, write_json


def analyze_community_structure(
    sample_root: Path,
    communities_root: Path,
    graph_root: Path,
    purity_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Summarize structural differences among discovered communities."""

    assignments = list(read_csv_rows(communities_root / "community_assignments.csv"))
    if not assignments:
        raise ValueError(f"No community assignments found under {communities_root}")

    purity_rows = list(read_csv_rows(purity_root / "community_purity_summary.csv"))
    if not purity_rows:
        raise ValueError(f"No community purity summary found under {purity_root}")

    purity_by_community = {
        str(row.get("community_id") or ""): row
        for row in purity_rows
        if row.get("community_id")
    }
    members_by_community: defaultdict[str, list[str]] = defaultdict(list)
    community_by_user: dict[str, str] = {}
    for row in assignments:
        user_id = str(row.get("user_id") or "")
        community_id = str(row.get("community_id") or "")
        if not user_id or not community_id:
            continue
        members_by_community[community_id].append(user_id)
        community_by_user[user_id] = community_id

    adjacency = _load_adjacency(graph_root / "user_knn_edges.csv")
    edge_counts, internal_degrees = _compute_internal_edge_stats(adjacency, community_by_user)

    summary_rows = []
    for community_id, members in members_by_community.items():
        purity_row = purity_by_community.get(community_id, {})
        community_size = len(members)
        internal_edges = edge_counts.get(community_id, 0)
        density = _density(community_size, internal_edges)
        average_degree = (2.0 * internal_edges / community_size) if community_size else 0.0
        clustering = _average_clustering(members, adjacency)
        predicted_label = str(purity_row.get("predicted_label_by_train_majority") or "")
        bot_ratio = _safe_float(purity_row.get("bot_ratio"))
        purity = _safe_float(purity_row.get("purity"))
        encoding_depth = _safe_float(purity_row.get("encoding_depth"))
        archetype = _assign_archetype(
            community_size=community_size,
            bot_ratio=bot_ratio,
            purity=purity,
            density=density,
            average_degree=average_degree,
        )
        summary_rows.append(
            {
                "community_id": community_id,
                "community_size": community_size,
                "bot_ratio": round(bot_ratio, 8),
                "purity": round(purity, 8),
                "density": round(density, 8),
                "average_degree": round(average_degree, 8),
                "clustering_coefficient": round(clustering, 8),
                "encoding_depth": round(encoding_depth, 8),
                "predicted_label": predicted_label,
                "train_count": int(_safe_float(purity_row.get("train_count"))),
                "valid_count": int(_safe_float(purity_row.get("valid_count"))),
                "test_count": int(_safe_float(purity_row.get("test_count"))),
                "human_count": int(_safe_float(purity_row.get("all_human_count"))),
                "bot_count": int(_safe_float(purity_row.get("all_bot_count"))),
                "internal_edges": internal_edges,
                "avg_internal_degree_from_edges": round(
                    sum(internal_degrees.get(user_id, 0) for user_id in members) / community_size,
                    8,
                )
                if community_size
                else 0.0,
                "archetype": archetype,
            }
        )

    summary_rows.sort(key=lambda row: (-int(row["community_size"]), str(row["community_id"])))
    archetype_counter = Counter(str(row["archetype"]) for row in summary_rows)
    representative_rows = _select_representative_rows(summary_rows)

    output_root.mkdir(parents=True, exist_ok=True)
    summary_csv_path = output_root / "community_structure_summary.csv"
    representatives_path = output_root / "representative_communities.csv"
    manifest_path = output_root / "community_structure_manifest.json"
    markdown_path = output_root / "community_structure_summary.md"

    write_csv(
        summary_csv_path,
        [
            "community_id",
            "community_size",
            "bot_ratio",
            "purity",
            "density",
            "average_degree",
            "clustering_coefficient",
            "encoding_depth",
            "predicted_label",
            "train_count",
            "valid_count",
            "test_count",
            "human_count",
            "bot_count",
            "internal_edges",
            "avg_internal_degree_from_edges",
            "archetype",
        ],
        summary_rows,
    )
    write_csv(
        representatives_path,
        [
            "archetype",
            "selection_rank",
            "community_id",
            "community_size",
            "bot_ratio",
            "purity",
            "density",
            "clustering_coefficient",
            "encoding_depth",
            "predicted_label",
        ],
        representative_rows,
    )

    manifest = {
        "sample_root": str(sample_root),
        "communities_root": str(communities_root),
        "graph_root": str(graph_root),
        "purity_root": str(purity_root),
        "output_root": str(output_root),
        "counts": {
            "communities": len(summary_rows),
            "representative_rows": len(representative_rows),
        },
        "archetype_counts": dict(sorted(archetype_counter.items())),
        "files": {
            "summary_csv": str(summary_csv_path),
            "representative_csv": str(representatives_path),
            "summary_md": str(markdown_path),
        },
    }
    write_json(manifest_path, manifest)
    markdown_path.write_text(_render_structure_summary(manifest, representative_rows), encoding="utf-8")
    return manifest


def _load_adjacency(path: Path) -> dict[str, set[str]]:
    adjacency: dict[str, set[str]] = defaultdict(set)
    for row in read_csv_rows(path):
        source_user_id = str(row.get("source_user_id") or "")
        target_user_id = str(row.get("target_user_id") or "")
        if not source_user_id or not target_user_id or source_user_id == target_user_id:
            continue
        adjacency[source_user_id].add(target_user_id)
        adjacency[target_user_id].add(source_user_id)
    return adjacency


def _compute_internal_edge_stats(
    adjacency: dict[str, set[str]],
    community_by_user: dict[str, str],
) -> tuple[dict[str, int], dict[str, int]]:
    edge_counts: defaultdict[str, int] = defaultdict(int)
    internal_degree: defaultdict[str, int] = defaultdict(int)
    for source_user_id, neighbors in adjacency.items():
        source_community = community_by_user.get(source_user_id, "")
        if not source_community:
            continue
        for target_user_id in neighbors:
            if source_user_id >= target_user_id:
                continue
            if community_by_user.get(target_user_id, "") != source_community:
                continue
            edge_counts[source_community] += 1
            internal_degree[source_user_id] += 1
            internal_degree[target_user_id] += 1
    return edge_counts, internal_degree


def _density(community_size: int, internal_edges: int) -> float:
    if community_size <= 1:
        return 0.0
    possible_edges = community_size * (community_size - 1) / 2.0
    return internal_edges / possible_edges if possible_edges else 0.0


def _average_clustering(members: list[str], adjacency: dict[str, set[str]]) -> float:
    if not members:
        return 0.0
    member_set = set(members)
    total = 0.0
    for user_id in members:
        neighbors = list(adjacency.get(user_id, set()) & member_set)
        degree = len(neighbors)
        if degree < 2:
            continue
        triangles = 0
        for index, neighbor_a in enumerate(neighbors[:-1]):
            neighbor_a_set = adjacency.get(neighbor_a, set())
            for neighbor_b in neighbors[index + 1 :]:
                if neighbor_b in neighbor_a_set:
                    triangles += 1
        total += (2.0 * triangles) / (degree * (degree - 1))
    return total / len(members)


def _assign_archetype(
    *,
    community_size: int,
    bot_ratio: float,
    purity: float,
    density: float,
    average_degree: float,
) -> str:
    if community_size >= 75 and purity >= 0.85 and bot_ratio <= 0.15:
        return "Pure human macro-communities"
    if community_size <= 80 and purity >= 0.7 and bot_ratio >= 0.55:
        return "Compact bot communities"
    if community_size <= 20 and (density < 0.25 or average_degree < 3.5):
        return "Sparse peripheral communities"
    if 0.15 < bot_ratio < 0.55 or purity < 0.85:
        return "Mixed transitional communities"
    return "Mixed transitional communities"


def _select_representative_rows(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in summary_rows:
        grouped[str(row["archetype"])].append(row)

    representatives: list[dict[str, Any]] = []
    for archetype, rows in sorted(grouped.items()):
        ranked = sorted(rows, key=lambda row: _representative_sort_key(archetype, row))
        for rank, row in enumerate(ranked[:3], start=1):
            representatives.append(
                {
                    "archetype": archetype,
                    "selection_rank": rank,
                    "community_id": row["community_id"],
                    "community_size": row["community_size"],
                    "bot_ratio": row["bot_ratio"],
                    "purity": row["purity"],
                    "density": row["density"],
                    "clustering_coefficient": row["clustering_coefficient"],
                    "encoding_depth": row["encoding_depth"],
                    "predicted_label": row["predicted_label"],
                }
            )
    return representatives


def _representative_sort_key(archetype: str, row: dict[str, Any]) -> tuple[Any, ...]:
    if archetype == "Pure human macro-communities":
        return (-int(row["community_size"]), -float(row["purity"]), str(row["community_id"]))
    if archetype == "Compact bot communities":
        return (-float(row["bot_ratio"]), -float(row["purity"]), -float(row["density"]), str(row["community_id"]))
    if archetype == "Sparse peripheral communities":
        return (float(row["density"]), float(row["average_degree"]), int(row["community_size"]), str(row["community_id"]))
    return (
        abs(float(row["bot_ratio"]) - 0.5),
        float(row["purity"]),
        -int(row["community_size"]),
        str(row["community_id"]),
    )


def _safe_float(value: Any) -> float:
    if value in {None, ""}:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _render_structure_summary(manifest: dict[str, Any], representative_rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Community Structure Summary",
        "",
        "## Archetype Counts",
    ]
    for archetype, count in manifest["archetype_counts"].items():
        lines.append(f"- {archetype}: {count}")
    lines.extend(["", "## Representative Communities"])
    for row in representative_rows:
        lines.append(
            f"- {row['archetype']} / rank {row['selection_rank']}: {row['community_id']} "
            f"(size={row['community_size']}, purity={row['purity']}, bot_ratio={row['bot_ratio']}, "
            f"density={row['density']}, predicted={row['predicted_label']})"
        )
    return "\n".join(lines) + "\n"

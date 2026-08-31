#!/usr/bin/env python3
"""Build an unsupervised grouping summary for the 10k partitions."""

from __future__ import annotations

import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path


def load_weighted_graph(path: Path) -> tuple[dict[str, dict[str, float]], dict[str, float], float]:
    adjacency: dict[str, dict[str, float]] = defaultdict(dict)
    weighted_degree: dict[str, float] = defaultdict(float)
    total_edge_weight = 0.0
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            source = str(row.get("source_user_id") or "")
            target = str(row.get("target_user_id") or "")
            if not source or not target or source == target:
                continue
            weight = float(row.get("weight") or 0.0)
            adjacency[source][target] = weight
            adjacency[target][source] = weight
            weighted_degree[source] += weight
            weighted_degree[target] += weight
            total_edge_weight += weight
    return adjacency, weighted_degree, total_edge_weight


def load_assignments(path: Path) -> dict[str, list[str]]:
    members_by_community: dict[str, list[str]] = defaultdict(list)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            user_id = str(row.get("user_id") or "")
            community_id = str(row.get("community_id") or "")
            if user_id and community_id:
                members_by_community[community_id].append(user_id)
    return members_by_community


def average_clustering(members: list[str], adjacency: dict[str, dict[str, float]]) -> float:
    if not members:
        return 0.0
    member_set = set(members)
    total = 0.0
    for user_id in members:
        neighbors = [neighbor for neighbor in adjacency.get(user_id, {}) if neighbor in member_set]
        degree = len(neighbors)
        if degree < 2:
            continue
        triangles = 0
        for index, neighbor_a in enumerate(neighbors[:-1]):
            neighbor_a_neighbors = adjacency.get(neighbor_a, {})
            for neighbor_b in neighbors[index + 1 :]:
                if neighbor_b in neighbor_a_neighbors and neighbor_b in member_set:
                    triangles += 1
        total += (2.0 * triangles) / (degree * (degree - 1))
    return total / len(members)


def summarize_partition(
    members_by_community: dict[str, list[str]],
    adjacency: dict[str, dict[str, float]],
    weighted_degree: dict[str, float],
    total_edge_weight: float,
) -> dict[str, float]:
    sizes = [len(users) for users in members_by_community.values()]
    total_users = sum(sizes)

    structural_entropy = 0.0
    weighted_modularity = 0.0
    weighted_density = 0.0
    weighted_clustering = 0.0
    weighted_conductance = 0.0

    graph_volume = 2.0 * total_edge_weight

    for users in members_by_community.values():
        user_set = set(users)
        internal_weight = 0.0
        cut_weight = 0.0
        internal_edges = 0
        volume = 0.0
        for user_id in users:
            volume += weighted_degree.get(user_id, 0.0)
            for neighbor_id, weight in adjacency.get(user_id, {}).items():
                if neighbor_id in user_set:
                    if user_id < neighbor_id:
                        internal_weight += weight
                        internal_edges += 1
                else:
                    cut_weight += weight

        community_size = len(users)
        possible_edges = community_size * (community_size - 1) / 2.0
        density = (internal_edges / possible_edges) if possible_edges else 0.0
        clustering = average_clustering(users, adjacency)
        remainder = graph_volume - volume
        conductance = cut_weight / min(volume, remainder) if min(volume, remainder) > 0 else 0.0

        weighted_density += density * community_size
        weighted_clustering += clustering * community_size
        weighted_conductance += conductance * community_size

        if volume > 0 and graph_volume > 0:
            structural_entropy += -(cut_weight / graph_volume) * math.log2(volume / graph_volume)
            structural_entropy += (volume / graph_volume) * math.log2(volume)
            weighted_modularity += (internal_weight / total_edge_weight) - (volume / graph_volume) ** 2

    return {
        "communities": len(sizes),
        "largest_community": max(sizes) if sizes else 0,
        "median_community": statistics.median(sizes) if sizes else 0.0,
        "structural_entropy": round(structural_entropy, 8),
        "weighted_modularity": round(weighted_modularity, 8),
        "weighted_mean_density": round(weighted_density / total_users, 8) if total_users else 0.0,
        "weighted_mean_clustering": round(weighted_clustering / total_users, 8) if total_users else 0.0,
        "weighted_mean_conductance": round(weighted_conductance / total_users, 8) if total_users else 0.0,
    }


def write_outputs(output_root: Path, rows: list[dict[str, object]]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    csv_path = output_root / "grouping_unsupervised_results.csv"
    md_path = output_root / "grouping_unsupervised_results.md"

    fieldnames = [
        "method",
        "communities",
        "largest_community",
        "median_community",
        "structural_entropy",
        "weighted_modularity",
        "weighted_mean_density",
        "weighted_mean_clustering",
        "weighted_mean_conductance",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# 10k Unsupervised Grouping Summary",
        "",
        "| Method | Communities | Largest | Median | Structural Entropy | Weighted Modularity | Weighted Mean Density | Weighted Mean Clustering | Weighted Mean Conductance |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['method']} | {row['communities']} | {row['largest_community']} | {row['median_community']} | "
            f"{row['structural_entropy']} | {row['weighted_modularity']} | {row['weighted_mean_density']} | "
            f"{row['weighted_mean_clustering']} | {row['weighted_mean_conductance']} |"
        )
    md_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    analysis_root = Path("/Users/interpy/Downloads/GhostWriting/data/samples/final_v1/analysis")
    adjacency, weighted_degree, total_edge_weight = load_weighted_graph(
        analysis_root / "run_10k_late/graph/user_knn_edges.csv"
    )

    methods = {
        "K-Means": analysis_root / "grouping_baselines_10k/kmeans/community_assignments.csv",
        "Weighted LPA": analysis_root / "grouping_baselines_10k/weighted_lpa/communities/community_assignments.csv",
        "Structural Entropy (Ours)": analysis_root / "run_10k_late/communities/community_assignments.csv",
    }

    rows = []
    for method_name, assignment_path in methods.items():
        summary = summarize_partition(
            load_assignments(assignment_path),
            adjacency,
            weighted_degree,
            total_edge_weight,
        )
        summary["method"] = method_name
        rows.append(summary)

    write_outputs(analysis_root / "grouping_baselines_10k/summary", rows)


if __name__ == "__main__":
    main()

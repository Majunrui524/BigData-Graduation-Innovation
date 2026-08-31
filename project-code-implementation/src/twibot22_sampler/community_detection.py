"""Community detection on exported user similarity graphs."""

from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any

from .readers import read_csv_rows, read_jsonl_records, write_csv, write_json
from .structural_entropy import (
    build_flat_encoding_tree,
    compute_partition_entropy,
    detect_structural_entropy_communities,
)

DEFAULT_COMMUNITY_ALGORITHM = "structural_entropy"
DEFAULT_COMMUNITY_MAX_ITERATIONS = 50
DEFAULT_COMMUNITY_MIN_SIZE = 1
DEFAULT_COMMUNITY_MUTUAL_SUPPORT_BONUS = 0.1


def detect_communities(
    sample_root: Path,
    graph_root: Path,
    output_root: Path,
    *,
    algorithm: str = DEFAULT_COMMUNITY_ALGORITHM,
    max_iterations: int = DEFAULT_COMMUNITY_MAX_ITERATIONS,
    min_community_size: int = DEFAULT_COMMUNITY_MIN_SIZE,
    seed: int = 42,
    mutual_support_bonus: float = DEFAULT_COMMUNITY_MUTUAL_SUPPORT_BONUS,
) -> dict[str, Any]:
    """Detect user communities from an exported undirected similarity graph."""

    if algorithm not in {"structural_entropy", "weighted_lpa"}:
        raise ValueError(f"Unsupported community algorithm: {algorithm}")

    user_ids = _load_user_ids(sample_root)
    split_by_user = {
        str(row.get("id") or row.get("user_id") or ""): str(row.get("split") or "")
        for row in read_csv_rows(sample_root / "split.csv")
        if row.get("id") or row.get("user_id")
    }
    label_by_user = {
        str(row.get("id") or row.get("user_id") or ""): str(row.get("label") or "")
        for row in read_csv_rows(sample_root / "label.csv")
        if row.get("id") or row.get("user_id")
    }
    adjacency, edge_count = _load_graph_edges(
        graph_root / "user_knn_edges.csv",
        user_ids=user_ids,
        mutual_support_bonus=mutual_support_bonus,
    )
    identity_labels = {user_id: user_id for user_id in user_ids}
    initial_entropy = compute_partition_entropy(identity_labels, adjacency)

    if algorithm == "structural_entropy":
        structural_payload = detect_structural_entropy_communities(user_ids, adjacency)
        labels = structural_payload["labels"]
        iterations_run = int(structural_payload["merge_count"])
        merge_count = int(structural_payload["merge_count"])
        encoding_tree = structural_payload["encoding_tree"]
        tree_depth = int(structural_payload["tree_depth"])
        final_entropy = float(structural_payload["final_entropy"])
        initial_entropy = float(structural_payload["initial_entropy"])
    else:
        labels, iterations_run = _run_weighted_label_propagation(
            user_ids,
            adjacency,
            max_iterations=max_iterations,
            seed=seed,
        )
        merge_count = 0
        encoding_tree = build_flat_encoding_tree(labels)
        tree_depth = _encoding_tree_depth(encoding_tree)
        final_entropy = compute_partition_entropy(labels, adjacency)

    post_min_merge_applied = False
    if min_community_size > 1:
        merged_labels = _merge_small_communities(
            labels,
            adjacency,
            min_community_size=min_community_size,
            seed=seed,
        )
        if merged_labels != labels:
            post_min_merge_applied = True
            labels = merged_labels
            final_entropy = compute_partition_entropy(labels, adjacency)
            encoding_tree = build_flat_encoding_tree(labels)
            tree_depth = _encoding_tree_depth(encoding_tree)

    communities = _build_communities(labels)
    community_id_by_label = _assign_community_ids(communities)
    community_sizes = {
        community_id_by_label[label]: len(members)
        for label, members in communities.items()
    }

    assignments = []
    for user_id in sorted(user_ids):
        label = labels[user_id]
        community_id = community_id_by_label[label]
        assignments.append(
            {
                "user_id": user_id,
                "community_id": community_id,
                "community_size": community_sizes[community_id],
                "split": split_by_user.get(user_id, ""),
                "label": label_by_user.get(user_id, ""),
            }
        )

    summary_rows = _build_community_summary_rows(
        assignments,
        communities,
        community_id_by_label,
    )

    output_root.mkdir(parents=True, exist_ok=True)
    assignments_path = output_root / "community_assignments.csv"
    summary_path = output_root / "community_summary.csv"
    encoding_tree_path = output_root / "encoding_tree.json"
    manifest_path = output_root / "community_manifest.json"
    markdown_path = output_root / "community_summary.md"

    write_csv(
        assignments_path,
        ["user_id", "community_id", "community_size", "split", "label"],
        assignments,
    )
    write_csv(
        summary_path,
        [
            "community_id",
            "community_size",
            "human_count",
            "bot_count",
            "unknown_label_count",
            "bot_ratio",
            "train_count",
            "valid_count",
            "test_count",
        ],
        summary_rows,
    )
    write_json(encoding_tree_path, encoding_tree)

    sizes = [len(members) for members in communities.values()]
    manifest = {
        "sample_root": str(sample_root),
        "graph_root": str(graph_root),
        "output_root": str(output_root),
        "algorithm": algorithm,
        "seed": seed,
        "max_iterations": max_iterations,
        "iterations_run": iterations_run,
        "min_community_size": max(int(min_community_size), 1),
        "mutual_support_bonus": float(mutual_support_bonus),
        "initial_entropy": round(float(initial_entropy), 12),
        "final_entropy": round(float(final_entropy), 12),
        "merge_count": int(merge_count),
        "tree_depth": int(tree_depth),
        "post_min_community_merge": int(post_min_merge_applied),
        "counts": {
            "users": len(user_ids),
            "graph_edges": edge_count,
            "communities": len(communities),
            "singleton_communities": sum(1 for size in sizes if size == 1),
        },
        "size_summary": {
            "largest_community": max(sizes) if sizes else 0,
            "smallest_community": min(sizes) if sizes else 0,
            "median_community": float(median(sizes)) if sizes else 0.0,
            "avg_community_size": round(sum(sizes) / len(sizes), 8) if sizes else 0.0,
        },
        "files": {
            "assignments": str(assignments_path),
            "summary_csv": str(summary_path),
            "encoding_tree": str(encoding_tree_path),
            "summary_md": str(markdown_path),
        },
    }
    write_json(manifest_path, manifest)
    markdown_path.write_text(_render_community_summary(manifest, summary_rows), encoding="utf-8")
    return manifest


def _load_user_ids(sample_root: Path) -> list[str]:
    user_ids = []
    for row in read_jsonl_records(sample_root / "user.jsonl"):
        user_id = str(row.get("id") or row.get("user_id") or "")
        if user_id:
            user_ids.append(user_id)
    user_ids.sort()
    return user_ids


def _load_graph_edges(
    path: Path,
    *,
    user_ids: list[str],
    mutual_support_bonus: float,
) -> tuple[dict[str, dict[str, float]], int]:
    adjacency: dict[str, dict[str, float]] = {user_id: {} for user_id in user_ids}
    edge_count = 0
    for row in read_csv_rows(path):
        source_user_id = str(row.get("source_user_id") or "")
        target_user_id = str(row.get("target_user_id") or "")
        if not source_user_id or not target_user_id:
            continue
        weight = float(row.get("weight") or 0.0)
        support = int(float(row.get("support") or 1))
        effective_weight = weight * (1.0 + max(support - 1, 0) * max(mutual_support_bonus, 0.0))
        adjacency.setdefault(source_user_id, {})[target_user_id] = effective_weight
        adjacency.setdefault(target_user_id, {})[source_user_id] = effective_weight
        edge_count += 1
    return adjacency, edge_count


def _run_weighted_label_propagation(
    user_ids: list[str],
    adjacency: dict[str, dict[str, float]],
    *,
    max_iterations: int,
    seed: int,
) -> tuple[dict[str, str], int]:
    labels = {user_id: user_id for user_id in user_ids}
    iterations_run = 0
    for iteration in range(1, max(int(max_iterations), 1) + 1):
        iterations_run = iteration
        changed = 0
        for user_id in _ordered_nodes_for_iteration(user_ids, adjacency, seed=seed, iteration=iteration):
            neighbors = adjacency.get(user_id, {})
            if not neighbors:
                continue
            scores: defaultdict[str, float] = defaultdict(float)
            for neighbor_id, weight in neighbors.items():
                scores[labels[neighbor_id]] += float(weight)
            current_label = labels[user_id]
            current_score = scores.get(current_label, 0.0)
            best_score = max(scores.values())
            if current_score >= best_score:
                continue
            best_labels = [
                label
                for label, score in scores.items()
                if math.isclose(score, best_score) or score == best_score
            ]
            selected_label = _select_label(best_labels, seed=seed, iteration=iteration, user_id=user_id)
            if selected_label != current_label:
                labels[user_id] = selected_label
                changed += 1
        if changed == 0:
            break
    return labels, iterations_run


def _ordered_nodes_for_iteration(
    user_ids: list[str],
    adjacency: dict[str, dict[str, float]],
    *,
    seed: int,
    iteration: int,
) -> list[str]:
    return sorted(
        user_ids,
        key=lambda user_id: (
            -len(adjacency.get(user_id, {})),
            _stable_rank(seed, iteration, user_id),
            user_id,
        ),
    )


def _select_label(labels: list[str], *, seed: int, iteration: int, user_id: str) -> str:
    return min(labels, key=lambda label: (_stable_rank(seed, iteration, f"{user_id}:{label}"), label))


def _stable_rank(seed: int, iteration: int, value: str) -> str:
    payload = f"{seed}:{iteration}:{value}".encode("utf-8")
    return hashlib.sha1(payload).hexdigest()


def _merge_small_communities(
    labels: dict[str, str],
    adjacency: dict[str, dict[str, float]],
    *,
    min_community_size: int,
    seed: int,
) -> dict[str, str]:
    updated = dict(labels)
    community_sizes = Counter(updated.values())
    small_labels = {label for label, size in community_sizes.items() if size < min_community_size}
    if not small_labels:
        return updated

    for user_id in sorted(updated, key=lambda node: (_stable_rank(seed, 9999, node), node)):
        current_label = updated[user_id]
        if current_label not in small_labels:
            continue
        neighbor_scores: defaultdict[str, float] = defaultdict(float)
        for neighbor_id, weight in adjacency.get(user_id, {}).items():
            neighbor_label = updated[neighbor_id]
            if community_sizes[neighbor_label] >= min_community_size:
                neighbor_scores[neighbor_label] += float(weight)
        if not neighbor_scores:
            continue
        best_score = max(neighbor_scores.values())
        best_labels = [
            label
            for label, score in neighbor_scores.items()
            if math.isclose(score, best_score) or score == best_score
        ]
        selected_label = min(best_labels, key=lambda label: (_stable_rank(seed, 10000, f"{user_id}:{label}"), label))
        community_sizes[current_label] -= 1
        updated[user_id] = selected_label
        community_sizes[selected_label] += 1
    return updated


def _build_communities(labels: dict[str, str]) -> dict[str, list[str]]:
    communities: defaultdict[str, list[str]] = defaultdict(list)
    for user_id, label in labels.items():
        communities[label].append(user_id)
    for members in communities.values():
        members.sort()
    return dict(communities)


def _assign_community_ids(communities: dict[str, list[str]]) -> dict[str, str]:
    ordered = sorted(
        communities.items(),
        key=lambda item: (-len(item[1]), item[1][0] if item[1] else item[0]),
    )
    return {label: f"c{index:04d}" for index, (label, _members) in enumerate(ordered, start=1)}


def _build_community_summary_rows(
    assignments: list[dict[str, Any]],
    communities: dict[str, list[str]],
    community_id_by_label: dict[str, str],
) -> list[dict[str, Any]]:
    rows = []
    assignment_by_community: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in assignments:
        assignment_by_community[str(row["community_id"])].append(row)

    for label, members in sorted(
        communities.items(),
        key=lambda item: (-len(item[1]), item[1][0] if item[1] else item[0]),
    ):
        community_id = community_id_by_label[label]
        user_rows = assignment_by_community[community_id]
        label_counter = Counter(str(row.get("label") or "") for row in user_rows)
        split_counter = Counter(str(row.get("split") or "") for row in user_rows)
        human_count = label_counter.get("human", 0)
        bot_count = label_counter.get("bot", 0)
        labeled_total = human_count + bot_count
        rows.append(
            {
                "community_id": community_id,
                "community_size": len(members),
                "human_count": human_count,
                "bot_count": bot_count,
                "unknown_label_count": len(members) - labeled_total,
                "bot_ratio": round(bot_count / labeled_total, 8) if labeled_total else 0.0,
                "train_count": split_counter.get("train", 0),
                "valid_count": split_counter.get("valid", 0),
                "test_count": split_counter.get("test", 0),
            }
        )
    return rows


def _encoding_tree_depth(tree: dict[str, Any]) -> int:
    nodes = tree.get("nodes") if isinstance(tree, dict) else {}
    roots = tree.get("roots") if isinstance(tree, dict) else []
    if not isinstance(nodes, dict) or not isinstance(roots, list):
        return 0

    def _depth(node_id: str) -> int:
        payload = nodes.get(node_id, {})
        children = payload.get("children") if isinstance(payload, dict) else []
        if not isinstance(children, list) or not children:
            return 1
        return 1 + max(_depth(str(child_id)) for child_id in children)

    return max((_depth(str(root_id)) for root_id in roots), default=0)


def _render_community_summary(manifest: dict[str, Any], summary_rows: list[dict[str, Any]]) -> str:
    counts = manifest["counts"]
    size_summary = manifest["size_summary"]
    lines = [
        "# Community Summary",
        "",
        "## Overall",
        f"- Users: {counts['users']}",
        f"- Graph edges: {counts['graph_edges']}",
        f"- Communities: {counts['communities']}",
        f"- Singleton communities: {counts['singleton_communities']}",
        f"- Algorithm: {manifest['algorithm']}",
        f"- Iterations: {manifest['iterations_run']}",
        f"- Merge count: {manifest['merge_count']}",
        f"- Initial entropy: {manifest['initial_entropy']}",
        f"- Final entropy: {manifest['final_entropy']}",
        f"- Tree depth: {manifest['tree_depth']}",
        f"- Largest community: {size_summary['largest_community']}",
        f"- Smallest community: {size_summary['smallest_community']}",
        f"- Median community: {size_summary['median_community']}",
        f"- Avg community size: {size_summary['avg_community_size']}",
        "",
        "## Top Communities",
    ]
    for row in summary_rows[:10]:
        lines.append(
            f"- {row['community_id']}: size={row['community_size']}, "
            f"bot_ratio={row['bot_ratio']}, human={row['human_count']}, bot={row['bot_count']}"
        )
    return "\n".join(lines) + "\n"

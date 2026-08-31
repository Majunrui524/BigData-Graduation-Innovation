"""Structural-entropy community detection on sparse weighted graphs."""

from __future__ import annotations

import heapq
import math
from collections import defaultdict
from typing import Any


def detect_structural_entropy_communities(
    user_ids: list[str],
    adjacency: dict[str, dict[str, float]],
) -> dict[str, Any]:
    """Greedily merge adjacent communities while structural entropy decreases."""

    total_volume = sum(sum(neighbors.values()) for neighbors in adjacency.values())
    if total_volume <= 0.0:
        labels = {user_id: user_id for user_id in user_ids}
        encoding_tree = _build_flat_encoding_tree(user_ids)
        return {
            "labels": labels,
            "initial_entropy": 0.0,
            "final_entropy": 0.0,
            "merge_count": 0,
            "encoding_tree": encoding_tree,
            "tree_depth": 1 if user_ids else 0,
        }

    active = set(user_ids)
    members: dict[str, set[str]] = {user_id: {user_id} for user_id in user_ids}
    volume: dict[str, float] = {
        user_id: sum(float(weight) for weight in adjacency.get(user_id, {}).values())
        for user_id in user_ids
    }
    internal_weight: dict[str, float] = {user_id: 0.0 for user_id in user_ids}
    community_adjacency: dict[str, dict[str, float]] = {
        user_id: {
            neighbor_id: float(weight)
            for neighbor_id, weight in adjacency.get(user_id, {}).items()
            if neighbor_id != user_id
        }
        for user_id in user_ids
    }

    tree_nodes: dict[str, dict[str, Any]] = {}
    node_for_community: dict[str, str] = {}
    depth_by_node: dict[str, int] = {}
    for user_id in user_ids:
        node_id = f"leaf:{user_id}"
        tree_nodes[node_id] = {
            "node_id": node_id,
            "type": "leaf",
            "user_id": user_id,
            "size": 1,
        }
        node_for_community[user_id] = node_id
        depth_by_node[node_id] = 1

    def community_term(community_id: str) -> float:
        return _community_entropy_term(
            volume=volume.get(community_id, 0.0),
            internal_weight=internal_weight.get(community_id, 0.0),
            total_volume=total_volume,
        )

    initial_objective = sum(community_term(community_id) for community_id in active)
    current_objective = initial_objective
    heap: list[tuple[float, str, str]] = []

    for source_user_id in user_ids:
        for target_user_id in community_adjacency.get(source_user_id, {}):
            if source_user_id >= target_user_id:
                continue
            delta = _merge_delta(
                source_user_id,
                target_user_id,
                volume=volume,
                internal_weight=internal_weight,
                community_adjacency=community_adjacency,
                total_volume=total_volume,
            )
            if delta > 0.0:
                heapq.heappush(heap, (-delta, source_user_id, target_user_id))

    merge_count = 0
    while heap:
        negative_delta, left_id, right_id = heapq.heappop(heap)
        if left_id not in active or right_id not in active:
            continue
        refreshed_delta = _merge_delta(
            left_id,
            right_id,
            volume=volume,
            internal_weight=internal_weight,
            community_adjacency=community_adjacency,
            total_volume=total_volume,
        )
        if refreshed_delta <= 1e-12:
            continue
        if abs(refreshed_delta + negative_delta) > 1e-9:
            heapq.heappush(heap, (-refreshed_delta, left_id, right_id))
            continue

        merge_count += 1
        new_community_id = f"merge:{merge_count:05d}"
        bridge_weight = float(community_adjacency.get(left_id, {}).get(right_id, 0.0))
        merged_members = members[left_id] | members[right_id]
        members[new_community_id] = merged_members
        volume[new_community_id] = volume[left_id] + volume[right_id]
        internal_weight[new_community_id] = internal_weight[left_id] + internal_weight[right_id] + bridge_weight

        new_neighbors: dict[str, float] = {}
        neighbor_ids = (set(community_adjacency.get(left_id, {})) | set(community_adjacency.get(right_id, {}))) - {
            left_id,
            right_id,
        }
        for neighbor_id in neighbor_ids:
            weight = float(community_adjacency.get(left_id, {}).get(neighbor_id, 0.0))
            weight += float(community_adjacency.get(right_id, {}).get(neighbor_id, 0.0))
            if weight > 0.0:
                new_neighbors[neighbor_id] = weight
        community_adjacency[new_community_id] = new_neighbors

        for neighbor_id in neighbor_ids:
            payload = community_adjacency.get(neighbor_id, {})
            left_weight = float(payload.pop(left_id, 0.0))
            right_weight = float(payload.pop(right_id, 0.0))
            merged_weight = left_weight + right_weight
            if merged_weight > 0.0:
                payload[new_community_id] = merged_weight

        current_objective -= refreshed_delta
        active.remove(left_id)
        active.remove(right_id)
        active.add(new_community_id)

        left_node_id = node_for_community[left_id]
        right_node_id = node_for_community[right_id]
        node_id = f"node:{merge_count:05d}"
        node_depth = 1 + max(depth_by_node[left_node_id], depth_by_node[right_node_id])
        tree_nodes[node_id] = {
            "node_id": node_id,
            "type": "merge",
            "children": [left_node_id, right_node_id],
            "size": len(merged_members),
            "merge_gain": round(refreshed_delta, 12),
            "objective_after": round(current_objective, 12),
        }
        node_for_community[new_community_id] = node_id
        depth_by_node[node_id] = node_depth

        for community_id in (left_id, right_id):
            members.pop(community_id, None)
            volume.pop(community_id, None)
            internal_weight.pop(community_id, None)
            community_adjacency.pop(community_id, None)
            node_for_community.pop(community_id, None)

        for neighbor_id in neighbor_ids:
            delta = _merge_delta(
                new_community_id,
                neighbor_id,
                volume=volume,
                internal_weight=internal_weight,
                community_adjacency=community_adjacency,
                total_volume=total_volume,
            )
            if delta > 0.0:
                first_id, second_id = sorted((new_community_id, neighbor_id))
                heapq.heappush(heap, (-delta, first_id, second_id))

    labels: dict[str, str] = {}
    for community_id in active:
        for user_id in members[community_id]:
            labels[user_id] = community_id

    root_nodes = [node_for_community[community_id] for community_id in sorted(active)]
    encoding_tree = {
        "algorithm": "structural_entropy",
        "roots": root_nodes,
        "nodes": tree_nodes,
    }
    tree_depth = max((depth_by_node[root_node] for root_node in root_nodes), default=0)
    identity_labels = {user_id: user_id for user_id in user_ids}
    return {
        "labels": labels,
        "initial_entropy": compute_partition_entropy(identity_labels, adjacency),
        "final_entropy": compute_partition_entropy(labels, adjacency),
        "merge_count": merge_count,
        "encoding_tree": encoding_tree,
        "tree_depth": tree_depth,
    }


def compute_partition_entropy(
    labels: dict[str, str],
    adjacency: dict[str, dict[str, float]],
) -> float:
    """Compute structural entropy for a flat partition."""

    total_volume = sum(sum(neighbors.values()) for neighbors in adjacency.values())
    if total_volume <= 0.0:
        return 0.0

    communities: defaultdict[str, set[str]] = defaultdict(set)
    for user_id, community_id in labels.items():
        communities[str(community_id)].add(user_id)

    entropy = 0.0
    for members in communities.values():
        volume_value = 0.0
        internal_value = 0.0
        for user_id in members:
            neighbors = adjacency.get(user_id, {})
            volume_value += sum(neighbors.values())
            for neighbor_id, weight in neighbors.items():
                if neighbor_id in members and user_id < neighbor_id:
                    internal_value += float(weight)
        entropy += _community_entropy_term(
            volume=volume_value,
            internal_weight=internal_value,
            total_volume=total_volume,
        )
    return round(entropy, 12)


def build_flat_encoding_tree(labels: dict[str, str]) -> dict[str, Any]:
    """Build a trivial encoding tree from a flat community assignment."""

    communities: defaultdict[str, list[str]] = defaultdict(list)
    for user_id, community_id in labels.items():
        communities[str(community_id)].append(user_id)
    nodes: dict[str, dict[str, Any]] = {}
    roots: list[str] = []
    for community_id in sorted(communities):
        community_node_id = f"community:{community_id}"
        child_nodes = []
        for user_id in sorted(communities[community_id]):
            leaf_id = f"leaf:{user_id}"
            nodes[leaf_id] = {
                "node_id": leaf_id,
                "type": "leaf",
                "user_id": user_id,
                "size": 1,
            }
            child_nodes.append(leaf_id)
        nodes[community_node_id] = {
            "node_id": community_node_id,
            "type": "community",
            "children": child_nodes,
            "size": len(child_nodes),
        }
        roots.append(community_node_id)
    return {
        "algorithm": "flat_partition",
        "roots": roots,
        "nodes": nodes,
    }


def _merge_delta(
    left_id: str,
    right_id: str,
    *,
    volume: dict[str, float],
    internal_weight: dict[str, float],
    community_adjacency: dict[str, dict[str, float]],
    total_volume: float,
) -> float:
    bridge_weight = float(community_adjacency.get(left_id, {}).get(right_id, 0.0))
    if bridge_weight <= 0.0:
        return 0.0
    left_term = _community_entropy_term(
        volume=volume[left_id],
        internal_weight=internal_weight[left_id],
        total_volume=total_volume,
    )
    right_term = _community_entropy_term(
        volume=volume[right_id],
        internal_weight=internal_weight[right_id],
        total_volume=total_volume,
    )
    merged_term = _community_entropy_term(
        volume=volume[left_id] + volume[right_id],
        internal_weight=internal_weight[left_id] + internal_weight[right_id] + bridge_weight,
        total_volume=total_volume,
    )
    return left_term + right_term - merged_term


def _community_entropy_term(
    *,
    volume: float,
    internal_weight: float,
    total_volume: float,
) -> float:
    if total_volume <= 0.0 or volume <= 0.0:
        return 0.0
    cut = max(volume - 2.0 * internal_weight, 0.0)
    ratio = volume / total_volume
    if ratio <= 0.0 or ratio > 1.0:
        return 0.0
    first_term = 0.0
    if cut > 0.0:
        first_term = -(cut / total_volume) * math.log2(ratio)
    second_term = (volume / total_volume) * math.log2(volume)
    return first_term + second_term


def _build_flat_encoding_tree(user_ids: list[str]) -> dict[str, Any]:
    return build_flat_encoding_tree({user_id: user_id for user_id in user_ids})

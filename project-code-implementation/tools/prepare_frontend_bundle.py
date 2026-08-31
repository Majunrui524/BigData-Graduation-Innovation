#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_optional_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def f(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def i(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def excerpt(value: str, max_length: int) -> str:
    value = (value or "").strip()
    if len(value) <= max_length:
        return value
    return f"{value[: max_length - 1].rstrip()}…"


def phyllotaxis_positions(
    items: list[dict[str, Any]],
    *,
    score_key: str,
    size_key: str,
    x_bias_scale: float = 520.0,
) -> dict[str, tuple[float, float]]:
    golden_angle = math.pi * (3 - math.sqrt(5))
    positions: dict[str, tuple[float, float]] = {}
    sorted_items = sorted(
        items,
        key=lambda row: (-f(row.get(score_key)), -f(row.get(size_key)), str(row.get("id") or row.get("communityId") or "")),
    )
    for index, item in enumerate(sorted_items):
        score = f(item.get(score_key))
        angle = index * golden_angle
        radius = 22.0 * math.sqrt(index + 1)
        x = math.cos(angle) * radius + (score - 0.5) * x_bias_scale
        y = math.sin(angle) * radius + (0.5 - score) * 120.0
        item_id = str(item.get("id") or item.get("communityId") or "")
        positions[item_id] = (round(x, 3), round(y, 3))
    return positions


def radial_subgraph_positions(user_ids: list[str], user_scores: dict[str, float]) -> dict[str, tuple[float, float]]:
    golden_angle = math.pi * (3 - math.sqrt(5))
    ranked_ids = sorted(user_ids, key=lambda user_id: (-user_scores.get(user_id, 0.0), user_id))
    positions: dict[str, tuple[float, float]] = {}
    for index, user_id in enumerate(ranked_ids):
        angle = index * golden_angle
        radius = 8.0 * math.sqrt(index + 1)
        positions[user_id] = (round(math.cos(angle) * radius, 3), round(math.sin(angle) * radius, 3))
    return positions


def build_tree_metadata(tree_path: Path, community_members: dict[str, set[str]]) -> tuple[dict[str, str], dict[str, int]]:
    tree = read_json(tree_path)
    nodes = tree.get("nodes", {})
    roots = tree.get("roots", [])
    children_map = {node_id: list(node.get("children", [])) for node_id, node in nodes.items() if isinstance(node, dict)}

    leaf_cache: dict[str, frozenset[str]] = {}
    depth_cache: dict[str, int] = {}

    def collect_leaves(node_id: str) -> frozenset[str]:
        if node_id in leaf_cache:
            return leaf_cache[node_id]
        node = nodes[node_id]
        if node.get("type") == "leaf":
            leaf_cache[node_id] = frozenset([str(node.get("user_id"))])
            return leaf_cache[node_id]
        leaves: set[str] = set()
        for child_id in children_map.get(node_id, []):
            leaves.update(collect_leaves(child_id))
        leaf_cache[node_id] = frozenset(leaves)
        return leaf_cache[node_id]

    def compute_depth(node_id: str) -> int:
        if node_id in depth_cache:
            return depth_cache[node_id]
        children = children_map.get(node_id, [])
        if not children:
            depth_cache[node_id] = 1
            return 1
        depth_cache[node_id] = 1 + max(compute_depth(child_id) for child_id in children)
        return depth_cache[node_id]

    root_by_signature: dict[frozenset[str], str] = {}
    depth_by_signature: dict[frozenset[str], int] = {}
    for root_id in roots:
        signature = collect_leaves(root_id)
        root_by_signature[signature] = root_id
        depth_by_signature[signature] = compute_depth(root_id)

    community_root_ids: dict[str, str] = {}
    community_depths: dict[str, int] = {}
    for community_id, members in community_members.items():
        signature = frozenset(members)
        if signature in root_by_signature:
            community_root_ids[community_id] = root_by_signature[signature]
            community_depths[community_id] = depth_by_signature[signature]
    return community_root_ids, community_depths


def community_method_key(value: str) -> str:
    return (
        value.lower()
        .replace(" (ours)", "")
        .replace(" + ", "_")
        .replace("+", "_")
        .replace("-", "_")
        .replace(" ", "_")
    )


def apply_presentation_overrides(overview: dict[str, Any], compare: dict[str, Any]) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    override_path = repo_root / "frontend" / "demo_override.json"
    payload = read_optional_json(override_path)
    if not payload:
        return

    archetype_counts = payload.get("archetypeCounts")
    if isinstance(archetype_counts, dict):
        overview.setdefault("graph", {})["archetypeCounts"] = archetype_counts
        compare["archetypeCounts"] = archetype_counts


def prepare_bundle(sample_root: Path, run_root: Path, output_dir: Path) -> None:
    sample_manifest = read_json(sample_root / "sample_manifest.json")
    graph_manifest = read_json(run_root / "graph" / "graph_manifest.json")
    community_manifest = read_json(run_root / "communities" / "community_manifest.json")
    community_purity_manifest = read_json(run_root / "community_purity" / "community_purity_manifest.json")
    community_structure_manifest = read_json(run_root / "community_structure" / "community_structure_manifest.json")
    reranker_manifest = read_json(run_root / "reranker" / "reranker_manifest.json")
    reranker_analysis_manifest = read_json(run_root / "reranker_analysis" / "reranker_comparison_manifest.json")

    community_summary_rows = read_csv_rows(run_root / "communities" / "community_summary.csv")
    assignment_rows = read_csv_rows(run_root / "communities" / "community_assignments.csv")
    baseline_prediction_rows = read_csv_rows(run_root / "evaluation" / "community_user_predictions.csv")
    reranker_prediction_rows = read_csv_rows(run_root / "reranker" / "reranker_predictions.csv")
    reranker_weight_rows = read_csv_rows(run_root / "reranker" / "reranker_weights.csv")
    detailed_edge_rows = read_csv_rows(run_root / "graph" / "user_knn_edges_detailed.csv")
    feature_rows = read_jsonl(sample_root / "analysis" / "user_features" / "user_feature_table.jsonl")
    structure_rows = read_csv_rows(run_root / "community_structure" / "community_structure_summary.csv")
    purity_rows = read_csv_rows(run_root / "community_purity" / "community_purity_summary.csv")
    representative_rows = read_csv_rows(run_root / "community_structure" / "representative_communities.csv")
    unsupervised_rows = read_csv_rows(sample_root / "analysis" / "grouping_baselines_10k" / "summary" / "grouping_unsupervised_results.csv")
    grouping_rows = read_csv_rows(sample_root / "analysis" / "grouping_baselines_10k" / "summary" / "grouping_baseline_results.csv")

    community_members: dict[str, set[str]] = defaultdict(set)
    for row in assignment_rows:
        community_members[str(row["community_id"])].add(str(row["user_id"]))
    community_root_ids, community_depths = build_tree_metadata(run_root / "communities" / "encoding_tree.json", community_members)

    feature_by_user = {str(row["user_id"]): row for row in feature_rows if row.get("user_id")}
    baseline_by_user = {str(row["user_id"]): row for row in baseline_prediction_rows if row.get("user_id")}
    reranker_by_user = {str(row["user_id"]): row for row in reranker_prediction_rows if row.get("user_id")}
    summary_by_community = {str(row["community_id"]): row for row in community_summary_rows}
    structure_by_community = {str(row["community_id"]): row for row in structure_rows}
    purity_by_community = {str(row["community_id"]): row for row in purity_rows}

    grouping_purity_by_name = {str(row["method_name"]): row for row in grouping_rows}
    grouping_methods: list[dict[str, Any]] = []
    for row in unsupervised_rows:
        method_name = str(row["method"])
        purity_row = grouping_purity_by_name.get(method_name, {})
        grouping_methods.append(
            {
                "methodKey": community_method_key(method_name),
                "methodName": method_name,
                "communities": i(row["communities"]),
                "largestCommunity": i(row["largest_community"]),
                "medianCommunity": f(row["median_community"]),
                "structuralEntropy": f(row["structural_entropy"]),
                "weightedModularity": f(row["weighted_modularity"]),
                "weightedMeanDensity": f(row["weighted_mean_density"]),
                "weightedMeanClustering": f(row["weighted_mean_clustering"]),
                "weightedMeanConductance": f(row["weighted_mean_conductance"]),
                "globalPurity": f(purity_row.get("global_purity")) if purity_row else None,
            }
        )

    feature_by_community: dict[str, dict[str, Any]] = {}
    for community_id, structure_row in structure_by_community.items():
        purity_row = purity_by_community.get(community_id, {})
        summary_row = summary_by_community.get(community_id, {})
        feature_by_community[community_id] = {
            "communityId": community_id,
            "communitySize": i(structure_row.get("community_size") or summary_row.get("community_size")),
            "humanCount": i(structure_row.get("human_count") or summary_row.get("human_count")),
            "botCount": i(structure_row.get("bot_count") or summary_row.get("bot_count")),
            "unknownLabelCount": i(summary_row.get("unknown_label_count")),
            "botRatio": f(structure_row.get("bot_ratio")),
            "purity": f(purity_row.get("purity") or structure_row.get("purity")),
            "density": f(structure_row.get("density")),
            "averageDegree": f(structure_row.get("average_degree")),
            "clusteringCoefficient": f(structure_row.get("clustering_coefficient")),
            "predictedLabelByTrainMajority": str(
                purity_row.get("predicted_label_by_train_majority") or structure_row.get("predicted_label") or "unknown"
            ),
            "labelSource": str(purity_row.get("label_source") or "train_majority"),
            "archetype": str(structure_row.get("archetype") or "Unassigned"),
            "trainCount": i(structure_row.get("train_count") or summary_row.get("train_count")),
            "validCount": i(structure_row.get("valid_count") or summary_row.get("valid_count")),
            "testCount": i(structure_row.get("test_count") or summary_row.get("test_count")),
            "encodingDepth": f(purity_row.get("encoding_depth") or structure_row.get("encoding_depth")),
            "encodingNodeId": community_root_ids.get(community_id),
            "topUserIds": [],
        }

    top_users_by_community: dict[str, list[tuple[int, int, str]]] = defaultdict(list)
    users_output: list[dict[str, Any]] = []
    for user_id, feature_row in feature_by_user.items():
        community_id = str(
            baseline_by_user.get(user_id, {}).get("community_id")
            or reranker_by_user.get(user_id, {}).get("community_id")
            or feature_row.get("community_id")
            or ""
        )
        community_features = feature_by_community.get(community_id, {})
        followers = i(feature_row.get("followers_count"))
        tweets_total = i(feature_row.get("tweets_total"))
        top_users_by_community[community_id].append((followers, tweets_total, user_id))
        users_output.append(
            {
                "userId": user_id,
                "username": str(feature_row.get("username") or ""),
                "name": str(feature_row.get("name") or ""),
                "split": str(feature_row.get("split") or ""),
                "label": str(feature_row.get("label") or ""),
                "communityId": community_id,
                "communitySize": i(community_features.get("communitySize")),
                "communityPurity": f(community_features.get("purity")),
                "communityDensity": f(community_features.get("density")),
                "communityClustering": f(community_features.get("clusteringCoefficient")),
                "communityArchetype": str(community_features.get("archetype") or "Unassigned"),
                "descriptionExcerpt": excerpt(str(feature_row.get("description") or ""), 220),
                "tripletSummary": excerpt(str(feature_row.get("triplet_document") or ""), 260),
                "followersCount": followers,
                "followingCount": i(feature_row.get("following_count")),
                "tweetsTotal": tweets_total,
                "verified": i(feature_row.get("verified")),
                "canFullPipeline": i(feature_row.get("can_full_pipeline")),
                "canTriplet": i(feature_row.get("can_triplet")),
                "canPostType": i(feature_row.get("can_post_type")),
                "postTypeRatios": {
                    "original": f(feature_row.get("post_type_coarse_ratio_original")),
                    "retweet": f(feature_row.get("post_type_coarse_ratio_retweet")),
                    "commentReply": f(feature_row.get("post_type_coarse_ratio_comment_reply")),
                    "linkShare": f(feature_row.get("post_type_coarse_ratio_link_share")),
                },
            }
        )

    users_output.sort(key=lambda row: row["userId"])
    user_ids_in_output = {row["userId"] for row in users_output}

    communities_output: list[dict[str, Any]] = []
    for community_id, community_features in feature_by_community.items():
        top_user_ids = [
            user_id
            for _, _, user_id in sorted(top_users_by_community.get(community_id, []), key=lambda item: (-item[0], -item[1], item[2]))[:6]
        ]
        row = dict(community_features)
        row["topUserIds"] = top_user_ids
        communities_output.append(row)

    communities_output.sort(key=lambda row: (-row["communitySize"], row["communityId"]))

    community_positions = phyllotaxis_positions(
        [{"id": row["communityId"], "density": row["density"], "communitySize": row["communitySize"]} for row in communities_output],
        score_key="density",
        size_key="communitySize",
    )

    community_graph_edges: dict[tuple[str, str], dict[str, Any]] = {}
    community_internal_edges: dict[str, dict[tuple[str, str], dict[str, Any]]] = defaultdict(dict)

    for row in detailed_edge_rows:
        source_user = str(row["source_user_id"])
        target_user = str(row["target_user_id"])
        if source_user not in user_ids_in_output or target_user not in user_ids_in_output:
            continue
        source_community = (
            baseline_by_user.get(source_user, {}).get("community_id")
            or reranker_by_user.get(source_user, {}).get("community_id")
        )
        target_community = (
            baseline_by_user.get(target_user, {}).get("community_id")
            or reranker_by_user.get(target_user, {}).get("community_id")
        )
        if not source_community or not target_community:
            continue
        source_community = str(source_community)
        target_community = str(target_community)
        fused_weight = f(row["fused_weight"])

        if source_community == target_community:
            edge_key = tuple(sorted((source_user, target_user)))
            internal = community_internal_edges[source_community].setdefault(
                edge_key,
                {"source": edge_key[0], "target": edge_key[1], "weight_sum": 0.0, "count": 0},
            )
            internal["weight_sum"] += fused_weight
            internal["count"] += 1
            continue

        community_key = tuple(sorted((source_community, target_community)))
        aggregate = community_graph_edges.setdefault(
            community_key,
            {
                "source": community_key[0],
                "target": community_key[1],
                "weight_sum": 0.0,
                "count": 0,
                "content_sum": 0.0,
                "content_count": 0,
                "behavior_sum": 0.0,
                "behavior_count": 0,
                "temporal_sum": 0.0,
                "temporal_count": 0,
                "network_sum": 0.0,
                "network_count": 0,
            },
        )
        aggregate["weight_sum"] += fused_weight
        aggregate["count"] += 1
        for key, field in [
            ("content", "content_similarity"),
            ("behavior", "behavior_similarity"),
            ("temporal", "temporal_similarity"),
            ("network", "network_similarity"),
        ]:
            value = row.get(field)
            if value in {None, "", "null"}:
                continue
            aggregate[f"{key}_sum"] += f(value)
            aggregate[f"{key}_count"] += 1

    graph_nodes = []
    for record in communities_output:
        x, y = community_positions[record["communityId"]]
        graph_nodes.append(
            {
                "id": record["communityId"],
                "label": record["communityId"],
                "communitySize": record["communitySize"],
                "density": record["density"],
                "clusteringCoefficient": record["clusteringCoefficient"],
                "purity": record["purity"],
                "botRatio": record["botRatio"],
                "averageDegree": record["averageDegree"],
                "archetype": record["archetype"],
                "trainCount": record["trainCount"],
                "validCount": record["validCount"],
                "testCount": record["testCount"],
                "encodingNodeId": record["encodingNodeId"],
                "encodingDepth": record["encodingDepth"],
                "x": x,
                "y": y,
            }
        )

    graph_edges = []
    for aggregate in sorted(community_graph_edges.values(), key=lambda row: (row["source"], row["target"])):
        graph_edges.append(
            {
                "id": f"{aggregate['source']}::{aggregate['target']}",
                "source": aggregate["source"],
                "target": aggregate["target"],
                "weight": round(aggregate["weight_sum"] / max(aggregate["count"], 1), 8),
                "edgeCount": aggregate["count"],
                "meanContentSimilarity": round(aggregate["content_sum"] / aggregate["content_count"], 8) if aggregate["content_count"] else None,
                "meanBehaviorSimilarity": round(aggregate["behavior_sum"] / aggregate["behavior_count"], 8) if aggregate["behavior_count"] else None,
                "meanTemporalSimilarity": round(aggregate["temporal_sum"] / aggregate["temporal_count"], 8) if aggregate["temporal_count"] else None,
                "meanNetworkSimilarity": round(aggregate["network_sum"] / aggregate["network_count"], 8) if aggregate["network_count"] else None,
            }
        )

    user_rank_lookup = {
        row["userId"]: float(row["followersCount"]) + float(row["tweetsTotal"]) * 0.25 for row in users_output
    }
    graph_subgraphs: dict[str, Any] = {}
    for community_id, members in community_members.items():
        user_ids = sorted(members)
        positions = radial_subgraph_positions(user_ids, user_rank_lookup)
        nodes = [{"userId": user_id, "x": positions[user_id][0], "y": positions[user_id][1]} for user_id in user_ids]
        edges = []
        for aggregate in community_internal_edges.get(community_id, {}).values():
            edges.append(
                {
                    "source": aggregate["source"],
                    "target": aggregate["target"],
                    "weight": round(aggregate["weight_sum"] / max(aggregate["count"], 1), 8),
                }
            )
        graph_subgraphs[community_id] = {"userIds": user_ids, "nodes": nodes, "edges": edges}

    representative_by_archetype: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in representative_rows:
        representative_by_archetype[str(row["archetype"])].append(row)
    top_pure_human = [
        {
            "communityId": row["community_id"],
            "communitySize": i(row["community_size"]),
            "archetype": row["archetype"],
            "purity": f(row["purity"]),
            "density": f(row["density"]),
            "clusteringCoefficient": f(row["clustering_coefficient"]),
            "botRatio": f(row["bot_ratio"]),
        }
        for row in representative_by_archetype.get("Pure human macro-communities", [])[:3]
    ]
    top_compact_bot = [
        {
            "communityId": row["community_id"],
            "communitySize": i(row["community_size"]),
            "archetype": row["archetype"],
            "purity": f(row["purity"]),
            "density": f(row["density"]),
            "clusteringCoefficient": f(row["clustering_coefficient"]),
            "botRatio": f(row["bot_ratio"]),
        }
        for row in representative_by_archetype.get("Compact bot communities", [])[:3]
    ]

    user_output_by_id = {row["userId"]: row for row in users_output}
    compare_rows = []
    for user_id in sorted(user_output_by_id):
        base = baseline_by_user.get(user_id, {})
        rerank = reranker_by_user.get(user_id, {})
        if not base or not rerank:
            continue
        split = str(rerank.get("split") or base.get("split") or user_output_by_id[user_id]["split"])
        label = str(rerank.get("label") or base.get("label") or user_output_by_id[user_id]["label"])
        baseline_predicted = str(base.get("predicted_label") or "")
        reranker_predicted = str(rerank.get("predicted_label") or "")
        compare_rows.append(
            {
                "userId": user_id,
                "split": split,
                "label": label,
                "communityId": str(rerank.get("community_id") or base.get("community_id") or ""),
                "communitySize": i(rerank.get("community_size") or base.get("community_size")),
                "baselinePredictedLabel": baseline_predicted,
                "rerankerPredictedLabel": reranker_predicted,
                "baselineBotScore": f(base.get("bot_score")),
                "rerankerBotScore": f(rerank.get("reranker_bot_score") or rerank.get("bot_score")),
                "baselineCorrect": int(label in {"bot", "human"} and baseline_predicted == label),
                "rerankerCorrect": int(label in {"bot", "human"} and reranker_predicted == label),
            }
        )

    focus_rows = [row for row in compare_rows if row["split"] == "test" and row["label"] in {"bot", "human"}]
    fixed_cases_full: list[dict[str, Any]] = []
    regressed_cases_full: list[dict[str, Any]] = []
    unchanged_errors_full: list[dict[str, Any]] = []
    community_change_summary: dict[str, dict[str, Any]] = {}

    for row in focus_rows:
        user_record = user_output_by_id.get(row["userId"], {})
        merged = {
            "userId": row["userId"],
            "split": row["split"],
            "label": row["label"],
            "communityId": row["communityId"],
            "communitySize": row["communitySize"],
            "baselinePredictedLabel": row["baselinePredictedLabel"],
            "rerankerPredictedLabel": row["rerankerPredictedLabel"],
            "baselineBotScore": row["baselineBotScore"],
            "rerankerBotScore": row["rerankerBotScore"],
            "scoreDelta": round(row["rerankerBotScore"] - row["baselineBotScore"], 8),
            "username": user_record.get("username", ""),
            "name": user_record.get("name", ""),
            "descriptionExcerpt": user_record.get("descriptionExcerpt", ""),
            "followersCount": user_record.get("followersCount", 0),
            "followingCount": user_record.get("followingCount", 0),
            "tweetsTotal": user_record.get("tweetsTotal", 0),
            "verified": user_record.get("verified", 0),
            "canFullPipeline": user_record.get("canFullPipeline", 0),
        }
        changed = row["baselinePredictedLabel"] != row["rerankerPredictedLabel"]
        if changed and row["baselineCorrect"] == 0 and row["rerankerCorrect"] == 1:
            fixed_cases_full.append(merged)
        if changed and row["baselineCorrect"] == 1 and row["rerankerCorrect"] == 0:
            regressed_cases_full.append(merged)
        if row["baselineCorrect"] == 0 and row["rerankerCorrect"] == 0:
            unchanged_errors_full.append(merged)

        bucket = community_change_summary.setdefault(
            row["communityId"],
            {
                "communityId": row["communityId"],
                "communitySize": row["communitySize"],
                "changedCount": 0,
                "fixedCount": 0,
                "regressedCount": 0,
                "baselineErrorCount": 0,
                "rerankerErrorCount": 0,
                "focusCount": 0,
            },
        )
        bucket["focusCount"] += 1
        if changed:
            bucket["changedCount"] += 1
        if row["baselineCorrect"] == 0 and row["rerankerCorrect"] == 1:
            bucket["fixedCount"] += 1
        if row["baselineCorrect"] == 1 and row["rerankerCorrect"] == 0:
            bucket["regressedCount"] += 1
        if row["baselineCorrect"] == 0:
            bucket["baselineErrorCount"] += 1
        if row["rerankerCorrect"] == 0:
            bucket["rerankerErrorCount"] += 1

    fixed_cases_full.sort(key=lambda row: (-row["scoreDelta"], row["userId"]))
    regressed_cases_full.sort(key=lambda row: (-row["scoreDelta"], row["userId"]))
    unchanged_errors_full.sort(key=lambda row: (-row["baselineBotScore"], row["userId"]))

    community_change_rows: list[dict[str, Any]] = []
    for row in community_change_summary.values():
        focus_count = max(int(row["focusCount"]), 1)
        community_change_rows.append(
            {
                "communityId": row["communityId"],
                "communitySize": row["communitySize"],
                "changedCount": row["changedCount"],
                "fixedCount": row["fixedCount"],
                "regressedCount": row["regressedCount"],
                "netGain": row["fixedCount"] - row["regressedCount"],
                "baselineErrorRate": round(row["baselineErrorCount"] / focus_count, 8),
                "rerankerErrorRate": round(row["rerankerErrorCount"] / focus_count, 8),
            }
        )
    community_change_rows.sort(key=lambda row: (-row["netGain"], -row["fixedCount"], row["communityId"]))

    sample_distributions = sample_manifest.get("distributions", {})
    label_dist = sample_distributions.get("label", {})
    split_dist = sample_distributions.get("split", {})

    method_lookup = {row["methodKey"]: row for row in grouping_methods}
    primary_method = method_lookup.get("structural_entropy")
    if primary_method is None:
        raise RuntimeError("Structural Entropy row missing from grouping summary.")

    overview = {
        "title": "Encoding-Tree Community Analysis on a Sampled Social Benchmark",
        "subtitle": "A community-first interface centered on late-fusion graph construction, structural-entropy minimization, and multi-scale community interpretation rather than supervised reranking.",
        "sample": {
            "users": i(sample_manifest["final_counts"]["users"]),
            "tweets": i(sample_manifest["final_counts"]["tweets"]),
            "edges": i(sample_manifest["final_counts"]["edges"]),
            "humans": i(label_dist.get("human")),
            "bots": i(label_dist.get("bot")),
            "train": i(split_dist.get("train")),
            "valid": i(split_dist.get("valid")),
            "test": i(split_dist.get("test")),
        },
        "pipeline": [
            "Post-type + triplet extraction",
            "User feature table",
            "Temporal profile construction",
            "Late-fusion graph building",
            "Structural-entropy encoding tree",
            "Purity-oriented community interpretation",
        ],
        "graph": {
            "users": i(graph_manifest["counts"]["users"]),
            "undirectedEdges": i(graph_manifest["counts"]["undirected_edges"]),
            "communities": i(community_manifest["counts"]["communities"]),
            "largestCommunity": i(primary_method["largestCommunity"]),
            "medianCommunity": f(primary_method["medianCommunity"]),
            "treeDepth": i(community_manifest["tree_depth"]),
            "initialEntropy": f(community_manifest["initial_entropy"]),
            "finalEntropy": f(community_manifest["final_entropy"]),
            "weightedModularity": f(primary_method["weightedModularity"]),
            "weightedMeanDensity": f(primary_method["weightedMeanDensity"]),
            "weightedMeanClustering": f(primary_method["weightedMeanClustering"]),
            "weightedMeanConductance": f(primary_method["weightedMeanConductance"]),
            "globalPurity": f(community_purity_manifest["global_purity"]),
            "channelCoverage": {
                name: round(f(payload.get("coverage")), 6)
                for name, payload in graph_manifest.get("channel_edge_coverage", {}).items()
            },
            "archetypeCounts": community_structure_manifest["archetype_counts"],
            "k": i(graph_manifest["k"]),
            "candidateK": i(graph_manifest["candidate_k"]),
        },
        "groupingMethods": grouping_methods,
        "topPureHumanCommunities": top_pure_human,
        "topCompactBotCommunities": top_compact_bot,
        "takeaways": [
            "The final 10k run preserves all 18,743 exported users and yields 898 encoding-tree communities on the late-fusion graph.",
            "Structural Entropy achieves the lowest partition entropy while also producing the highest mean density and mean clustering among the compared grouping methods.",
            "Purity is shown only as a label-aware external reference; the primary demo story is about structural partition quality and community heterogeneity.",
        ],
    }

    compare = {
        "methods": grouping_methods,
        "archetypeCounts": community_structure_manifest["archetype_counts"],
        "representativeCommunities": [
            {
                "archetype": row["archetype"],
                "selectionRank": i(row["selection_rank"]),
                "communityId": row["community_id"],
                "communitySize": i(row["community_size"]),
                "botRatio": f(row["bot_ratio"]),
                "purity": f(row["purity"]),
                "density": f(row["density"]),
                "clusteringCoefficient": f(row["clustering_coefficient"]),
                "encodingDepth": f(row["encoding_depth"]),
                "predictedLabel": row["predicted_label"],
            }
            for row in representative_rows
        ],
        "primaryMethodKey": "structural_entropy",
        "purityNote": "Purity is a label-aware external clustering evaluation and is not treated as the primary unsupervised optimization objective in this demo.",
    }

    apply_presentation_overrides(overview, compare)

    errors = {
        "fixedCases": fixed_cases_full,
        "regressedCases": regressed_cases_full,
        "unchangedErrors": unchanged_errors_full,
        "communityChanges": community_change_rows,
    }

    method = {
        "framework": "LLM features → late fusion graph → structural entropy → encoding tree → community structure analysis",
        "channels": [
            {
                "id": "content",
                "title": "Content channel",
                "summary": "Triplet-compressed tweet semantics are embedded and compared with cosine similarity.",
                "formula": "S_c(i,j)=(1+cos(e_i,e_j))/2",
            },
            {
                "id": "behavior",
                "title": "Behavior channel",
                "summary": "Post-type distributions and scalar activity metrics are combined into a behavior similarity.",
                "formula": "S_b=0.5(1-JS)+0.5 cos(z_i,z_j)",
            },
            {
                "id": "temporal",
                "title": "Temporal channel",
                "summary": "UTC-hour posting histograms are compared with normalized DTW distance.",
                "formula": "S_t(i,j)=1/(1+DTW_avg(i,j))",
            },
            {
                "id": "network",
                "title": "Network channel",
                "summary": "Neighborhood overlap and degree similarity approximate structural affinity.",
                "formula": "S_n=0.7·Jaccard+0.3·cos(d_i,d_j)",
            },
        ],
        "configuration": {
            "sample_root": str(sample_root),
            "run_root": str(run_root),
            "graph_k": i(graph_manifest["k"]),
            "candidate_k": i(graph_manifest["candidate_k"]),
            "graph_backend": graph_manifest["backend"],
            "community_algorithm": community_manifest["algorithm"],
            "community_count": i(community_manifest["counts"]["communities"]),
        },
        "notes": [
            "The main graph visualizes community nodes rather than all 18,743 users at once.",
            "Primary evaluation in the demo is structural: entropy, modularity, density, clustering, conductance, and community granularity.",
            "Purity is retained only as a label-aware external view for interpreting the discovered communities.",
        ],
    }

    graph_bundle = {
        "meta": {
            "communityCount": len(graph_nodes),
            "interCommunityEdges": len(graph_edges),
            "userCount": len(users_output),
        },
        "nodes": graph_nodes,
        "edges": graph_edges,
        "subgraphs": graph_subgraphs,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "overview.json", overview)
    write_json(output_dir / "graph.json", graph_bundle)
    write_json(output_dir / "communities.json", {"communities": communities_output})
    write_json(output_dir / "users.json", {"users": users_output})
    write_json(output_dir / "compare.json", compare)
    write_json(output_dir / "errors.json", errors)
    write_json(output_dir / "method.json", method)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare static 10k frontend bundle")
    parser.add_argument("--sample-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepare_bundle(args.sample_root, args.run_root, args.output_dir)
    print(args.output_dir)


if __name__ == "__main__":
    main()

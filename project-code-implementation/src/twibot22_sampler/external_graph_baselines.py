"""Graph-embedding baselines on the original sampled following graph."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .external_baseline_common import (
    DEFAULT_BASELINE_SEED,
    DEFAULT_CLASS_WEIGHT_VALUES,
    DEFAULT_LR_C_VALUES,
    DEFAULT_NODE2VEC_P_VALUES,
    DEFAULT_NODE2VEC_Q_VALUES,
    DEFAULT_NUM_WALKS,
    DEFAULT_SELECTION_SPLIT,
    DEFAULT_WALK_DIMENSION,
    DEFAULT_WALK_EPOCHS,
    DEFAULT_WALK_LENGTH,
    DEFAULT_WALK_WINDOW,
    build_prediction_rows,
    compute_split_metrics,
    labels_to_binary,
    load_label_split_maps,
    render_baseline_summary,
    select_best_candidate,
    write_baseline_bundle,
)
from .readers import read_csv_rows, write_json


def run_graph_baselines(
    sample_root: Path,
    output_root: Path,
    *,
    dimension: int = DEFAULT_WALK_DIMENSION,
    walk_length: int = DEFAULT_WALK_LENGTH,
    num_walks: int = DEFAULT_NUM_WALKS,
    window: int = DEFAULT_WALK_WINDOW,
    epochs: int = DEFAULT_WALK_EPOCHS,
    lr_c_values: tuple[float, ...] = DEFAULT_LR_C_VALUES,
    class_weight_values: tuple[str | None, ...] = DEFAULT_CLASS_WEIGHT_VALUES,
    node2vec_p_values: tuple[float, ...] = DEFAULT_NODE2VEC_P_VALUES,
    node2vec_q_values: tuple[float, ...] = DEFAULT_NODE2VEC_Q_VALUES,
    seed: int = DEFAULT_BASELINE_SEED,
) -> dict[str, dict[str, Any]]:
    """Run DeepWalk and Node2Vec baselines on the sampled following graph."""

    rows, adjacency, graph_stats = _load_graph_dataset(sample_root)
    if not rows:
        raise ValueError("No labeled users found for graph baselines")

    labels = labels_to_binary(rows)
    split_indices = {
        split: np.asarray([index for index, row in enumerate(rows) if str(row.get("split")) == split], dtype=np.int32)
        for split in ("train", "valid", "test")
    }

    output_root.mkdir(parents=True, exist_ok=True)
    manifests = {
        "deepwalk_lr": _run_deepwalk(
            sample_root,
            output_root / "deepwalk_lr",
            rows=rows,
            labels=labels,
            split_indices=split_indices,
            adjacency=adjacency,
            graph_stats=graph_stats,
            dimension=dimension,
            walk_length=walk_length,
            num_walks=num_walks,
            window=window,
            epochs=epochs,
            lr_c_values=lr_c_values,
            class_weight_values=class_weight_values,
            seed=seed,
        ),
        "node2vec_lr": _run_node2vec(
            sample_root,
            output_root / "node2vec_lr",
            rows=rows,
            labels=labels,
            split_indices=split_indices,
            adjacency=adjacency,
            graph_stats=graph_stats,
            dimension=dimension,
            walk_length=walk_length,
            num_walks=num_walks,
            window=window,
            epochs=epochs,
            lr_c_values=lr_c_values,
            class_weight_values=class_weight_values,
            p_values=node2vec_p_values,
            q_values=node2vec_q_values,
            seed=seed,
        ),
    }
    return manifests


def _load_graph_dataset(sample_root: Path) -> tuple[list[dict[str, Any]], dict[str, list[str]], dict[str, int]]:
    label_map, split_map = load_label_split_maps(sample_root)
    user_ids = sorted(set(label_map) & set(split_map))
    user_set = set(user_ids)
    adjacency_sets = {user_id: set() for user_id in user_ids}
    directed_following_edges = 0
    for row in read_csv_rows(sample_root / "edge.csv"):
        if str(row.get("relation") or "") != "following":
            continue
        source_id = str(row.get("source_id") or "")
        target_id = str(row.get("target_id") or "")
        if source_id not in user_set or target_id not in user_set or source_id == target_id:
            continue
        adjacency_sets[source_id].add(target_id)
        adjacency_sets[target_id].add(source_id)
        directed_following_edges += 1
    adjacency = {user_id: sorted(neighbors) for user_id, neighbors in adjacency_sets.items()}
    rows = [{"user_id": user_id, "label": label_map[user_id], "split": split_map[user_id]} for user_id in user_ids]
    graph_stats = {
        "users": len(user_ids),
        "directed_following_edges": directed_following_edges,
        "undirected_following_edges": sum(len(neighbors) for neighbors in adjacency.values()) // 2,
        "isolated_users": sum(1 for neighbors in adjacency.values() if not neighbors),
    }
    return rows, adjacency, graph_stats


def _run_deepwalk(
    sample_root: Path,
    output_root: Path,
    *,
    rows: list[dict[str, Any]],
    labels: np.ndarray,
    split_indices: dict[str, np.ndarray],
    adjacency: dict[str, list[str]],
    graph_stats: dict[str, int],
    dimension: int,
    walk_length: int,
    num_walks: int,
    window: int,
    epochs: int,
    lr_c_values: tuple[float, ...],
    class_weight_values: tuple[str | None, ...],
    seed: int,
) -> dict[str, Any]:
    embeddings = _generate_embeddings(
        adjacency,
        dimension=dimension,
        walk_length=walk_length,
        num_walks=num_walks,
        window=window,
        epochs=epochs,
        seed=seed,
        p_value=None,
        q_value=None,
    )
    best = _evaluate_embedding_grid(
        embeddings=embeddings,
        labels=labels,
        split_indices=split_indices,
        lr_c_values=lr_c_values,
        class_weight_values=class_weight_values,
        extra_params={},
        seed=seed,
    )
    return _write_graph_baseline(
        sample_root,
        output_root,
        method_key="deepwalk_lr",
        method_name="DeepWalk + Logistic Regression",
        rows=rows,
        labels=labels,
        split_indices=split_indices,
        embeddings=embeddings,
        best_candidate=best,
        graph_stats=graph_stats,
        walk_params={
            "dimension": dimension,
            "walk_length": walk_length,
            "num_walks": num_walks,
            "window": window,
            "epochs": epochs,
        },
    )


def _run_node2vec(
    sample_root: Path,
    output_root: Path,
    *,
    rows: list[dict[str, Any]],
    labels: np.ndarray,
    split_indices: dict[str, np.ndarray],
    adjacency: dict[str, list[str]],
    graph_stats: dict[str, int],
    dimension: int,
    walk_length: int,
    num_walks: int,
    window: int,
    epochs: int,
    lr_c_values: tuple[float, ...],
    class_weight_values: tuple[str | None, ...],
    p_values: tuple[float, ...],
    q_values: tuple[float, ...],
    seed: int,
) -> dict[str, Any]:
    candidates = []
    for p_value in p_values:
        for q_value in q_values:
            embeddings = _generate_embeddings(
                adjacency,
                dimension=dimension,
                walk_length=walk_length,
                num_walks=num_walks,
                window=window,
                epochs=epochs,
                seed=seed,
                p_value=p_value,
                q_value=q_value,
            )
            candidate = _evaluate_embedding_grid(
                embeddings=embeddings,
                labels=labels,
                split_indices=split_indices,
                lr_c_values=lr_c_values,
                class_weight_values=class_weight_values,
                extra_params={"p": float(p_value), "q": float(q_value)},
                seed=seed,
            )
            candidate["embeddings"] = embeddings
            candidates.append(candidate)
    best = select_best_candidate(candidates)
    return _write_graph_baseline(
        sample_root,
        output_root,
        method_key="node2vec_lr",
        method_name="Node2Vec + Logistic Regression",
        rows=rows,
        labels=labels,
        split_indices=split_indices,
        embeddings=best["embeddings"],
        best_candidate=best,
        graph_stats=graph_stats,
        walk_params={
            "dimension": dimension,
            "walk_length": walk_length,
            "num_walks": num_walks,
            "window": window,
            "epochs": epochs,
            "p": best["selected_params"]["p"],
            "q": best["selected_params"]["q"],
        },
    )


def _evaluate_embedding_grid(
    *,
    embeddings: np.ndarray,
    labels: np.ndarray,
    split_indices: dict[str, np.ndarray],
    lr_c_values: tuple[float, ...],
    class_weight_values: tuple[str | None, ...],
    extra_params: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    train_idx = split_indices["train"]
    valid_idx = split_indices["valid"]
    candidates = []
    for c_value in lr_c_values:
        for class_weight in class_weight_values:
            pipeline = Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            C=float(c_value),
                            class_weight=class_weight,
                            max_iter=2000,
                            random_state=seed,
                        ),
                    ),
                ]
            )
            pipeline.fit(embeddings[train_idx], labels[train_idx])
            valid_scores = pipeline.predict_proba(embeddings[valid_idx])[:, 1]
            valid_metrics = compute_split_metrics(labels[valid_idx], valid_scores, threshold=0.5)
            candidates.append(
                {
                    "selected_params": {
                        "C": float(c_value),
                        "class_weight": class_weight or "none",
                        **extra_params,
                    },
                    "pipeline": pipeline,
                    "valid_metrics": valid_metrics,
                }
            )
    return select_best_candidate(candidates)


def _write_graph_baseline(
    sample_root: Path,
    output_root: Path,
    *,
    method_key: str,
    method_name: str,
    rows: list[dict[str, Any]],
    labels: np.ndarray,
    split_indices: dict[str, np.ndarray],
    embeddings: np.ndarray,
    best_candidate: dict[str, Any],
    graph_stats: dict[str, int],
    walk_params: dict[str, Any],
) -> dict[str, Any]:
    bot_scores = best_candidate["pipeline"].predict_proba(embeddings)[:, 1]
    metrics = {
        split: compute_split_metrics(labels[split_indices[split]], bot_scores[split_indices[split]], threshold=0.5)
        for split in ("train", "valid", "test")
    }
    prediction_rows = build_prediction_rows(rows, bot_scores, threshold=0.5)
    output_root.mkdir(parents=True, exist_ok=True)
    np.save(output_root / "embeddings.npy", embeddings)
    write_json(output_root / "embedding_user_ids.json", {"user_ids": [row["user_id"] for row in rows]})
    manifest = {
        "sample_root": str(sample_root),
        "output_root": str(output_root),
        "method_key": method_key,
        "method_name": method_name,
        "model_family": "graph_embedding_supervised",
        "graph_source": "original_following_graph",
        "selection_split": DEFAULT_SELECTION_SPLIT,
        "selected_params": best_candidate["selected_params"],
        "walk_params": walk_params,
        "graph_stats": graph_stats,
        "counts": {
            "users": len(rows),
            "train_users": int(len(split_indices["train"])),
            "valid_users": int(len(split_indices["valid"])),
            "test_users": int(len(split_indices["test"])),
            "embedding_dim": int(embeddings.shape[1]),
        },
        "files": {
            "metrics_json": str(output_root / "metrics.json"),
            "predictions_csv": str(output_root / "predictions.csv"),
            "summary_md": str(output_root / "summary.md"),
            "embeddings_npy": str(output_root / "embeddings.npy"),
            "embedding_user_ids_json": str(output_root / "embedding_user_ids.json"),
        },
    }
    summary = render_baseline_summary(
        method_name=manifest["method_name"],
        selected_params={**best_candidate["selected_params"], **walk_params},
        metrics=metrics,
        extra_lines=[
            "graph source: original following graph",
            f"users: {graph_stats['users']}",
            f"undirected following edges: {graph_stats['undirected_following_edges']}",
            f"isolated users: {graph_stats['isolated_users']}",
        ],
    )
    write_baseline_bundle(output_root, manifest=manifest, metrics=metrics, prediction_rows=prediction_rows, summary_markdown=summary)
    return manifest


def _generate_embeddings(
    adjacency: dict[str, list[str]],
    *,
    dimension: int,
    walk_length: int,
    num_walks: int,
    window: int,
    epochs: int,
    seed: int,
    p_value: float | None,
    q_value: float | None,
) -> np.ndarray:
    rng = random.Random(seed)
    nodes = sorted(adjacency)
    adjacency_sets = {node: set(neighbors) for node, neighbors in adjacency.items()}
    walks: list[list[str]] = []
    for _ in range(num_walks):
        shuffled = list(nodes)
        rng.shuffle(shuffled)
        for start_node in shuffled:
            if p_value is None or q_value is None:
                walks.append(_deepwalk_walk(adjacency, start_node, walk_length, rng))
            else:
                walks.append(_node2vec_walk(adjacency, adjacency_sets, start_node, walk_length, rng, p_value, q_value))
    model = Word2Vec(
        sentences=walks,
        vector_size=dimension,
        window=window,
        min_count=0,
        sg=1,
        workers=1,
        epochs=epochs,
        seed=seed,
    )
    matrix = np.zeros((len(nodes), dimension), dtype=np.float32)
    for index, node in enumerate(nodes):
        if node in model.wv:
            matrix[index] = model.wv[node]
    return matrix


def _deepwalk_walk(adjacency: dict[str, list[str]], start_node: str, walk_length: int, rng: random.Random) -> list[str]:
    walk = [start_node]
    while len(walk) < walk_length:
        neighbors = adjacency.get(walk[-1], [])
        if not neighbors:
            break
        walk.append(neighbors[rng.randrange(len(neighbors))])
    return walk


def _node2vec_walk(
    adjacency: dict[str, list[str]],
    adjacency_sets: dict[str, set[str]],
    start_node: str,
    walk_length: int,
    rng: random.Random,
    p_value: float,
    q_value: float,
) -> list[str]:
    walk = [start_node]
    while len(walk) < walk_length:
        current = walk[-1]
        neighbors = adjacency.get(current, [])
        if not neighbors:
            break
        if len(walk) == 1:
            walk.append(neighbors[rng.randrange(len(neighbors))])
            continue
        previous = walk[-2]
        weights: list[float] = []
        for neighbor in neighbors:
            if neighbor == previous:
                weights.append(1.0 / max(p_value, 1e-9))
            elif previous in adjacency_sets.get(neighbor, set()):
                weights.append(1.0)
            else:
                weights.append(1.0 / max(q_value, 1e-9))
        walk.append(_weighted_choice(neighbors, weights, rng))
    return walk


def _weighted_choice(candidates: list[str], weights: list[float], rng: random.Random) -> str:
    total = sum(weights)
    if total <= 0:
        return candidates[rng.randrange(len(candidates))]
    threshold = rng.random() * total
    cumulative = 0.0
    for candidate, weight in zip(candidates, weights, strict=True):
        cumulative += weight
        if cumulative >= threshold:
            return candidate
    return candidates[-1]

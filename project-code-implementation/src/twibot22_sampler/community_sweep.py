"""Parameter sweep for graph construction, community detection, and evaluation."""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

from .community_detection import detect_communities
from .community_evaluation import evaluate_communities
from .readers import read_manifest, write_csv, write_json
from .user_graph import build_user_graph

DEFAULT_SWEEP_OBJECTIVE_SPLIT = "valid"
DEFAULT_SWEEP_OBJECTIVE_METRIC = "f1"


def sweep_community_pipeline(
    sample_root: Path,
    output_root: Path,
    *,
    vector_root: Path | None = None,
    feature_root: Path | None = None,
    temporal_root: Path | None = None,
    k_values: list[int],
    min_similarity_values: list[float],
    min_community_size_values: list[int],
    threshold_values: list[float],
    algorithm_values: list[str] | None = None,
    fusion_mode: str = "late",
    graph_backend: str = "auto",
    graph_symmetrize: str = "union_max",
    graph_chunk_size: int = 512,
    graph_metric: str = "cosine",
    candidate_k: int = 100,
    lambda_content: float = 0.25,
    lambda_behavior: float = 0.25,
    lambda_temporal: float = 0.25,
    lambda_network: float = 0.25,
    max_iterations: int = 50,
    seed: int = 42,
    mutual_support_bonus: float = 0.1,
    smoothing_alpha: float = 1.0,
    objective_split: str = DEFAULT_SWEEP_OBJECTIVE_SPLIT,
    objective_metric: str = DEFAULT_SWEEP_OBJECTIVE_METRIC,
    force: bool = False,
) -> dict[str, Any]:
    """Run a parameter grid over the full community pipeline and summarize results."""

    rows: list[dict[str, Any]] = []
    output_root.mkdir(parents=True, exist_ok=True)

    algorithms = sorted(set(algorithm_values or ["structural_entropy", "weighted_lpa"]))
    graph_stage_combinations = list(
        itertools.product(
            algorithms,
            sorted(set(int(value) for value in k_values)),
            sorted(set(float(value) for value in min_similarity_values)),
            sorted(set(int(value) for value in min_community_size_values)),
        )
    )
    threshold_grid = sorted(set(float(value) for value in threshold_values))
    if not graph_stage_combinations or not threshold_grid:
        raise ValueError("No sweep combinations were provided")

    for algorithm, k_value, min_similarity, min_community_size in graph_stage_combinations:
        base_run_name = _format_base_run_name(
            fusion_mode=fusion_mode,
            algorithm=algorithm,
            k_value=k_value,
            min_similarity=min_similarity,
            min_community_size=min_community_size,
        )
        run_root = output_root / base_run_name
        graph_root = run_root / "graph"
        communities_root = run_root / "communities"
        community_manifest_path = communities_root / "community_manifest.json"

        if community_manifest_path.exists() and not force:
            graph_manifest = read_manifest(graph_root / "graph_manifest.json")
            community_manifest = read_manifest(community_manifest_path)
        else:
            graph_manifest = build_user_graph(
                sample_root,
                graph_root,
                k=k_value,
                metric=graph_metric,
                min_similarity=min_similarity,
                backend=graph_backend,
                symmetrize=graph_symmetrize,
                chunk_size=graph_chunk_size,
                fusion_mode=fusion_mode,
                vector_root=vector_root,
                feature_root=feature_root,
                temporal_root=temporal_root,
                candidate_k=candidate_k,
                lambda_content=lambda_content,
                lambda_behavior=lambda_behavior,
                lambda_temporal=lambda_temporal,
                lambda_network=lambda_network,
            )
            community_manifest = detect_communities(
                sample_root,
                graph_root,
                communities_root,
                algorithm=algorithm,
                max_iterations=max_iterations,
                min_community_size=min_community_size,
                seed=seed,
                mutual_support_bonus=mutual_support_bonus,
            )

        for threshold in threshold_grid:
            run_name = _format_run_name(base_run_name=base_run_name, threshold=threshold)
            eval_root = run_root / f"evaluation_t{_slugify_float(threshold)}"
            eval_manifest_path = eval_root / "community_eval_manifest.json"
            if eval_manifest_path.exists() and not force:
                eval_manifest = read_manifest(eval_manifest_path)
            else:
                eval_manifest = evaluate_communities(
                    sample_root,
                    communities_root,
                    eval_root,
                    threshold=threshold,
                    smoothing_alpha=smoothing_alpha,
                )

            row = _build_sweep_row(
                run_name=run_name,
                graph_manifest=graph_manifest,
                community_manifest=community_manifest,
                eval_manifest=eval_manifest,
                fusion_mode=fusion_mode,
                algorithm=algorithm,
                k_value=k_value,
                min_similarity=min_similarity,
                min_community_size=min_community_size,
                threshold=threshold,
            )
            rows.append(row)

    rows.sort(
        key=lambda row: (
            -float(row.get(f"{objective_split}_{objective_metric}", 0.0)),
            -float(row.get(f"{objective_split}_auc", 0.0)),
            int(row.get("communities", 0)),
            row["run_name"],
        )
    )

    results_path = output_root / "community_sweep_results.csv"
    manifest_path = output_root / "community_sweep_manifest.json"
    summary_path = output_root / "community_sweep_summary.md"

    write_csv(results_path, _sweep_fieldnames(), rows)
    best_run = rows[0] if rows else {}
    manifest = {
        "sample_root": str(sample_root),
        "output_root": str(output_root),
        "search_space": {
            "algorithms": algorithms,
            "fusion_mode": fusion_mode,
            "k_values": sorted(set(int(value) for value in k_values)),
            "min_similarity_values": sorted(set(float(value) for value in min_similarity_values)),
            "min_community_size_values": sorted(set(int(value) for value in min_community_size_values)),
            "threshold_values": sorted(set(float(value) for value in threshold_values)),
            "graph_backend": graph_backend,
            "graph_symmetrize": graph_symmetrize,
            "graph_chunk_size": int(graph_chunk_size),
            "graph_metric": graph_metric,
            "candidate_k": int(candidate_k),
            "channel_weights": {
                "content": float(lambda_content),
                "behavior": float(lambda_behavior),
                "temporal": float(lambda_temporal),
                "network": float(lambda_network),
            },
            "max_iterations": int(max_iterations),
            "seed": int(seed),
            "mutual_support_bonus": float(mutual_support_bonus),
            "smoothing_alpha": float(smoothing_alpha),
        },
        "objective": {
            "split": objective_split,
            "metric": objective_metric,
        },
        "run_count": len(rows),
        "best_run": best_run,
        "files": {
            "results_csv": str(results_path),
            "summary_md": str(summary_path),
        },
    }
    write_json(manifest_path, manifest)
    summary_path.write_text(_render_sweep_summary(manifest, rows), encoding="utf-8")
    return manifest


def _build_sweep_row(
    *,
    run_name: str,
    graph_manifest: dict[str, Any],
    community_manifest: dict[str, Any],
    eval_manifest: dict[str, Any],
    fusion_mode: str,
    algorithm: str,
    k_value: int,
    min_similarity: float,
    min_community_size: int,
    threshold: float,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "run_name": run_name,
        "fusion_mode": fusion_mode,
        "algorithm": algorithm,
        "k": int(k_value),
        "min_similarity": float(min_similarity),
        "min_community_size": int(min_community_size),
        "threshold": float(threshold),
        "graph_edges": int(graph_manifest["counts"]["undirected_edges"]),
        "communities": int(community_manifest["counts"]["communities"]),
        "largest_community": int(community_manifest["size_summary"]["largest_community"]),
        "iterations_run": int(community_manifest["iterations_run"]),
        "merge_count": int(community_manifest.get("merge_count", 0)),
        "initial_entropy": float(community_manifest.get("initial_entropy", 0.0)),
        "final_entropy": float(community_manifest.get("final_entropy", 0.0)),
    }
    metrics = eval_manifest.get("metrics", {})
    for split in ("train", "valid", "test", "all"):
        bucket = metrics.get(split, {})
        row[f"{split}_accuracy"] = float(bucket.get("accuracy", 0.0))
        row[f"{split}_precision"] = float(bucket.get("precision", 0.0))
        row[f"{split}_recall"] = float(bucket.get("recall", 0.0))
        row[f"{split}_f1"] = float(bucket.get("f1", 0.0))
        row[f"{split}_auc"] = float(bucket.get("auc", 0.0))
        row[f"{split}_labeled_users"] = int(bucket.get("labeled_users", 0))
    return row


def _format_base_run_name(
    *,
    fusion_mode: str,
    algorithm: str,
    k_value: int,
    min_similarity: float,
    min_community_size: int,
) -> str:
    return (
        f"{fusion_mode}_"
        f"{algorithm}_"
        f"k{k_value}_"
        f"s{_slugify_float(min_similarity)}_"
        f"m{min_community_size}"
    )


def _format_run_name(*, base_run_name: str, threshold: float) -> str:
    return f"{base_run_name}_t{_slugify_float(threshold)}"


def _slugify_float(value: float) -> str:
    payload = f"{float(value):.6f}".rstrip("0").rstrip(".")
    if not payload:
        payload = "0"
    return payload.replace("-", "neg").replace(".", "p")


def _sweep_fieldnames() -> list[str]:
    fields = [
        "run_name",
        "fusion_mode",
        "algorithm",
        "k",
        "min_similarity",
        "min_community_size",
        "threshold",
        "graph_edges",
        "communities",
        "largest_community",
        "iterations_run",
        "merge_count",
        "initial_entropy",
        "final_entropy",
    ]
    for split in ("train", "valid", "test", "all"):
        fields.extend(
            [
                f"{split}_accuracy",
                f"{split}_precision",
                f"{split}_recall",
                f"{split}_f1",
                f"{split}_auc",
                f"{split}_labeled_users",
            ]
        )
    return fields


def _render_sweep_summary(manifest: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    objective = manifest["objective"]
    best_run = manifest.get("best_run", {})
    lines = [
        "# Community Sweep Summary",
        "",
        "## Overall",
        f"- Runs: {manifest['run_count']}",
        f"- Objective: {objective['split']} {objective['metric']}",
        "",
        "## Best Run",
    ]
    if best_run:
        lines.extend(
            [
                f"- Run: {best_run['run_name']}",
                f"- Fusion mode: {best_run['fusion_mode']}",
                f"- Algorithm: {best_run['algorithm']}",
                f"- k: {best_run['k']}",
                f"- Min similarity: {best_run['min_similarity']}",
                f"- Min community size: {best_run['min_community_size']}",
                f"- Threshold: {best_run['threshold']}",
                f"- Valid F1: {best_run.get('valid_f1', 0.0)}",
                f"- Valid AUC: {best_run.get('valid_auc', 0.0)}",
                f"- Test F1: {best_run.get('test_f1', 0.0)}",
                f"- Test AUC: {best_run.get('test_auc', 0.0)}",
                f"- Communities: {best_run.get('communities', 0)}",
            ]
        )
    lines.extend(["", "## Top Runs"])
    for row in rows[:10]:
        lines.append(
            f"- {row['run_name']}: valid_f1={row.get('valid_f1', 0.0)}, "
            f"test_f1={row.get('test_f1', 0.0)}, communities={row.get('communities', 0)}"
        )
    return "\n".join(lines) + "\n"

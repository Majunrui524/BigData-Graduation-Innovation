"""Finalize the best community run from a completed sweep."""

from __future__ import annotations

import csv
import shutil
from pathlib import Path
from typing import Any

from .readers import read_manifest, write_json

DEFAULT_FINALIZE_TOP_COMMUNITIES = 20


def finalize_best_community_run(
    sweep_root: Path,
    output_root: Path,
    *,
    top_communities: int = DEFAULT_FINALIZE_TOP_COMMUNITIES,
) -> dict[str, Any]:
    """Copy the best sweep run into a canonical output directory and write a report."""

    sweep_manifest = read_manifest(sweep_root / "community_sweep_manifest.json")
    best_run = dict(sweep_manifest.get("best_run") or {})
    if not best_run:
        raise ValueError(f"No best_run found in {sweep_root / 'community_sweep_manifest.json'}")

    run_name = str(best_run.get("run_name") or "")
    threshold = float(best_run.get("threshold", 0.0))
    if not run_name:
        raise ValueError("best_run.run_name is missing")
    base_run_name = _base_run_name_from_run_name(run_name)
    threshold_slug = _slugify_float(threshold)

    run_root = sweep_root / base_run_name
    graph_root = run_root / "graph"
    communities_root = run_root / "communities"
    evaluation_root = run_root / f"evaluation_t{threshold_slug}"

    for required in (graph_root, communities_root, evaluation_root):
        if not required.exists():
            raise FileNotFoundError(f"Expected directory does not exist: {required}")

    output_root.mkdir(parents=True, exist_ok=True)
    final_graph_root = output_root / "graph"
    final_communities_root = output_root / "communities"
    final_evaluation_root = output_root / "evaluation"

    shutil.copytree(graph_root, final_graph_root, dirs_exist_ok=True)
    shutil.copytree(communities_root, final_communities_root, dirs_exist_ok=True)
    shutil.copytree(evaluation_root, final_evaluation_root, dirs_exist_ok=True)

    top_bot_rows = _select_top_communities(
        final_evaluation_root / "community_scores.csv",
        top_n=top_communities,
        reverse=True,
    )
    top_human_rows = _select_top_communities(
        final_evaluation_root / "community_scores.csv",
        top_n=top_communities,
        reverse=False,
    )

    top_bot_path = output_root / "top_bot_communities.csv"
    top_human_path = output_root / "top_human_communities.csv"
    _write_rows(top_bot_path, top_bot_rows)
    _write_rows(top_human_path, top_human_rows)

    manifest = {
        "sweep_root": str(sweep_root),
        "output_root": str(output_root),
        "selected_run": best_run,
        "selected_paths": {
            "run_root": str(run_root),
            "graph_root": str(graph_root),
            "communities_root": str(communities_root),
            "evaluation_root": str(evaluation_root),
        },
        "files": {
            "graph_root": str(final_graph_root),
            "communities_root": str(final_communities_root),
            "evaluation_root": str(final_evaluation_root),
            "top_bot_communities_csv": str(top_bot_path),
            "top_human_communities_csv": str(top_human_path),
            "summary_md": str(output_root / "best_run_summary.md"),
        },
    }
    write_json(output_root / "best_run_manifest.json", manifest)
    (output_root / "best_run_summary.md").write_text(
        _render_best_run_summary(manifest, top_bot_rows, top_human_rows),
        encoding="utf-8",
    )
    return manifest


def _base_run_name_from_run_name(run_name: str) -> str:
    if "_t" not in run_name:
        raise ValueError(f"Could not derive base run name from {run_name}")
    return run_name.rsplit("_t", 1)[0]


def _slugify_float(value: float) -> str:
    payload = f"{float(value):.6f}".rstrip("0").rstrip(".")
    if not payload:
        payload = "0"
    return payload.replace("-", "neg").replace(".", "p")


def _select_top_communities(path: Path, *, top_n: int, reverse: bool) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    rows.sort(
        key=lambda row: (
            float(row.get("bot_score") or 0.0),
            int(float(row.get("community_size") or 0)),
            row.get("community_id") or "",
        ),
        reverse=reverse,
    )
    return rows[: max(int(top_n), 0)]


def _write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _render_best_run_summary(
    manifest: dict[str, Any],
    top_bot_rows: list[dict[str, str]],
    top_human_rows: list[dict[str, str]],
) -> str:
    selected_run = manifest["selected_run"]
    lines = [
        "# Best Community Run Summary",
        "",
        "## Selected Run",
        f"- Run: {selected_run['run_name']}",
        f"- k: {selected_run['k']}",
        f"- Min similarity: {selected_run['min_similarity']}",
        f"- Min community size: {selected_run['min_community_size']}",
        f"- Threshold: {selected_run['threshold']}",
        f"- Communities: {selected_run['communities']}",
        f"- Largest community: {selected_run['largest_community']}",
        f"- Test F1: {selected_run['test_f1']}",
        f"- Test AUC: {selected_run['test_auc']}",
        "",
        "## Top Bot-Leaning Communities",
    ]
    for row in top_bot_rows[:10]:
        lines.append(
            f"- {row['community_id']}: size={row['community_size']}, "
            f"bot_score={row['bot_score']}, predicted={row['predicted_label']}"
        )
    lines.extend(["", "## Top Human-Leaning Communities"])
    for row in top_human_rows[:10]:
        lines.append(
            f"- {row['community_id']}: size={row['community_size']}, "
            f"bot_score={row['bot_score']}, predicted={row['predicted_label']}"
        )
    return "\n".join(lines) + "\n"

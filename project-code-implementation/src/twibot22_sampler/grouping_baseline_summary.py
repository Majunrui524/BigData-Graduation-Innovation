"""Summary table for purity-based grouping baselines."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .readers import write_csv, write_json

GROUPING_METHOD_ORDER = (
    "kmeans",
    "weighted_lpa",
    "structural_entropy",
)

DISPLAY_NAMES = {
    "kmeans": "K-Means",
    "weighted_lpa": "Weighted LPA",
    "structural_entropy": "Structural Entropy (Ours)",
}


def summarize_grouping_baselines(
    sample_root: Path,
    output_root: Path,
    *,
    kmeans_root: Path,
    weighted_lpa_purity_root: Path,
    structural_entropy_purity_root: Path,
) -> dict[str, Any]:
    """Build a paper-ready grouping-baseline table from purity manifests."""

    inputs = {
        "kmeans": kmeans_root,
        "weighted_lpa": weighted_lpa_purity_root,
        "structural_entropy": structural_entropy_purity_root,
    }
    rows = []
    for method_key in GROUPING_METHOD_ORDER:
        root = inputs[method_key]
        manifest_path = root / "community_purity_manifest.json"
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        metrics = manifest.get("metrics", {})
        test_metrics = metrics.get("test", {})
        rows.append(
            {
                "method_key": method_key,
                "method_name": DISPLAY_NAMES.get(method_key, str(manifest.get("method_name") or method_key)),
                "selection_split": "valid",
                "communities": int(manifest.get("counts", {}).get("communities", 0)),
                "global_purity": round(float(manifest.get("global_purity", 0.0)), 8),
                "test_accuracy": round(float(test_metrics.get("accuracy", 0.0)), 8),
                "test_precision": round(float(test_metrics.get("precision", 0.0)), 8),
                "test_recall": round(float(test_metrics.get("recall", 0.0)), 8),
                "test_f1": round(float(test_metrics.get("f1", 0.0)), 8),
                "test_auc": round(float(test_metrics.get("auc", 0.0)), 8),
                "selected_params": json.dumps(manifest.get("selected_params", {}), ensure_ascii=False, sort_keys=True),
                "source_root": str(root),
            }
        )

    output_root.mkdir(parents=True, exist_ok=True)
    csv_path = output_root / "grouping_baseline_results.csv"
    markdown_path = output_root / "grouping_baseline_results.md"
    manifest_path = output_root / "grouping_baseline_manifest.json"

    write_csv(
        csv_path,
        [
            "method_key",
            "method_name",
            "selection_split",
            "communities",
            "global_purity",
            "test_accuracy",
            "test_precision",
            "test_recall",
            "test_f1",
            "test_auc",
            "selected_params",
            "source_root",
        ],
        rows,
    )
    markdown_path.write_text(_render_summary_markdown(rows), encoding="utf-8")
    manifest = {
        "sample_root": str(sample_root),
        "output_root": str(output_root),
        "counts": {
            "methods": len(rows),
        },
        "files": {
            "results_csv": str(csv_path),
            "results_md": str(markdown_path),
        },
    }
    write_json(manifest_path, manifest)
    return manifest


def _render_summary_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# 10k Grouping Baseline Comparison",
        "",
        "| Method | Communities | Global Purity | ACC | Precision | Recall | F1 | AUC |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['method_name']} | {row['communities']} | {row['global_purity']} | "
            f"{row['test_accuracy']} | {row['test_precision']} | {row['test_recall']} | "
            f"{row['test_f1']} | {row['test_auc']} |"
        )
    lines.extend(
        [
            "",
            "Notes:",
            "- The main comparison is grouping-method-centered rather than reranker-centered.",
            "- Community labels are projected using train-split majority labels within each discovered group.",
            "- `Structural Entropy (Ours)` should be treated as the primary result in the main paper body.",
            "",
        ]
    )
    return "\n".join(lines)

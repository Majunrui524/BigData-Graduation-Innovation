"""Aggregate completed external baselines into a comparison table."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .external_baseline_common import build_summary_rows, write_summary_bundle


def summarize_external_baselines(
    sample_root: Path,
    baselines_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Summarize external baseline outputs into CSV/Markdown artifacts."""

    rows = build_summary_rows(baselines_root)
    if not rows:
        raise ValueError(f"No completed external baseline outputs found under {baselines_root}")
    return write_summary_bundle(output_root, rows, sample_root=sample_root, baselines_root=baselines_root)

"""Validation utilities for exported sample subsets."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from . import config
from .normalize import normalize_label, normalize_split
from .readers import read_csv_rows, read_jsonl_records, read_manifest


def validate_sample(sample_root: Path, report_out: Path | None = None) -> dict[str, Any]:
    """Validate a sampled subset and optionally write a markdown report."""

    manifest = read_manifest(sample_root / "sample_manifest.json")
    user_ids = {str(record["id"]) for record in read_jsonl_records(sample_root / "user.jsonl") if "id" in record}
    tweet_ids = {str(record["id"]) for record in read_jsonl_records(sample_root / "tweet_0.jsonl") if "id" in record}

    split_rows = list(read_csv_rows(sample_root / "split.csv"))
    label_rows = list(read_csv_rows(sample_root / "label.csv"))
    edge_rows = list(read_csv_rows(sample_root / "edge.csv"))

    split_distribution = Counter(normalize_split(row.get("split")) for row in split_rows)
    label_distribution = Counter(normalize_label(row.get("label")) for row in label_rows)
    endpoint_errors = _validate_edge_endpoints(edge_rows, user_ids, tweet_ids)
    split_ratio_delta = _ratio_delta(
        actual=split_distribution,
        target=Counter(_flatten_counts(manifest.get("seed_sampling", {}).get("split_quotas", {}))),
    )
    label_ratio_delta = _ratio_delta(
        actual=label_distribution,
        target=Counter(_flatten_nested_counts(manifest.get("seed_sampling", {}).get("label_quotas", {}))),
    )

    tweet_budget = manifest.get("tweet_budget", {})
    tweet_budget_ok = len(tweet_ids) <= int(tweet_budget.get("limit", 0) or 0)
    passed = (
        not endpoint_errors
        and split_ratio_delta <= config.VALIDATE_MAX_SPLIT_RATIO_DELTA
        and label_ratio_delta <= config.VALIDATE_MAX_LABEL_RATIO_DELTA
        and tweet_budget_ok
    )

    result = {
        "passed": passed,
        "endpoint_error_count": len(endpoint_errors),
        "split_ratio_delta": split_ratio_delta,
        "label_ratio_delta": label_ratio_delta,
        "tweet_budget_ok": tweet_budget_ok,
        "tweet_budget_limit": tweet_budget.get("limit", 0),
        "tweet_budget_used": len(tweet_ids),
        "endpoint_errors": endpoint_errors[:20],
    }
    if report_out is not None:
        report_out.parent.mkdir(parents=True, exist_ok=True)
        report_out.write_text(_render_validation_report(result), encoding="utf-8")
    return result


def _validate_edge_endpoints(
    edge_rows: list[dict[str, str]],
    user_ids: set[str],
    tweet_ids: set[str],
) -> list[str]:
    errors: list[str] = []
    for row in edge_rows:
        source_id = str(row.get("source_id", ""))
        target_id = str(row.get("target_id", ""))
        relation = str(row.get("relation", ""))
        if relation == "following":
            if source_id not in user_ids or target_id not in user_ids:
                errors.append(f"following endpoint missing: {source_id}->{target_id}")
        elif relation == "post":
            if source_id not in user_ids or target_id not in tweet_ids:
                errors.append(f"post endpoint missing: {source_id}->{target_id}")
        elif relation in {"retweet", "quote", "reply"}:
            if source_id not in tweet_ids or target_id not in tweet_ids:
                errors.append(f"{relation} endpoint missing: {source_id}->{target_id}")
        elif relation == "mention":
            if source_id not in tweet_ids or target_id not in user_ids:
                errors.append(f"mention endpoint missing: {source_id}->{target_id}")
    return errors


def _ratio_delta(actual: Counter[str], target: Counter[str]) -> float:
    if not actual or not target:
        return 0.0
    actual_total = sum(actual.values())
    target_total = sum(target.values())
    delta = 0.0
    keys = set(actual) | set(target)
    for key in keys:
        delta = max(
            delta,
            abs((actual.get(key, 0) / actual_total) - (target.get(key, 0) / target_total)),
        )
    return delta


def _flatten_counts(counts: dict[str, Any]) -> dict[str, int]:
    return {str(key): int(value) for key, value in counts.items()}


def _flatten_nested_counts(counts: dict[str, Any]) -> dict[str, int]:
    flattened: Counter[str] = Counter()
    for _outer_key, nested in counts.items():
        for key, value in nested.items():
            flattened[str(key)] += int(value)
    return dict(flattened)


def _render_validation_report(result: dict[str, Any]) -> str:
    lines = [
        "# Validation Report",
        "",
        f"- Passed: {result['passed']}",
        f"- Endpoint errors: {result['endpoint_error_count']}",
        f"- Split ratio delta: {result['split_ratio_delta']:.6f}",
        f"- Label ratio delta: {result['label_ratio_delta']:.6f}",
        f"- Tweet budget ok: {result['tweet_budget_ok']}",
        f"- Tweet budget limit: {result['tweet_budget_limit']}",
        f"- Tweet budget used: {result['tweet_budget_used']}",
    ]
    if result["endpoint_errors"]:
        lines.extend(["", "## Sample Endpoint Errors"])
        lines.extend(f"- {message}" for message in result["endpoint_errors"])
    return "\n".join(lines) + "\n"

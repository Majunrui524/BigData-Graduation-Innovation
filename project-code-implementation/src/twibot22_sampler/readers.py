"""Readers for TwiBot-style raw files and exported sample files."""

from __future__ import annotations

import csv
import json
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable, Iterator

from . import config

try:
    import ijson
except ImportError:  # pragma: no cover - dependency is optional during import
    ijson = None


def require_path(path: Path) -> Path:
    """Raise a helpful error when a required file is missing."""

    if not path.exists():
        raise FileNotFoundError(f"Expected file does not exist: {path}")
    return path


def resolve_user_path(data_root: Path) -> Path:
    """Locate the user data file."""

    for candidate in (
        data_root / "user.json",
        data_root / "user.jsonl",
    ):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not locate user.json under {data_root}")


def resolve_edge_path(data_root: Path) -> Path:
    """Locate edge.csv."""

    return require_path(data_root / config.EDGE_CSV_BASENAME)


def resolve_split_path(data_root: Path) -> Path:
    """Locate split.csv."""

    return require_path(data_root / config.SPLIT_CSV_BASENAME)


def resolve_label_path(data_root: Path) -> Path:
    """Locate label.csv."""

    return require_path(data_root / config.LABEL_CSV_BASENAME)


def resolve_tweet_paths(data_root: Path) -> list[Path]:
    """Locate tweet shards in a stable order."""

    candidates = sorted(data_root.glob("tweet_*.json")) + sorted(data_root.glob("tweet_*.jsonl"))
    if not candidates:
        raise FileNotFoundError(f"Could not locate tweet shards under {data_root}")
    return sorted(set(candidates))


def read_csv_rows(path: Path) -> Iterator[dict[str, str]]:
    """Yield CSV rows as dictionaries."""

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            yield {str(key): (value if value is not None else "") for key, value in row.items()}


def read_label_map(path: Path) -> dict[str, str]:
    """Load label.csv into a map."""

    mapping: dict[str, str] = {}
    for row in read_csv_rows(path):
        identifier = row.get("id") or row.get("user_id")
        if identifier:
            mapping[str(identifier)] = row.get("label", "")
    return mapping


def read_split_map(path: Path) -> dict[str, str]:
    """Load split.csv into a map."""

    mapping: dict[str, str] = {}
    for row in read_csv_rows(path):
        identifier = row.get("id") or row.get("user_id")
        if identifier:
            mapping[str(identifier)] = row.get("split", "")
    return mapping


def iter_json_records(path: Path) -> Iterator[dict[str, Any]]:
    """Iterate records from JSON array, JSON object, or JSONL files."""

    if path.suffix == ".jsonl":
        yield from _iter_jsonl(path)
        return
    first_char = _peek_non_whitespace(path)
    if first_char == "[":
        yield from _iter_json_array(path)
        return
    if first_char == "{":
        yield from _iter_json_object(path)
        return
    yield from _iter_jsonl(path)


def _peek_non_whitespace(path: Path) -> str:
    with path.open("r", encoding="utf-8") as handle:
        while True:
            char = handle.read(1)
            if not char:
                return ""
            if not char.isspace():
                return char


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if isinstance(payload, dict):
                yield payload


def _iter_json_array(path: Path) -> Iterator[dict[str, Any]]:
    if ijson is None:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, list):
            for item in payload:
                if isinstance(item, dict):
                    yield item
        return
    with path.open("rb") as handle:
        for item in ijson.items(handle, "item"):
            if isinstance(item, dict):
                yield item


def _iter_json_object(path: Path) -> Iterator[dict[str, Any]]:
    if ijson is None:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        yield from _iter_loaded_object(payload)
        return
    with path.open("rb") as handle:
        for key, value in ijson.kvitems(handle, ""):
            if key in config.WRAPPER_KEYS and isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        yield item
                continue
            if key in config.WRAPPER_KEYS and isinstance(value, dict):
                for nested_key, nested_value in value.items():
                    if isinstance(nested_value, dict):
                        nested_value.setdefault("id", nested_key)
                        yield nested_value
                continue
            if isinstance(value, dict):
                value.setdefault("id", key)
                yield value


def _iter_loaded_object(payload: Any) -> Iterator[dict[str, Any]]:
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key in config.WRAPPER_KEYS and isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        yield item
                continue
            if key in config.WRAPPER_KEYS and isinstance(value, dict):
                for nested_key, nested_value in value.items():
                    if isinstance(nested_value, dict):
                        nested_value.setdefault("id", nested_key)
                        yield nested_value
                continue
            if isinstance(value, dict):
                value.setdefault("id", key)
                yield value
        return
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                yield item


def read_jsonl_records(path: Path) -> Iterator[dict[str, Any]]:
    """Explicit JSONL reader used for exported sample files."""

    yield from _iter_jsonl(path)


def read_manifest(path: Path) -> dict[str, Any]:
    """Read a JSON manifest."""

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON file with stable formatting."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(payload), handle, indent=2, ensure_ascii=False, sort_keys=True)


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    """Write JSONL and return the number of records written."""

    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(_json_safe(record), ensure_ascii=False, sort_keys=True))
            handle.write("\n")
            count += 1
    return count


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> int:
    """Write CSV rows and return the count."""

    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            count += 1
    return count


def _json_safe(value: Any) -> Any:
    """Recursively convert values into JSON-serializable primitives."""

    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, set):
        return [_json_safe(item) for item in sorted(value, key=str)]
    if isinstance(value, Decimal):
        if value == value.to_integral_value():
            return int(value)
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return value

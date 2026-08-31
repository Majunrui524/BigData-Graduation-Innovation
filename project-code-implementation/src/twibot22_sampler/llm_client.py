"""Minimal OpenAI-compatible LLM client built on the standard library."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any
from urllib import error, request

DEFAULT_BASE_URL = "https://api.openai.com/v1"


@dataclass(frozen=True)
class LLMSettings:
    """Runtime settings for the LLM client."""

    model: str
    api_key: str
    base_url: str = DEFAULT_BASE_URL
    timeout_s: float = 60.0
    max_retries: int = 3
    retry_backoff_s: float = 2.0
    temperature: float = 0.0
    concurrency: int = 1
    requests_per_minute: int | None = None


class OpenAICompatibleClient:
    """Tiny client for chat-completions style endpoints."""

    def __init__(self, settings: LLMSettings) -> None:
        self.settings = settings
        self._rate_limiter = SlidingWindowRateLimiter(settings.requests_per_minute)

    def chat_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 800,
    ) -> dict[str, Any]:
        """Send a chat-completions request and parse the JSON reply."""

        payload = {
            "model": self.settings.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.settings.temperature,
            "max_tokens": max_tokens,
        }
        url = self.settings.base_url.rstrip("/")
        if not url.endswith("/chat/completions"):
            url = f"{url}/chat/completions"

        last_error: Exception | None = None
        for attempt in range(self.settings.max_retries + 1):
            try:
                self._rate_limiter.acquire()
                response_payload = _post_json(
                    url,
                    payload,
                    api_key=self.settings.api_key,
                    timeout_s=self.settings.timeout_s,
                )
                content = extract_message_content(response_payload)
                parsed = extract_json_object(content)
                return {
                    "raw_response": response_payload,
                    "content": content,
                    "parsed": parsed,
                }
            except Exception as exc:  # pragma: no cover - network path
                last_error = exc
                if attempt >= self.settings.max_retries:
                    break
                time.sleep(self.settings.retry_backoff_s * (attempt + 1))
        assert last_error is not None
        raise last_error

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Send an embeddings request and return vectors in input order."""

        payload = {
            "model": self.settings.model,
            "input": texts,
        }
        url = self.settings.base_url.rstrip("/")
        if url.endswith("/chat/completions"):
            url = url[: -len("/chat/completions")]
        if not url.endswith("/embeddings"):
            url = f"{url}/embeddings"

        last_error: Exception | None = None
        for attempt in range(self.settings.max_retries + 1):
            try:
                self._rate_limiter.acquire()
                response_payload = _post_json(
                    url,
                    payload,
                    api_key=self.settings.api_key,
                    timeout_s=self.settings.timeout_s,
                )
                return extract_embedding_vectors(response_payload)
            except Exception as exc:  # pragma: no cover - network path
                last_error = exc
                if attempt >= self.settings.max_retries:
                    break
                time.sleep(self.settings.retry_backoff_s * (attempt + 1))
        assert last_error is not None
        raise last_error


def load_llm_settings(
    *,
    model: str | None = None,
    model_env: str = "OPENAI_MODEL",
    default_model: str = "gpt-4o-mini",
    base_url: str | None = None,
    api_key_env: str = "OPENAI_API_KEY",
    env_file: str | os.PathLike[str] | None = None,
    env_search_roots: tuple[Path, ...] = (),
    timeout_s: float = 60.0,
    max_retries: int = 3,
    temperature: float = 0.0,
    concurrency: int | None = None,
    requests_per_minute: int | None = None,
) -> LLMSettings:
    """Load LLM settings from the environment plus optional CLI overrides."""

    resolved_env_file = resolve_env_file(env_file, search_roots=env_search_roots)
    if resolved_env_file is not None:
        load_env_file(resolved_env_file)

    api_key = os.getenv(api_key_env, "").strip()
    if not api_key:
        raise ValueError(
            f"Environment variable {api_key_env} is required for LLM calls"
            f"{_format_env_hint(resolved_env_file, env_file)}"
        )
    resolved_model = model or os.getenv(model_env, default_model)
    resolved_base_url = base_url or os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE") or DEFAULT_BASE_URL
    resolved_concurrency = _coerce_positive_int(
        concurrency if concurrency is not None else os.getenv("OPENAI_CONCURRENCY"),
        default=1,
        field_name="concurrency",
    )
    resolved_requests_per_minute = _coerce_optional_positive_int(
        requests_per_minute
        if requests_per_minute is not None
        else (os.getenv("OPENAI_REQUESTS_PER_MINUTE") or os.getenv("OPENAI_MAX_REQUESTS_PER_MINUTE")),
        field_name="requests_per_minute",
    )
    return LLMSettings(
        model=resolved_model,
        api_key=api_key,
        base_url=resolved_base_url,
        timeout_s=timeout_s,
        max_retries=max_retries,
        temperature=temperature,
        concurrency=resolved_concurrency,
        requests_per_minute=resolved_requests_per_minute,
    )


def resolve_env_file(
    env_file: str | os.PathLike[str] | None,
    *,
    search_roots: tuple[Path, ...] = (),
) -> Path | None:
    """Resolve an explicit env file or discover a project-local `.env`."""

    if env_file:
        candidate = Path(env_file).expanduser()
        if not candidate.is_absolute():
            candidate = (Path.cwd() / candidate).resolve()
        else:
            candidate = candidate.resolve()
        if not candidate.exists():
            raise FileNotFoundError(f"Env file not found: {candidate}")
        if not candidate.is_file():
            raise ValueError(f"Env path is not a file: {candidate}")
        return candidate

    seen: set[Path] = set()
    for anchor in (Path.cwd(), *search_roots):
        for directory in _iter_project_search_dirs(anchor):
            if directory in seen:
                continue
            seen.add(directory)
            candidate = directory / ".env"
            if candidate.is_file():
                return candidate
    return None


def load_env_file(path: str | os.PathLike[str]) -> dict[str, str]:
    """Load simple KEY=VALUE pairs from a dotenv file into the environment."""

    env_path = Path(path).expanduser()
    if not env_path.is_absolute():
        env_path = (Path.cwd() / env_path).resolve()
    else:
        env_path = env_path.resolve()
    loaded: dict[str, str] = {}
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        parsed_value = _parse_env_value(value.strip())
        os.environ[key] = parsed_value
        loaded[key] = parsed_value
    return loaded


def _iter_project_search_dirs(anchor: Path) -> list[Path]:
    """Yield an anchor and its parents until the project root or filesystem root."""

    resolved_anchor = anchor.expanduser().resolve()
    directories = [resolved_anchor]
    current = resolved_anchor
    while True:
        if (current / "pyproject.toml").exists():
            break
        parent = current.parent
        if parent == current:
            break
        directories.append(parent)
        current = parent
    return directories


def _parse_env_value(value: str) -> str:
    if not value:
        return ""
    if value[0] == value[-1] and value[0] in {'"', "'"} and len(value) >= 2:
        value = value[1:-1]
    return value.replace("\\n", "\n")


def _format_env_hint(resolved_env_file: Path | None, explicit_env_file: str | os.PathLike[str] | None) -> str:
    if resolved_env_file is not None:
        return f" (loaded config from {resolved_env_file})"
    if explicit_env_file:
        return ""
    return "; set it in the environment or add it to a project-local .env file"


def _coerce_positive_int(value: Any, *, default: int, field_name: str) -> int:
    if value in (None, ""):
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer, got {value!r}") from exc
    if parsed < 1:
        raise ValueError(f"{field_name} must be >= 1, got {parsed}")
    return parsed


def _coerce_optional_positive_int(value: Any, *, field_name: str) -> int | None:
    if value in (None, "", 0, "0"):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer, got {value!r}") from exc
    if parsed < 1:
        raise ValueError(f"{field_name} must be >= 1 when set, got {parsed}")
    return parsed


class SlidingWindowRateLimiter:
    """Thread-safe fixed-window limiter for outbound API calls."""

    def __init__(self, requests_per_minute: int | None) -> None:
        self.requests_per_minute = requests_per_minute
        self._timestamps: list[float] = []
        self._lock = Lock()

    def acquire(self) -> None:
        if not self.requests_per_minute:
            return
        while True:
            wait_time = 0.0
            with self._lock:
                now = time.monotonic()
                cutoff = now - 60.0
                self._timestamps = [stamp for stamp in self._timestamps if stamp > cutoff]
                if len(self._timestamps) < self.requests_per_minute:
                    self._timestamps.append(now)
                    return
                oldest = min(self._timestamps)
                wait_time = max(0.0, 60.0 - (now - oldest))
            if wait_time > 0:
                time.sleep(wait_time)


def extract_message_content(payload: dict[str, Any]) -> str:
    """Extract the assistant content from a chat-completions response."""

    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("LLM response did not contain choices")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise ValueError("LLM response did not contain a message")
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        if parts:
            return "\n".join(parts)
    raise ValueError("LLM response did not contain text content")


def extract_json_object(text: str) -> dict[str, Any]:
    """Extract a JSON object from plain text or fenced markdown."""

    stripped = text.strip()
    candidates = [stripped]
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3:
            candidates.append("\n".join(lines[1:-1]).strip())
    if "```json" in stripped:
        start = stripped.find("```json")
        tail = stripped[start + len("```json") :]
        end = tail.find("```")
        if end != -1:
            candidates.append(tail[:end].strip())

    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload

    start = stripped.find("{")
    if start == -1:
        raise ValueError("Could not find JSON object in LLM response")
    depth = 0
    in_string = False
    escaped = False
    for index, char in enumerate(stripped[start:], start=start):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                payload = json.loads(stripped[start : index + 1])
                if isinstance(payload, dict):
                    return payload
                break
    raise ValueError("Could not parse JSON object from LLM response")


def extract_embedding_vectors(payload: dict[str, Any]) -> list[list[float]]:
    """Extract embedding vectors from an embeddings API response."""

    data = payload.get("data")
    if not isinstance(data, list):
        raise ValueError("Embedding response did not contain a data list")
    vectors: list[list[float]] = []
    for item in data:
        if not isinstance(item, dict):
            raise ValueError("Embedding response contained a non-object item")
        embedding = item.get("embedding")
        if not isinstance(embedding, list):
            raise ValueError("Embedding response item did not contain an embedding list")
        vector: list[float] = []
        for value in embedding:
            try:
                vector.append(float(value))
            except (TypeError, ValueError) as exc:
                raise ValueError("Embedding vector contained a non-numeric value") from exc
        vectors.append(vector)
    return vectors


def _post_json(url: str, payload: dict[str, Any], *, api_key: str, timeout_s: float) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url,
        method="POST",
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    try:
        with request.urlopen(req, timeout=timeout_s) as resp:  # pragma: no cover - network path
            data = resp.read().decode("utf-8")
    except error.HTTPError as exc:  # pragma: no cover - network path
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"LLM request failed with HTTP {exc.code}: {detail}") from exc
    except error.URLError as exc:  # pragma: no cover - network path
        raise RuntimeError(f"LLM request failed: {exc}") from exc
    payload = json.loads(data)
    if not isinstance(payload, dict):
        raise ValueError("LLM response was not a JSON object")
    return payload

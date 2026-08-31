from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from twibot22_sampler.llm_client import extract_embedding_vectors, extract_json_object, load_llm_settings


class LLMClientTests(unittest.TestCase):
    def test_extract_json_object_supports_fenced_payload(self) -> None:
        payload = extract_json_object(
            """```json
{"compressed_text":"hello","triplets":[{"subject":"a","predicate":"b","object":"c"}]}
```"""
        )
        self.assertEqual(payload["compressed_text"], "hello")
        self.assertEqual(payload["triplets"][0]["subject"], "a")

    def test_extract_json_object_supports_embedded_json(self) -> None:
        payload = extract_json_object('Result:\n{"coarse_type":"original","detail_type":"original"}')
        self.assertEqual(payload["coarse_type"], "original")

    def test_extract_embedding_vectors_parses_embeddings_payload(self) -> None:
        vectors = extract_embedding_vectors(
            {
                "data": [
                    {"embedding": [0.1, 0.2], "index": 0},
                    {"embedding": [0.3, 0.4], "index": 1},
                ]
            }
        )
        self.assertEqual(vectors, [[0.1, 0.2], [0.3, 0.4]])

    def test_load_llm_settings_reads_explicit_env_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            env_path = Path(tmp_dir) / ".env"
            env_path.write_text(
                "OPENAI_API_KEY=test-key\nOPENAI_MODEL=test-model\nOPENAI_BASE_URL=https://example.com/v1\n",
                encoding="utf-8",
            )
            with mock.patch.dict(os.environ, {}, clear=True):
                settings = load_llm_settings(env_file=env_path)
        self.assertEqual(settings.api_key, "test-key")
        self.assertEqual(settings.model, "test-model")
        self.assertEqual(settings.base_url, "https://example.com/v1")

    def test_load_llm_settings_auto_discovers_project_env(self) -> None:
        original_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_root = Path(tmp_dir)
            (project_root / "pyproject.toml").write_text("[project]\nname='tmp'\n", encoding="utf-8")
            (project_root / ".env").write_text("OPENAI_API_KEY=auto-key\n", encoding="utf-8")
            nested_dir = project_root / "data" / "samples" / "final_v1"
            nested_dir.mkdir(parents=True)
            try:
                os.chdir(nested_dir)
                with mock.patch.dict(os.environ, {}, clear=True):
                    settings = load_llm_settings(env_search_roots=(nested_dir,))
            finally:
                os.chdir(original_cwd)
        self.assertEqual(settings.api_key, "auto-key")
        self.assertEqual(settings.model, "gpt-4o-mini")

    def test_cli_overrides_take_priority_over_env_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            env_path = Path(tmp_dir) / ".env"
            env_path.write_text(
                "OPENAI_API_KEY=test-key\nOPENAI_MODEL=env-model\nOPENAI_BASE_URL=https://env.example/v1\nOPENAI_CONCURRENCY=2\nOPENAI_REQUESTS_PER_MINUTE=60\n",
                encoding="utf-8",
            )
            with mock.patch.dict(os.environ, {}, clear=True):
                settings = load_llm_settings(
                    env_file=env_path,
                    model="cli-model",
                    base_url="https://cli.example/v1",
                    concurrency=5,
                    requests_per_minute=150,
                )
        self.assertEqual(settings.api_key, "test-key")
        self.assertEqual(settings.model, "cli-model")
        self.assertEqual(settings.base_url, "https://cli.example/v1")
        self.assertEqual(settings.concurrency, 5)
        self.assertEqual(settings.requests_per_minute, 150)

    def test_load_llm_settings_reads_concurrency_and_rate_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            env_path = Path(tmp_dir) / ".env"
            env_path.write_text(
                "OPENAI_API_KEY=test-key\nOPENAI_CONCURRENCY=3\nOPENAI_REQUESTS_PER_MINUTE=90\n",
                encoding="utf-8",
            )
            with mock.patch.dict(os.environ, {}, clear=True):
                settings = load_llm_settings(env_file=env_path)
        self.assertEqual(settings.concurrency, 3)
        self.assertEqual(settings.requests_per_minute, 90)

    def test_load_llm_settings_supports_custom_model_env(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            env_path = Path(tmp_dir) / ".env"
            env_path.write_text(
                "OPENAI_API_KEY=test-key\nOPENAI_EMBEDDING_MODEL=text-embedding-3-small\n",
                encoding="utf-8",
            )
            with mock.patch.dict(os.environ, {}, clear=True):
                settings = load_llm_settings(
                    env_file=env_path,
                    model_env="OPENAI_EMBEDDING_MODEL",
                    default_model="fallback-embedding",
                )
        self.assertEqual(settings.model, "text-embedding-3-small")


if __name__ == "__main__":
    unittest.main()

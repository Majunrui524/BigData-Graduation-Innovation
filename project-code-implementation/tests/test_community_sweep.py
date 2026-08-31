from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from twibot22_sampler.community_sweep import sweep_community_pipeline
from twibot22_sampler.readers import read_csv_rows, write_csv, write_jsonl


class CommunitySweepTests(unittest.TestCase):
    def test_sweep_community_pipeline_writes_ranked_results(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            sample_root = root / "sample"
            vector_root = sample_root / "analysis" / "user_vectors"
            output_root = sample_root / "analysis" / "community_sweep"
            vector_root.mkdir(parents=True)
            sample_root.mkdir(parents=True, exist_ok=True)

            write_jsonl(
                sample_root / "user.jsonl",
                [
                    {"id": "u1"},
                    {"id": "u2"},
                    {"id": "u3"},
                    {"id": "u4"},
                ],
            )
            write_csv(
                sample_root / "split.csv",
                ["id", "split"],
                [
                    {"id": "u1", "split": "train"},
                    {"id": "u2", "split": "train"},
                    {"id": "u3", "split": "valid"},
                    {"id": "u4", "split": "test"},
                ],
            )
            write_csv(
                sample_root / "label.csv",
                ["id", "label"],
                [
                    {"id": "u1", "label": "bot"},
                    {"id": "u2", "label": "human"},
                    {"id": "u3", "label": "bot"},
                    {"id": "u4", "label": "human"},
                ],
            )
            (vector_root / "user_fused_vectors.jsonl").write_text(
                "\n".join(
                    json.dumps(row)
                    for row in [
                        {"user_id": "u1", "fused_vector": [1.0, 0.0, 0.0]},
                        {"user_id": "u2", "fused_vector": [-1.0, 0.0, 0.0]},
                        {"user_id": "u3", "fused_vector": [0.95, 0.0, 0.0]},
                        {"user_id": "u4", "fused_vector": [-0.95, 0.0, 0.0]},
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            manifest = sweep_community_pipeline(
                sample_root,
                output_root,
                vector_root=vector_root,
                k_values=[1],
                min_similarity_values=[0.0],
                min_community_size_values=[1],
                threshold_values=[0.4, 0.5],
                algorithm_values=["weighted_lpa"],
                fusion_mode="early",
                graph_backend="python",
            )

            self.assertEqual(manifest["run_count"], 2)
            self.assertEqual(manifest["objective"]["split"], "valid")
            self.assertIn("best_run", manifest)
            self.assertTrue((output_root / "community_sweep_results.csv").exists())

            rows = list(read_csv_rows(output_root / "community_sweep_results.csv"))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["run_name"], manifest["best_run"]["run_name"])


if __name__ == "__main__":
    unittest.main()

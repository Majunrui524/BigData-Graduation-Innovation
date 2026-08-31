from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from twibot22_sampler.external_graph_baselines import run_graph_baselines
from twibot22_sampler.readers import read_csv_rows, write_csv


class ExternalGraphBaselineTests(unittest.TestCase):
    def test_run_graph_baselines_writes_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            sample_root = root / "sample"
            output_root = sample_root / "analysis" / "external_baselines_10k"
            sample_root.mkdir(parents=True)

            write_csv(
                sample_root / "split.csv",
                ["id", "split"],
                [
                    {"id": "u1", "split": "train"},
                    {"id": "u2", "split": "train"},
                    {"id": "u3", "split": "valid"},
                    {"id": "u4", "split": "valid"},
                    {"id": "u5", "split": "test"},
                    {"id": "u6", "split": "test"},
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
                    {"id": "u5", "label": "bot"},
                    {"id": "u6", "label": "human"},
                ],
            )
            write_csv(
                sample_root / "edge.csv",
                ["source_id", "target_id", "relation"],
                [
                    {"source_id": "u1", "target_id": "u3", "relation": "following"},
                    {"source_id": "u3", "target_id": "u5", "relation": "following"},
                    {"source_id": "u2", "target_id": "u4", "relation": "following"},
                    {"source_id": "u4", "target_id": "u6", "relation": "following"},
                ],
            )

            manifests = run_graph_baselines(
                sample_root,
                output_root,
                dimension=16,
                walk_length=8,
                num_walks=4,
                window=3,
                epochs=2,
                lr_c_values=(1.0,),
                class_weight_values=(None,),
                node2vec_p_values=(1.0,),
                node2vec_q_values=(1.0,),
            )
            self.assertIn("deepwalk_lr", manifests)
            self.assertIn("node2vec_lr", manifests)
            self.assertTrue((output_root / "deepwalk_lr" / "embeddings.npy").exists())
            self.assertTrue((output_root / "node2vec_lr" / "metrics.json").exists())

            rows = list(read_csv_rows(output_root / "node2vec_lr" / "predictions.csv"))
            self.assertEqual(len(rows), 6)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from twibot22_sampler.community_purity import evaluate_community_purity
from twibot22_sampler.readers import read_csv_rows, write_csv, write_json


class CommunityPurityTests(unittest.TestCase):
    def test_evaluate_community_purity_writes_expected_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            sample_root = root / "sample"
            communities_root = sample_root / "analysis" / "communities"
            output_root = sample_root / "analysis" / "community_purity"
            communities_root.mkdir(parents=True)

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
            write_csv(
                communities_root / "community_assignments.csv",
                ["user_id", "community_id", "community_size", "split", "label"],
                [
                    {"user_id": "u1", "community_id": "c0001", "community_size": 2, "split": "train", "label": "bot"},
                    {"user_id": "u3", "community_id": "c0001", "community_size": 2, "split": "valid", "label": "bot"},
                    {"user_id": "u2", "community_id": "c0002", "community_size": 2, "split": "train", "label": "human"},
                    {"user_id": "u4", "community_id": "c0002", "community_size": 2, "split": "test", "label": "human"},
                ],
            )
            write_json(
                communities_root / "community_manifest.json",
                {"algorithm": "structural_entropy"},
            )
            write_json(
                communities_root / "encoding_tree.json",
                {
                    "algorithm": "structural_entropy",
                    "roots": ["node:r1", "node:r2"],
                    "nodes": {
                        "node:r1": {"node_id": "node:r1", "type": "merge", "children": ["leaf:u1", "leaf:u3"], "size": 2},
                        "node:r2": {"node_id": "node:r2", "type": "merge", "children": ["leaf:u2", "leaf:u4"], "size": 2},
                        "leaf:u1": {"node_id": "leaf:u1", "type": "leaf", "user_id": "u1", "size": 1},
                        "leaf:u2": {"node_id": "leaf:u2", "type": "leaf", "user_id": "u2", "size": 1},
                        "leaf:u3": {"node_id": "leaf:u3", "type": "leaf", "user_id": "u3", "size": 1},
                        "leaf:u4": {"node_id": "leaf:u4", "type": "leaf", "user_id": "u4", "size": 1},
                    },
                },
            )

            manifest = evaluate_community_purity(sample_root, communities_root, output_root)
            self.assertEqual(manifest["counts"]["communities"], 2)
            self.assertGreater(manifest["global_purity"], 0.0)
            rows = list(read_csv_rows(output_root / "community_purity_summary.csv"))
            self.assertEqual(len(rows), 2)
            self.assertIn("purity", rows[0])
            self.assertIn("predicted_label_by_train_majority", rows[0])


if __name__ == "__main__":
    unittest.main()

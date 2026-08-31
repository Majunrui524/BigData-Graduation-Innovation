from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from twibot22_sampler.community_structure import analyze_community_structure
from twibot22_sampler.readers import read_csv_rows, write_csv


class CommunityStructureTests(unittest.TestCase):
    def test_analyze_community_structure_writes_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            sample_root = root / "sample"
            communities_root = sample_root / "analysis" / "communities"
            graph_root = sample_root / "analysis" / "user_graph"
            purity_root = sample_root / "analysis" / "community_purity"
            output_root = sample_root / "analysis" / "community_structure"
            communities_root.mkdir(parents=True)
            graph_root.mkdir(parents=True)
            purity_root.mkdir(parents=True)

            write_csv(
                communities_root / "community_assignments.csv",
                ["user_id", "community_id", "community_size", "split", "label"],
                [
                    {"user_id": "u1", "community_id": "c0001", "community_size": 3, "split": "train", "label": "human"},
                    {"user_id": "u2", "community_id": "c0001", "community_size": 3, "split": "valid", "label": "human"},
                    {"user_id": "u3", "community_id": "c0001", "community_size": 3, "split": "test", "label": "human"},
                    {"user_id": "u4", "community_id": "c0002", "community_size": 2, "split": "train", "label": "bot"},
                    {"user_id": "u5", "community_id": "c0002", "community_size": 2, "split": "test", "label": "bot"},
                ],
            )
            write_csv(
                graph_root / "user_knn_edges.csv",
                ["source_user_id", "target_user_id", "weight", "support"],
                [
                    {"source_user_id": "u1", "target_user_id": "u2", "weight": 0.8, "support": 1},
                    {"source_user_id": "u2", "target_user_id": "u3", "weight": 0.8, "support": 1},
                    {"source_user_id": "u1", "target_user_id": "u3", "weight": 0.8, "support": 1},
                    {"source_user_id": "u4", "target_user_id": "u5", "weight": 0.7, "support": 1},
                ],
            )
            write_csv(
                purity_root / "community_purity_summary.csv",
                [
                    "community_id",
                    "community_size",
                    "train_human_count",
                    "train_bot_count",
                    "train_labeled_count",
                    "all_human_count",
                    "all_bot_count",
                    "all_labeled_count",
                    "bot_ratio",
                    "purity",
                    "bot_score",
                    "predicted_label_by_train_majority",
                    "label_source",
                    "encoding_depth",
                    "train_count",
                    "valid_count",
                    "test_count",
                ],
                [
                    {
                        "community_id": "c0001",
                        "community_size": 3,
                        "train_human_count": 1,
                        "train_bot_count": 0,
                        "train_labeled_count": 1,
                        "all_human_count": 3,
                        "all_bot_count": 0,
                        "all_labeled_count": 3,
                        "bot_ratio": 0.0,
                        "purity": 1.0,
                        "bot_score": 0.1,
                        "predicted_label_by_train_majority": "human",
                        "label_source": "train_majority",
                        "encoding_depth": 2.0,
                        "train_count": 1,
                        "valid_count": 1,
                        "test_count": 1,
                    },
                    {
                        "community_id": "c0002",
                        "community_size": 2,
                        "train_human_count": 0,
                        "train_bot_count": 1,
                        "train_labeled_count": 1,
                        "all_human_count": 0,
                        "all_bot_count": 2,
                        "all_labeled_count": 2,
                        "bot_ratio": 1.0,
                        "purity": 1.0,
                        "bot_score": 0.9,
                        "predicted_label_by_train_majority": "bot",
                        "label_source": "train_majority",
                        "encoding_depth": 3.0,
                        "train_count": 1,
                        "valid_count": 0,
                        "test_count": 1,
                    },
                ],
            )

            manifest = analyze_community_structure(sample_root, communities_root, graph_root, purity_root, output_root)
            self.assertEqual(manifest["counts"]["communities"], 2)
            rows = list(read_csv_rows(output_root / "community_structure_summary.csv"))
            self.assertEqual(len(rows), 2)
            self.assertIn("archetype", rows[0])
            self.assertTrue((output_root / "representative_communities.csv").exists())


if __name__ == "__main__":
    unittest.main()

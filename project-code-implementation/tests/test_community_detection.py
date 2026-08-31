from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from twibot22_sampler.community_detection import detect_communities
from twibot22_sampler.readers import read_csv_rows, read_manifest, write_csv, write_jsonl


class CommunityDetectionTests(unittest.TestCase):
    def test_detect_communities_structural_entropy_finds_two_clusters(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            sample_root, graph_root, output_root = _build_cluster_fixture(Path(temp_dir))

            manifest = detect_communities(
                sample_root,
                graph_root,
                output_root,
                algorithm="structural_entropy",
                seed=42,
                max_iterations=20,
            )

            self.assertEqual(manifest["algorithm"], "structural_entropy")
            self.assertEqual(manifest["counts"]["users"], 4)
            self.assertEqual(manifest["counts"]["communities"], 2)
            self.assertEqual(manifest["size_summary"]["largest_community"], 2)
            self.assertLessEqual(manifest["final_entropy"], manifest["initial_entropy"])

            assignments = list(read_csv_rows(output_root / "community_assignments.csv"))
            community_by_user = {row["user_id"]: row["community_id"] for row in assignments}
            self.assertEqual(community_by_user["u1"], community_by_user["u2"])
            self.assertEqual(community_by_user["u3"], community_by_user["u4"])
            self.assertNotEqual(community_by_user["u1"], community_by_user["u3"])

            tree = read_manifest(output_root / "encoding_tree.json")
            self.assertTrue(tree["roots"])

    def test_detect_communities_weighted_lpa_finds_two_clusters(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            sample_root, graph_root, output_root = _build_cluster_fixture(Path(temp_dir))

            manifest = detect_communities(
                sample_root,
                graph_root,
                output_root,
                algorithm="weighted_lpa",
                seed=42,
                max_iterations=20,
            )

            self.assertEqual(manifest["algorithm"], "weighted_lpa")
            self.assertEqual(manifest["counts"]["users"], 4)
            self.assertEqual(manifest["counts"]["communities"], 2)

            summary_rows = list(read_csv_rows(output_root / "community_summary.csv"))
            self.assertEqual(len(summary_rows), 2)
            bot_ratios = sorted(float(row["bot_ratio"]) for row in summary_rows)
            self.assertEqual(bot_ratios, [0.0, 1.0])


def _build_cluster_fixture(root: Path) -> tuple[Path, Path, Path]:
    sample_root = root / "sample"
    graph_root = sample_root / "analysis" / "user_graph"
    output_root = sample_root / "analysis" / "communities"
    sample_root.mkdir(parents=True)
    graph_root.mkdir(parents=True)

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
            {"id": "u3", "split": "test"},
            {"id": "u4", "split": "test"},
        ],
    )
    write_csv(
        sample_root / "label.csv",
        ["id", "label"],
        [
            {"id": "u1", "label": "human"},
            {"id": "u2", "label": "human"},
            {"id": "u3", "label": "bot"},
            {"id": "u4", "label": "bot"},
        ],
    )
    write_csv(
        graph_root / "user_knn_edges.csv",
        ["source_user_id", "target_user_id", "weight", "support"],
        [
            {"source_user_id": "u1", "target_user_id": "u2", "weight": 0.95, "support": 2},
            {"source_user_id": "u3", "target_user_id": "u4", "weight": 0.94, "support": 2},
            {"source_user_id": "u2", "target_user_id": "u3", "weight": 0.02, "support": 1},
        ],
    )
    return sample_root, graph_root, output_root


if __name__ == "__main__":
    unittest.main()

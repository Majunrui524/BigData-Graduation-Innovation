from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from twibot22_sampler.community_evaluation import evaluate_communities
from twibot22_sampler.readers import read_csv_rows, write_csv, write_json


class CommunityEvaluationTests(unittest.TestCase):
    def test_evaluate_communities_uses_train_projection(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            sample_root = root / "sample"
            communities_root = sample_root / "analysis" / "communities"
            output_root = sample_root / "analysis" / "community_eval"
            sample_root.mkdir(parents=True)
            communities_root.mkdir(parents=True)

            write_csv(
                sample_root / "split.csv",
                ["id", "split"],
                [
                    {"id": "u1", "split": "train"},
                    {"id": "u2", "split": "train"},
                    {"id": "u3", "split": "test"},
                    {"id": "u4", "split": "test"},
                    {"id": "u5", "split": "valid"},
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
                    {"id": "u5", "label": "human"},
                ],
            )
            write_csv(
                communities_root / "community_assignments.csv",
                ["user_id", "community_id", "community_size", "split", "label"],
                [
                    {"user_id": "u1", "community_id": "c0001", "community_size": 3, "split": "train", "label": "human"},
                    {"user_id": "u3", "community_id": "c0001", "community_size": 3, "split": "test", "label": "bot"},
                    {"user_id": "u5", "community_id": "c0001", "community_size": 3, "split": "valid", "label": "human"},
                    {"user_id": "u2", "community_id": "c0002", "community_size": 2, "split": "train", "label": "bot"},
                    {"user_id": "u4", "community_id": "c0002", "community_size": 2, "split": "test", "label": "human"},
                ],
            )
            write_csv(
                communities_root / "community_summary.csv",
                ["community_id", "community_size"],
                [
                    {"community_id": "c0001", "community_size": 3},
                    {"community_id": "c0002", "community_size": 2},
                ],
            )
            write_json(
                communities_root / "community_manifest.json",
                {
                    "counts": {"users": 5, "communities": 2},
                },
            )

            manifest = evaluate_communities(
                sample_root,
                communities_root,
                output_root,
                threshold=0.5,
                smoothing_alpha=1.0,
            )

            self.assertEqual(manifest["counts"]["users"], 5)
            self.assertEqual(manifest["counts"]["communities"], 2)

            metrics = manifest["metrics"]
            self.assertAlmostEqual(metrics["train"]["accuracy"], 1.0)
            self.assertAlmostEqual(metrics["test"]["accuracy"], 1.0)
            self.assertAlmostEqual(metrics["test"]["f1"], 1.0)

            predictions = list(read_csv_rows(output_root / "community_user_predictions.csv"))
            predicted_by_user = {row["user_id"]: row["predicted_label"] for row in predictions}
            self.assertEqual(predicted_by_user["u3"], "bot")
            self.assertEqual(predicted_by_user["u4"], "human")

            community_scores = list(read_csv_rows(output_root / "community_scores.csv"))
            by_community = {row["community_id"]: row for row in community_scores}
            self.assertEqual(by_community["c0001"]["score_source"], "train_members")
            self.assertEqual(by_community["c0002"]["score_source"], "train_members")


if __name__ == "__main__":
    unittest.main()

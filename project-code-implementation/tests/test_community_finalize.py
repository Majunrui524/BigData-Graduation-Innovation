from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from twibot22_sampler.community_finalize import finalize_best_community_run
from twibot22_sampler.readers import write_csv, write_json


class CommunityFinalizeTests(unittest.TestCase):
    def test_finalize_best_community_run_copies_selected_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            sweep_root = root / "community_sweep"
            run_root = sweep_root / "k10_s0_m1"
            (run_root / "graph").mkdir(parents=True)
            (run_root / "communities").mkdir(parents=True)
            (run_root / "evaluation_t0p2").mkdir(parents=True)

            write_json(
                sweep_root / "community_sweep_manifest.json",
                {
                    "best_run": {
                        "run_name": "k10_s0_m1_t0p2",
                        "k": 10,
                        "min_similarity": 0.0,
                        "min_community_size": 1,
                        "threshold": 0.2,
                        "communities": 2,
                        "largest_community": 3,
                        "test_f1": 0.4,
                        "test_auc": 0.7,
                    }
                },
            )
            write_json(run_root / "graph" / "graph_manifest.json", {"counts": {"users": 4}})
            write_csv(
                run_root / "graph" / "user_knn_edges.csv",
                ["source_user_id", "target_user_id", "weight", "support"],
                [{"source_user_id": "u1", "target_user_id": "u2", "weight": 0.9, "support": 2}],
            )
            write_json(run_root / "communities" / "community_manifest.json", {"counts": {"communities": 2}})
            write_csv(
                run_root / "communities" / "community_assignments.csv",
                ["user_id", "community_id", "community_size", "split", "label"],
                [{"user_id": "u1", "community_id": "c0001", "community_size": 2, "split": "train", "label": "bot"}],
            )
            write_csv(
                run_root / "evaluation_t0p2" / "community_scores.csv",
                [
                    "community_id",
                    "community_size",
                    "train_human_count",
                    "train_bot_count",
                    "train_labeled_count",
                    "all_human_count",
                    "all_bot_count",
                    "all_labeled_count",
                    "bot_score",
                    "predicted_label",
                    "score_source",
                ],
                [
                    {
                        "community_id": "c0001",
                        "community_size": 2,
                        "train_human_count": 0,
                        "train_bot_count": 1,
                        "train_labeled_count": 1,
                        "all_human_count": 0,
                        "all_bot_count": 2,
                        "all_labeled_count": 2,
                        "bot_score": 0.8,
                        "predicted_label": "bot",
                        "score_source": "train_members",
                    },
                    {
                        "community_id": "c0002",
                        "community_size": 2,
                        "train_human_count": 1,
                        "train_bot_count": 0,
                        "train_labeled_count": 1,
                        "all_human_count": 2,
                        "all_bot_count": 0,
                        "all_labeled_count": 2,
                        "bot_score": 0.1,
                        "predicted_label": "human",
                        "score_source": "train_members",
                    },
                ],
            )
            write_json(run_root / "evaluation_t0p2" / "community_eval_manifest.json", {"metrics": {"test": {"f1": 0.4}}})
            write_csv(
                run_root / "evaluation_t0p2" / "community_user_predictions.csv",
                ["user_id", "split", "label", "community_id", "community_size", "bot_score", "predicted_label", "score_source"],
                [{"user_id": "u1", "split": "train", "label": "bot", "community_id": "c0001", "community_size": 2, "bot_score": 0.8, "predicted_label": "bot", "score_source": "train_members"}],
            )

            output_root = root / "community_best"
            manifest = finalize_best_community_run(sweep_root, output_root, top_communities=1)

            self.assertEqual(manifest["selected_run"]["run_name"], "k10_s0_m1_t0p2")
            self.assertTrue((output_root / "graph" / "graph_manifest.json").exists())
            self.assertTrue((output_root / "communities" / "community_assignments.csv").exists())
            self.assertTrue((output_root / "evaluation" / "community_eval_manifest.json").exists())
            self.assertTrue((output_root / "top_bot_communities.csv").exists())
            self.assertTrue((output_root / "best_run_summary.md").exists())


if __name__ == "__main__":
    unittest.main()

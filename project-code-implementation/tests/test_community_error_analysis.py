from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from twibot22_sampler.community_error_analysis import analyze_community_errors
from twibot22_sampler.readers import read_csv_rows, write_csv, write_json, write_jsonl


class CommunityErrorAnalysisTests(unittest.TestCase):
    def test_analyze_community_errors_exports_fp_fn_and_community_tables(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            sample_root = root / "sample"
            best_root = sample_root / "analysis" / "community_best"
            output_root = sample_root / "analysis" / "community_error_analysis"
            (sample_root / "analysis" / "user_features").mkdir(parents=True)
            (best_root / "evaluation").mkdir(parents=True)

            write_json(
                best_root / "best_run_manifest.json",
                {
                    "selected_run": {
                        "threshold": 0.2,
                    }
                },
            )
            write_jsonl(
                sample_root / "analysis" / "user_features" / "user_feature_table.jsonl",
                [
                    {
                        "user_id": "u1",
                        "username": "alpha",
                        "name": "Alpha",
                        "description": "alpha desc",
                        "followers_count": 10,
                        "following_count": 2,
                        "tweets_total": 5,
                        "verified": 0,
                        "can_triplet": 1,
                        "can_post_type": 1,
                        "can_time_feature": 1,
                        "can_network_feature": 1,
                        "can_full_pipeline": 1,
                        "triplet_document_present": 1,
                        "triplet_tweet_count": 5,
                        "post_type_tweet_count": 5,
                        "triplet_incomplete_flag": 0,
                        "post_type_incomplete_flag": 0,
                    },
                    {
                        "user_id": "u2",
                        "username": "beta",
                        "name": "Beta",
                        "description": "beta desc",
                        "followers_count": 20,
                        "following_count": 3,
                        "tweets_total": 8,
                        "verified": 0,
                        "can_triplet": 1,
                        "can_post_type": 1,
                        "can_time_feature": 1,
                        "can_network_feature": 0,
                        "can_full_pipeline": 0,
                        "triplet_document_present": 1,
                        "triplet_tweet_count": 8,
                        "post_type_tweet_count": 8,
                        "triplet_incomplete_flag": 0,
                        "post_type_incomplete_flag": 0,
                    },
                    {
                        "user_id": "u3",
                        "username": "gamma",
                        "name": "Gamma",
                        "description": "gamma desc",
                        "followers_count": 5,
                        "following_count": 1,
                        "tweets_total": 3,
                        "verified": 1,
                        "can_triplet": 0,
                        "can_post_type": 1,
                        "can_time_feature": 1,
                        "can_network_feature": 0,
                        "can_full_pipeline": 0,
                        "triplet_document_present": 0,
                        "triplet_tweet_count": 0,
                        "post_type_tweet_count": 3,
                        "triplet_incomplete_flag": 0,
                        "post_type_incomplete_flag": 0,
                    },
                ],
            )
            write_csv(
                best_root / "evaluation" / "community_user_predictions.csv",
                ["user_id", "split", "label", "community_id", "community_size", "bot_score", "predicted_label", "score_source"],
                [
                    {"user_id": "u1", "split": "test", "label": "human", "community_id": "c1", "community_size": 2, "bot_score": 0.3, "predicted_label": "bot", "score_source": "train_members"},
                    {"user_id": "u2", "split": "test", "label": "bot", "community_id": "c2", "community_size": 1, "bot_score": 0.1, "predicted_label": "human", "score_source": "train_members"},
                    {"user_id": "u3", "split": "train", "label": "human", "community_id": "c1", "community_size": 2, "bot_score": 0.3, "predicted_label": "bot", "score_source": "train_members"},
                ],
            )
            write_csv(
                best_root / "evaluation" / "community_scores.csv",
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
                    {"community_id": "c1", "community_size": 2, "train_human_count": 1, "train_bot_count": 0, "train_labeled_count": 1, "all_human_count": 2, "all_bot_count": 0, "all_labeled_count": 2, "bot_score": 0.3, "predicted_label": "bot", "score_source": "train_members"},
                    {"community_id": "c2", "community_size": 1, "train_human_count": 0, "train_bot_count": 1, "train_labeled_count": 1, "all_human_count": 0, "all_bot_count": 1, "all_labeled_count": 1, "bot_score": 0.1, "predicted_label": "human", "score_source": "train_members"},
                ],
            )

            manifest = analyze_community_errors(
                sample_root,
                best_root,
                output_root,
                focus_split="test",
                top_k=10,
            )

            self.assertEqual(manifest["counts"]["false_positives_focus"], 1)
            self.assertEqual(manifest["counts"]["false_negatives_focus"], 1)
            self.assertTrue((output_root / "test_false_positives.csv").exists())
            self.assertTrue((output_root / "test_false_negatives.csv").exists())
            self.assertTrue((output_root / "community_error_summary.csv").exists())

            fp_rows = list(read_csv_rows(output_root / "test_false_positives.csv"))
            fn_rows = list(read_csv_rows(output_root / "test_false_negatives.csv"))
            self.assertEqual(fp_rows[0]["user_id"], "u1")
            self.assertEqual(fn_rows[0]["user_id"], "u2")


if __name__ == "__main__":
    unittest.main()

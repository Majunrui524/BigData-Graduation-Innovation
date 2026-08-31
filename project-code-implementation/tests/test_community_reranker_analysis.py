from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from twibot22_sampler.community_reranker_analysis import analyze_community_reranker
from twibot22_sampler.readers import read_csv_rows, write_csv, write_jsonl


class CommunityRerankerAnalysisTests(unittest.TestCase):
    def test_analyze_community_reranker_exports_fixed_and_regressed_cases(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            sample_root = root / "sample"
            best_root = sample_root / "analysis" / "community_best"
            reranker_root = sample_root / "analysis" / "community_reranker"
            output_root = sample_root / "analysis" / "community_reranker_analysis"
            (sample_root / "analysis" / "user_features").mkdir(parents=True)
            (best_root / "evaluation").mkdir(parents=True)
            reranker_root.mkdir(parents=True)

            write_jsonl(
                sample_root / "analysis" / "user_features" / "user_feature_table.jsonl",
                [
                    {"user_id": "u1", "username": "u1name", "followers_count": 10, "following_count": 2, "tweets_total": 3},
                    {"user_id": "u2", "username": "u2name", "followers_count": 20, "following_count": 3, "tweets_total": 4},
                    {"user_id": "u3", "username": "u3name", "followers_count": 30, "following_count": 4, "tweets_total": 5},
                ],
            )
            write_csv(
                best_root / "evaluation" / "community_user_predictions.csv",
                ["user_id", "split", "label", "community_id", "community_size", "bot_score", "predicted_label", "score_source"],
                [
                    {"user_id": "u1", "split": "test", "label": "bot", "community_id": "c1", "community_size": 10, "bot_score": 0.1, "predicted_label": "human", "score_source": "train_members"},
                    {"user_id": "u2", "split": "test", "label": "human", "community_id": "c1", "community_size": 10, "bot_score": 0.1, "predicted_label": "human", "score_source": "train_members"},
                    {"user_id": "u3", "split": "test", "label": "human", "community_id": "c2", "community_size": 8, "bot_score": 0.6, "predicted_label": "bot", "score_source": "train_members"},
                ],
            )
            write_csv(
                reranker_root / "reranker_predictions.csv",
                ["user_id", "split", "label", "community_id", "community_size", "community_bot_score", "baseline_predicted_label", "reranker_bot_score", "bot_score", "predicted_label", "selected_threshold"],
                [
                    {"user_id": "u1", "split": "test", "label": "bot", "community_id": "c1", "community_size": 10, "community_bot_score": 0.1, "baseline_predicted_label": "human", "reranker_bot_score": 0.7, "bot_score": 0.7, "predicted_label": "bot", "selected_threshold": 0.25},
                    {"user_id": "u2", "split": "test", "label": "human", "community_id": "c1", "community_size": 10, "community_bot_score": 0.1, "baseline_predicted_label": "human", "reranker_bot_score": 0.8, "bot_score": 0.8, "predicted_label": "bot", "selected_threshold": 0.25},
                    {"user_id": "u3", "split": "test", "label": "human", "community_id": "c2", "community_size": 8, "community_bot_score": 0.6, "baseline_predicted_label": "bot", "reranker_bot_score": 0.9, "bot_score": 0.9, "predicted_label": "bot", "selected_threshold": 0.25},
                ],
            )

            manifest = analyze_community_reranker(
                sample_root,
                best_root,
                reranker_root,
                output_root,
                focus_split="test",
                top_k=10,
            )

            self.assertEqual(manifest["counts"]["changed_predictions"], 2)
            self.assertEqual(manifest["counts"]["fixed_cases"], 1)
            self.assertEqual(manifest["counts"]["regressed_cases"], 1)

            fixed_rows = list(read_csv_rows(output_root / "test_fixed_cases.csv"))
            regressed_rows = list(read_csv_rows(output_root / "test_regressed_cases.csv"))
            self.assertEqual(fixed_rows[0]["user_id"], "u1")
            self.assertEqual(regressed_rows[0]["user_id"], "u2")


if __name__ == "__main__":
    unittest.main()

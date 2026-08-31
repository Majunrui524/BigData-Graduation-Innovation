from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from twibot22_sampler.community_reranker import train_community_reranker
from twibot22_sampler.readers import read_csv_rows, write_csv, write_json, write_jsonl


class CommunityRerankerTests(unittest.TestCase):
    def test_train_community_reranker_improves_on_ambiguous_community_scores(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            sample_root = root / "sample"
            best_root = sample_root / "analysis" / "community_best"
            output_root = sample_root / "analysis" / "community_reranker"
            (sample_root / "analysis" / "user_features").mkdir(parents=True)
            (best_root / "evaluation").mkdir(parents=True)

            write_jsonl(
                sample_root / "analysis" / "user_features" / "user_feature_table.jsonl",
                [
                    _feature_row("u1", followers=1000),
                    _feature_row("u2", followers=10),
                    _feature_row("u3", followers=900),
                    _feature_row("u4", followers=15),
                    _feature_row("u5", followers=850),
                    _feature_row("u6", followers=20),
                    _feature_row("u7", followers=800),
                    _feature_row("u8", followers=12),
                ],
            )
            write_csv(
                best_root / "evaluation" / "community_user_predictions.csv",
                ["user_id", "split", "label", "community_id", "community_size", "bot_score", "predicted_label", "score_source"],
                [
                    {"user_id": "u1", "split": "train", "label": "bot", "community_id": "c1", "community_size": 10, "bot_score": 0.2, "predicted_label": "bot", "score_source": "train_members"},
                    {"user_id": "u2", "split": "train", "label": "human", "community_id": "c2", "community_size": 10, "bot_score": 0.2, "predicted_label": "bot", "score_source": "train_members"},
                    {"user_id": "u3", "split": "train", "label": "bot", "community_id": "c3", "community_size": 12, "bot_score": 0.7, "predicted_label": "bot", "score_source": "train_members"},
                    {"user_id": "u4", "split": "train", "label": "human", "community_id": "c4", "community_size": 12, "bot_score": 0.1, "predicted_label": "human", "score_source": "train_members"},
                    {"user_id": "u5", "split": "valid", "label": "bot", "community_id": "c5", "community_size": 9, "bot_score": 0.2, "predicted_label": "bot", "score_source": "train_members"},
                    {"user_id": "u6", "split": "valid", "label": "human", "community_id": "c6", "community_size": 9, "bot_score": 0.2, "predicted_label": "bot", "score_source": "train_members"},
                    {"user_id": "u7", "split": "test", "label": "bot", "community_id": "c7", "community_size": 8, "bot_score": 0.2, "predicted_label": "bot", "score_source": "train_members"},
                    {"user_id": "u8", "split": "test", "label": "human", "community_id": "c8", "community_size": 8, "bot_score": 0.2, "predicted_label": "bot", "score_source": "train_members"},
                ],
            )
            write_json(
                best_root / "evaluation" / "community_eval_manifest.json",
                {
                    "metrics": {
                        "test": {"f1": 0.66666667, "auc": 0.5},
                        "valid": {"f1": 0.66666667, "auc": 0.5},
                    }
                },
            )

            manifest = train_community_reranker(
                sample_root,
                best_root,
                output_root,
                learning_rate=0.1,
                max_epochs=200,
                l2=0.0,
                threshold_values=[0.3, 0.4, 0.5, 0.6],
                early_stopping_rounds=20,
            )

            self.assertGreater(manifest["reranker_metrics"]["test"]["f1"], manifest["baseline_metrics"]["test"]["f1"])
            self.assertTrue((output_root / "reranker_predictions.csv").exists())
            self.assertTrue((output_root / "reranker_weights.csv").exists())

            predictions = list(read_csv_rows(output_root / "reranker_predictions.csv"))
            by_user = {row["user_id"]: row for row in predictions}
            self.assertEqual(by_user["u7"]["predicted_label"], "bot")
            self.assertEqual(by_user["u8"]["predicted_label"], "human")


def _feature_row(user_id: str, *, followers: int) -> dict[str, object]:
    return {
        "user_id": user_id,
        "followers_count": followers,
        "following_count": 10,
        "tweets_total": 8,
        "triplet_tweet_count": 8,
        "post_type_tweet_count": 8,
        "following_in_degree": 1,
        "following_out_degree": 1,
        "verified": 0,
        "can_triplet": 1,
        "can_post_type": 1,
        "can_time_feature": 1,
        "can_network_feature": 1,
        "can_full_pipeline": 1,
        "triplet_document_present": 1,
        "triplet_incomplete_flag": 0,
        "post_type_incomplete_flag": 0,
        "tweets_with_created_at": 8,
        "tweets_with_public_metrics": 8,
        "tweets_with_references": 1,
        "tweets_with_external_url": 0,
        "post_type_coarse_ratio_original": 0.2,
        "post_type_coarse_ratio_retweet": 0.3,
        "post_type_coarse_ratio_comment_reply": 0.3,
        "post_type_coarse_ratio_link_share": 0.2,
    }


if __name__ == "__main__":
    unittest.main()

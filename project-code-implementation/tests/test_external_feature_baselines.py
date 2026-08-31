from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from twibot22_sampler.external_feature_baselines import run_feature_baselines
from twibot22_sampler.readers import read_csv_rows, write_csv, write_jsonl


class ExternalFeatureBaselineTests(unittest.TestCase):
    def test_run_feature_baselines_writes_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            sample_root = root / "sample"
            feature_root = sample_root / "analysis" / "user_features"
            output_root = sample_root / "analysis" / "external_baselines_10k"
            feature_root.mkdir(parents=True)

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
            write_jsonl(
                feature_root / "user_feature_table.jsonl",
                [
                    _feature_row("u1", 10, 2, 0.9),
                    _feature_row("u2", 2, 10, 0.1),
                    _feature_row("u3", 9, 2, 0.8),
                    _feature_row("u4", 3, 9, 0.2),
                    _feature_row("u5", 8, 3, 0.75),
                    _feature_row("u6", 2, 8, 0.15),
                ],
            )

            manifests = run_feature_baselines(sample_root, output_root)
            self.assertIn("logreg", manifests)
            self.assertIn("random_forest", manifests)
            self.assertTrue((output_root / "logreg" / "metrics.json").exists())
            self.assertTrue((output_root / "random_forest" / "metrics.json").exists())

            rows = list(read_csv_rows(output_root / "logreg" / "predictions.csv"))
            self.assertEqual(len(rows), 6)
            self.assertEqual({row["split"] for row in rows}, {"train", "valid", "test"})


def _feature_row(user_id: str, followers: int, following: int, bot_ratio: float) -> dict[str, object]:
    return {
        "user_id": user_id,
        "username": user_id,
        "label": "",
        "split": "",
        "followers_count": followers,
        "following_count": following,
        "tweets_total": followers + following,
        "verified": 1 if bot_ratio < 0.3 else 0,
        "can_triplet": 1,
        "can_post_type": 1,
        "can_time_feature": 1,
        "can_network_feature": 1,
        "can_full_pipeline": 1,
        "triplet_document_present": 1,
        "triplet_incomplete_flag": 0,
        "post_type_incomplete_flag": 0,
        "triplet_count": 5,
        "triplet_tweet_count": 5,
        "post_type_tweet_count": 5,
        "following_in_degree": following,
        "following_out_degree": followers,
        "post_type_coarse_ratio_original": 1.0 - bot_ratio,
        "post_type_coarse_ratio_retweet": bot_ratio,
        "post_type_coarse_ratio_comment_reply": 0.0,
        "post_type_coarse_ratio_link_share": 0.0,
        "tweets_with_created_at": 5,
        "tweets_with_public_metrics": 5,
        "tweets_with_references": 5,
        "tweets_with_external_url": int(round(bot_ratio * 5)),
    }


if __name__ == "__main__":
    unittest.main()

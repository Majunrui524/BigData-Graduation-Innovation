from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from twibot22_sampler.grouping_baseline_summary import summarize_grouping_baselines
from twibot22_sampler.grouping_baselines import run_kmeans_grouping_baseline
from twibot22_sampler.readers import write_csv, write_json, write_jsonl


class GroupingBaselineTests(unittest.TestCase):
    def test_run_kmeans_grouping_baseline_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            sample_root = root / "sample"
            feature_root = sample_root / "analysis" / "user_features"
            kmeans_root = sample_root / "analysis" / "grouping_baselines_10k" / "kmeans"
            weighted_root = sample_root / "analysis" / "grouping_baselines_10k" / "weighted_lpa" / "community_purity"
            structural_root = sample_root / "analysis" / "run_10k_late" / "community_purity"
            summary_root = sample_root / "analysis" / "grouping_baselines_10k" / "summary"
            feature_root.mkdir(parents=True)
            weighted_root.mkdir(parents=True)
            structural_root.mkdir(parents=True)

            write_csv(
                sample_root / "split.csv",
                ["id", "split"],
                [
                    {"id": "u1", "split": "train"},
                    {"id": "u2", "split": "train"},
                    {"id": "u3", "split": "train"},
                    {"id": "u4", "split": "valid"},
                    {"id": "u5", "split": "valid"},
                    {"id": "u6", "split": "test"},
                    {"id": "u7", "split": "test"},
                    {"id": "u8", "split": "test"},
                ],
            )
            write_csv(
                sample_root / "label.csv",
                ["id", "label"],
                [
                    {"id": "u1", "label": "bot"},
                    {"id": "u2", "label": "bot"},
                    {"id": "u3", "label": "human"},
                    {"id": "u4", "label": "bot"},
                    {"id": "u5", "label": "human"},
                    {"id": "u6", "label": "bot"},
                    {"id": "u7", "label": "human"},
                    {"id": "u8", "label": "human"},
                ],
            )
            write_jsonl(
                feature_root / "user_feature_table.jsonl",
                [
                    _feature_row("u1", 9, 2, 1.0),
                    _feature_row("u2", 8, 3, 1.0),
                    _feature_row("u3", 2, 9, 0.0),
                    _feature_row("u4", 7, 2, 1.0),
                    _feature_row("u5", 3, 8, 0.0),
                    _feature_row("u6", 8, 2, 1.0),
                    _feature_row("u7", 2, 8, 0.0),
                    _feature_row("u8", 3, 9, 0.0),
                ],
            )

            manifest = run_kmeans_grouping_baseline(sample_root, kmeans_root, k_values=(2, 3, 4))
            self.assertIn("selected_params", manifest)
            self.assertTrue((kmeans_root / "community_purity_manifest.json").exists())

            _write_minimal_purity_manifest(weighted_root / "community_purity_manifest.json", "weighted_lpa", "Weighted LPA")
            _write_minimal_purity_manifest(
                structural_root / "community_purity_manifest.json",
                "structural_entropy",
                "Structural Entropy",
            )

            summary_manifest = summarize_grouping_baselines(
                sample_root,
                summary_root,
                kmeans_root=kmeans_root,
                weighted_lpa_purity_root=weighted_root,
                structural_entropy_purity_root=structural_root,
            )
            self.assertEqual(summary_manifest["counts"]["methods"], 3)
            self.assertTrue((summary_root / "grouping_baseline_results.csv").exists())


def _feature_row(user_id: str, followers: int, following: int, bot_flag: float) -> dict[str, object]:
    return {
        "user_id": user_id,
        "username": user_id,
        "label": "",
        "split": "",
        "followers_count": followers,
        "following_count": following,
        "tweets_total": followers + following,
        "verified": 0,
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
        "post_type_coarse_ratio_original": 1.0 - bot_flag,
        "post_type_coarse_ratio_retweet": bot_flag,
        "post_type_coarse_ratio_comment_reply": 0.0,
        "post_type_coarse_ratio_link_share": 0.0,
        "tweets_with_created_at": 5,
        "tweets_with_public_metrics": 5,
        "tweets_with_references": 5,
        "tweets_with_external_url": int(bot_flag * 5),
    }


def _write_minimal_purity_manifest(path: Path, method_key: str, method_name: str) -> None:
    write_json(
        path,
        {
            "method_key": method_key,
            "method_name": method_name,
            "global_purity": 0.8,
            "selected_params": {"threshold": 0.5},
            "counts": {"communities": 4},
            "metrics": {
                "test": {
                    "accuracy": 0.6,
                    "precision": 0.5,
                    "recall": 0.4,
                    "f1": 0.44,
                    "auc": 0.55,
                }
            },
        },
    )


if __name__ == "__main__":
    unittest.main()

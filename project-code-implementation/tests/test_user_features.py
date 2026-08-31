from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from twibot22_sampler.user_features import build_user_feature_table


class UserFeatureTableTests(unittest.TestCase):
    def test_build_user_feature_table_merges_sources_and_tracks_triplet_gaps(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            sample_root = root / "sample"
            output_root = sample_root / "analysis" / "user_features"
            (sample_root / "analysis" / "field_audit").mkdir(parents=True)
            (sample_root / "derived" / "post_types").mkdir(parents=True)
            (sample_root / "derived" / "triplets").mkdir(parents=True)

            (sample_root / "user.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "id": "u1",
                                "username": "alpha",
                                "name": "Alpha",
                                "description": "desc",
                                "url": "https://example.com",
                                "verified": False,
                                "public_metrics": {"followers_count": 10, "following_count": 2, "tweet_count": 2},
                            }
                        ),
                        json.dumps(
                            {
                                "id": "u2",
                                "username": "beta",
                                "name": "Beta",
                                "description": "",
                                "url": "",
                                "verified": True,
                                "public_metrics": {"followers_count": 4, "following_count": 1, "tweet_count": 2},
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (sample_root / "tweet_0.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"id": "t1", "author_id": 1, "text": "first", "created_at": "2024-01-01T00:00:00Z"}),
                        json.dumps({"id": "t2", "author_id": 1, "text": "second", "created_at": "2024-01-02T00:00:00Z"}),
                        json.dumps({"id": "t3", "author_id": 2, "text": "third", "created_at": "2024-01-03T00:00:00Z"}),
                        json.dumps({"id": "t4", "author_id": 2, "text": "fourth", "created_at": "2024-01-04T00:00:00Z"}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (sample_root / "split.csv").write_text("id,split\nu1,train\nu2,test\n", encoding="utf-8")
            (sample_root / "label.csv").write_text("id,label\nu1,human\nu2,bot\n", encoding="utf-8")
            (sample_root / "analysis" / "field_audit" / "user_feature_availability.csv").write_text(
                "\n".join(
                    [
                        "user_id,split,label,user_created_at_present,description_present,profile_url_present,followers_count_present,following_count_present,listed_count_present,user_tweet_count_present,verified_present,verified_true,tweets_total,tweets_with_text,tweets_with_created_at,tweets_with_public_metrics,tweets_with_like_count,tweets_with_reply_count,tweets_with_retweet_count,tweets_with_quote_count,tweets_with_references,tweets_with_external_url,tweets_with_lang,tweets_with_source,following_out_degree,following_in_degree,post_edge_count,can_triplet,can_post_type,can_time_feature,can_behavior_feature,can_network_feature,can_full_pipeline",
                        "u1,train,human,1,1,1,1,1,1,1,1,0,2,2,2,2,2,2,2,2,0,0,2,2,1,0,2,1,1,1,1,1,1",
                        "u2,test,bot,1,0,0,1,1,1,1,1,1,2,2,2,2,2,2,2,2,0,0,2,2,0,1,2,1,1,1,1,1,1",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (sample_root / "derived" / "post_types" / "run_manifest.json").write_text(
                json.dumps(
                    {
                        "selection": {
                            "per_user_limit": 2,
                            "min_user_tweets": 1,
                            "max_users": None,
                            "max_tweets": None,
                        }
                    }
                ),
                encoding="utf-8",
            )
            (sample_root / "derived" / "post_types" / "tweet_post_types.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"tweet_id": "t1", "author_id": "u1"}),
                        json.dumps({"tweet_id": "t2", "author_id": "u1"}),
                        json.dumps({"tweet_id": "t3", "author_id": "u2"}),
                        json.dumps({"tweet_id": "t4", "author_id": "u2"}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (sample_root / "derived" / "post_types" / "tweet_post_type_errors.jsonl").write_text("", encoding="utf-8")
            (sample_root / "derived" / "post_types" / "user_post_type_distribution.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "author_id": "u1",
                                "tweet_count": 2,
                                "coarse_counts": {"original": 1, "retweet": 1},
                                "detail_counts": {"original": 1, "retweet": 1},
                            }
                        ),
                        json.dumps(
                            {
                                "author_id": "u2",
                                "tweet_count": 2,
                                "coarse_counts": {"comment_reply": 1, "link_share": 1},
                                "detail_counts": {"reply": 1, "link_share": 1},
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (sample_root / "derived" / "triplets" / "run_manifest.json").write_text(
                json.dumps(
                    {
                        "selection": {
                            "per_user_limit": 2,
                            "min_user_tweets": 2,
                            "max_users": None,
                            "max_tweets": None,
                        }
                    }
                ),
                encoding="utf-8",
            )
            (sample_root / "derived" / "triplets" / "tweet_triplets.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"tweet_id": "t1", "author_id": "u1"}),
                        json.dumps({"tweet_id": "t3", "author_id": "u2"}),
                        json.dumps({"tweet_id": "t4", "author_id": "u2"}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (sample_root / "derived" / "triplets" / "tweet_triplet_errors.jsonl").write_text(
                json.dumps({"tweet_id": "t2", "author_id": "u1", "error": "bad json"}) + "\n",
                encoding="utf-8",
            )
            (sample_root / "derived" / "triplets" / "user_triplet_documents.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "author_id": "u1",
                                "tweet_count": 1,
                                "triplet_count": 2,
                                "triplet_document": "doc-1",
                            }
                        ),
                        json.dumps(
                            {
                                "author_id": "u2",
                                "tweet_count": 2,
                                "triplet_count": 3,
                                "triplet_document": "doc-2",
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            manifest = build_user_feature_table(sample_root, output_root, triplet_seed=42, post_type_seed=42)

            self.assertEqual(manifest["counts"]["users"], 2)
            self.assertEqual(manifest["counts"]["triplet_incomplete_users"], 1)
            self.assertEqual(manifest["totals"]["triplet_unresolved_errors"], 1)

            with (output_root / "user_feature_table.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            row_by_id = {row["user_id"]: row for row in rows}
            self.assertEqual(row_by_id["u1"]["triplet_incomplete_flag"], "1")
            self.assertEqual(row_by_id["u1"]["triplet_unresolved_error_count"], "1")
            self.assertEqual(row_by_id["u1"]["triplet_expected_tweet_count"], "2")
            self.assertEqual(row_by_id["u1"]["triplet_tweet_count"], "1")
            self.assertEqual(row_by_id["u2"]["post_type_incomplete_flag"], "0")
            self.assertTrue((output_root / "feature_table_summary.md").exists())


if __name__ == "__main__":
    unittest.main()

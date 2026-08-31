from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from twibot22_sampler.audit import run_field_audit


class AuditTests(unittest.TestCase):
    def test_run_field_audit_produces_summary_and_csv(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            sample_root = root / "sample"
            output_root = root / "audit"
            sample_root.mkdir(parents=True)

            (sample_root / "user.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "id": "u1",
                                "created_at": "2024-01-01T00:00:00Z",
                                "description": "bot account",
                                "url": "https://example.com",
                                "verified": False,
                                "public_metrics": {
                                    "followers_count": 10,
                                    "following_count": 2,
                                    "listed_count": 1,
                                    "tweet_count": 9,
                                },
                            }
                        ),
                        json.dumps(
                            {
                                "id": "u2",
                                "created_at": "2024-01-02T00:00:00Z",
                                "description": "",
                                "verified": True,
                                "public_metrics": {
                                    "followers_count": 5,
                                    "following_count": 1,
                                    "listed_count": 0,
                                    "tweet_count": 1,
                                },
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
                        json.dumps(
                            {
                                "id": "t1",
                                "author_id": "u1",
                                "text": "hello https://example.com",
                                "created_at": "2024-01-01T00:01:00Z",
                                "lang": "en",
                                "source": "web",
                                "entities": {"urls": [{"expanded_url": "https://example.com"}]},
                                "public_metrics": {
                                    "like_count": 1,
                                    "reply_count": 2,
                                    "retweet_count": 3,
                                    "quote_count": 0,
                                },
                            }
                        ),
                        json.dumps(
                            {
                                "id": "t2",
                                "author_id": "u1",
                                "text": "reply",
                                "created_at": "2024-01-01T00:02:00Z",
                                "lang": "en",
                                "in_reply_to_user_id": "u2",
                                "referenced_tweets": [{"id": "t9", "type": "replied_to"}],
                                "public_metrics": {
                                    "like_count": 0,
                                    "reply_count": 1,
                                    "retweet_count": 0,
                                    "quote_count": 0,
                                },
                            }
                        ),
                        json.dumps(
                            {
                                "id": "t3",
                                "author_id": "u2",
                                "text": "plain",
                                "created_at": "2024-01-02T00:02:00Z",
                                "lang": "en",
                                "public_metrics": {
                                    "like_count": 0,
                                    "reply_count": 0,
                                    "retweet_count": 0,
                                    "quote_count": 0,
                                },
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (sample_root / "edge.csv").write_text(
                "\n".join(
                    [
                        "source_id,target_id,relation",
                        "u1,u2,following",
                        "u1,t1,post",
                        "u1,t2,post",
                        "u2,t3,post",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (sample_root / "split.csv").write_text("id,split\nu1,train\nu2,test\n", encoding="utf-8")
            (sample_root / "label.csv").write_text("id,label\nu1,bot\nu2,human\n", encoding="utf-8")

            summary = run_field_audit(sample_root, output_root, min_triplet_tweets=2, min_time_tweets=1)

            self.assertEqual(summary["overall"]["users"], 2)
            self.assertEqual(summary["availability_counts"]["triplet_ready_users"], 1)
            self.assertEqual(summary["availability_counts"]["post_type_ready_users"], 2)
            self.assertTrue((output_root / "user_feature_availability.csv").exists())
            self.assertTrue((output_root / "audit_summary.md").exists())

    def test_run_field_audit_normalizes_numeric_tweet_author_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            sample_root = root / "sample"
            output_root = root / "audit"
            sample_root.mkdir(parents=True)

            (sample_root / "user.jsonl").write_text(
                json.dumps({"id": "u1", "public_metrics": {"followers_count": 1, "following_count": 1, "tweet_count": 2}})
                + "\n",
                encoding="utf-8",
            )
            (sample_root / "tweet_0.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"id": "t1", "author_id": 1, "text": "a", "created_at": "2024-01-01T00:00:00Z"}),
                        json.dumps({"id": "t2", "author_id": "1", "text": "b", "created_at": "2024-01-01T00:01:00Z"}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (sample_root / "edge.csv").write_text(
                "source_id,target_id,relation\nu1,u2,following\nu1,t1,post\nu1,t2,post\n",
                encoding="utf-8",
            )
            (sample_root / "split.csv").write_text("id,split\nu1,train\n", encoding="utf-8")
            (sample_root / "label.csv").write_text("id,label\nu1,human\n", encoding="utf-8")

            summary = run_field_audit(sample_root, output_root, min_triplet_tweets=2, min_time_tweets=2)

            self.assertEqual(summary["overall"]["users"], 1)
            self.assertEqual(summary["overall"]["tweets"], 2)
            self.assertEqual(summary["label_distribution"]["human"], 1)


if __name__ == "__main__":
    unittest.main()

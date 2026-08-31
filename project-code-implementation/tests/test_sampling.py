from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from twibot22_sampler.config import SamplingThresholds
from twibot22_sampler.context_sampling import expand_context_users
from twibot22_sampler.tweet_sampling import (
    collect_post_candidates_and_user_edges,
    collect_third_pass_edges,
    expand_referenced_tweets,
    finalize_post_selection,
    load_tweet_records,
)
from twibot22_sampler.user_sampling import select_seed_users


class SamplingTests(unittest.TestCase):
    def test_seed_sampling_is_deterministic(self) -> None:
        rows = [
            {
                "user_id": f"u{index}",
                "split": "train" if index < 4 else "test",
                "label": "bot" if index % 2 else "human",
                "verified_bucket": "false",
                "followers_count": 10 + index,
                "following_count": 5 + index,
                "tweet_count_hint": 9 if index < 6 else 3,
                "degree_proxy": 15 + (index * 2),
                "primary_pool": index < 6,
                "sparse_pool": index >= 6,
            }
            for index in range(8)
        ]
        with patch.dict("twibot22_sampler.config.PRESET_SIZES", {"smoke": 4, "main": 8}, clear=True):
            first_ids, first_summary = select_seed_users(rows, preset="smoke", seed=7)
            second_ids, second_summary = select_seed_users(rows, preset="smoke", seed=7)

        self.assertEqual(first_ids, second_ids)
        self.assertEqual(len(first_ids), 4)
        self.assertEqual(first_summary["split_quotas"], {"test": 2, "train": 2})
        self.assertEqual(second_summary["selected_seed_count"], 4)

    def test_context_sampling_keeps_mutual_and_drops_single_seed_hub(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            edge_path = Path(temp_dir) / "edge.csv"
            edge_path.write_text(
                "\n".join(
                    [
                        "source_id,target_id,relation",
                        "s1,c1,following",
                        "c1,s1,following",
                        "c2,s1,following",
                        "s1,c3,following",
                        "hub,s1,following",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            final_users, summary = expand_context_users(
                edge_path,
                seed_user_ids=["s1"],
                profile_by_user_id={
                    "c1": {"degree_proxy": 10},
                    "c2": {"degree_proxy": 11},
                    "c3": {"degree_proxy": 12},
                    "hub": {"degree_proxy": 1000},
                },
                thresholds=SamplingThresholds(max_context_mutual=2, max_context_follower=2, max_context_following=2),
                seed=42,
            )

            self.assertIn("s1", final_users)
            self.assertIn("c1", final_users)
            self.assertIn("c2", final_users)
            self.assertIn("c3", final_users)
            self.assertNotIn("hub", final_users)
            self.assertEqual(summary["category_totals"]["mutual"], 1)

    def test_tweet_sampling_respects_limits_and_reference_closure(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            edge_path = root / "edge.csv"
            edge_path.write_text(
                "\n".join(
                    [
                        "source_id,target_id,relation",
                        "u1,u2,following",
                        "u1,t1,post",
                        "u1,t2,post",
                        "u1,t3,post",
                        "t1,t99,reply",
                        "t2,u2,mention",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            tweet_path = root / "tweet_0.jsonl"
            tweet_path.write_text(
                "\n".join(
                    [
                        '{"id":"t1","created_at":"2024-01-01T00:00:00Z","referenced_tweets":[{"id":"t99"}],"text":"a"}',
                        '{"id":"t2","created_at":"2024-01-02T00:00:00Z","text":"b"}',
                        '{"id":"t3","created_at":"2024-01-03T00:00:00Z","text":"c"}',
                        '{"id":"t99","created_at":"2023-12-31T00:00:00Z","text":"parent"}',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            thresholds = SamplingThresholds(seed_user_max_tweets=2, context_user_max_tweets=1)
            post_candidates, user_edges, _summary = collect_post_candidates_and_user_edges(
                edge_path,
                final_user_ids={"u1", "u2"},
                seed_user_ids={"u1"},
                thresholds=thresholds,
                seed=5,
            )
            tweet_records = load_tweet_records([tweet_path], {tweet_id for ids in post_candidates.values() for tweet_id in ids})
            selected_tweets, post_edges, post_summary = finalize_post_selection(
                post_candidates,
                tweet_records=tweet_records,
                seed_user_ids={"u1"},
                thresholds=thresholds,
                seed=5,
            )
            final_tweets, reference_summary = expand_referenced_tweets(
                [tweet_path],
                selected_tweet_records=selected_tweets,
            )
            extra_edges, relation_counts = collect_third_pass_edges(
                edge_path,
                final_user_ids={"u1", "u2"},
                final_tweet_ids=set(final_tweets),
            )

            self.assertEqual(len(user_edges), 1)
            self.assertEqual(post_summary["selected_post_tweet_count"], 2)
            self.assertIn("t99", final_tweets)
            self.assertEqual(reference_summary["referenced_tweets_added"], 1)
            self.assertIn(("t2", "u2", "mention"), extra_edges)
            self.assertEqual(relation_counts["mention"], 1)
            self.assertTrue(any(edge[2] == "post" for edge in post_edges))


if __name__ == "__main__":
    unittest.main()

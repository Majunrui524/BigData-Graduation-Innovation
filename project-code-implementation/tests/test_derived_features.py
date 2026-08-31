from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from twibot22_sampler.post_types import (
    build_user_post_type_distribution,
    classify_post_type_heuristic,
    normalize_post_type_response,
)
from twibot22_sampler.triplets import build_user_triplet_documents, normalize_triplet_response


class DerivedFeatureTests(unittest.TestCase):
    def test_triplet_normalization_accepts_alias_keys(self) -> None:
        payload = normalize_triplet_response(
            {
                "summary": "compressed",
                "relations": [{"subj": "OpenAI", "relation": "builds", "obj": "models"}],
                "confidence": "0.8",
            }
        )
        self.assertEqual(payload["compressed_text"], "compressed")
        self.assertEqual(payload["triplets"][0]["predicate"], "builds")
        self.assertEqual(payload["confidence"], 0.8)

    def test_build_user_triplet_documents_aggregates_lines(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "tweet_triplets.jsonl"
            path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "tweet_id": "t1",
                                "author_id": "u1",
                                "triplets": [{"subject": "a", "predicate": "b", "object": "c"}],
                                "triplet_text": "a | b | c",
                            }
                        ),
                        json.dumps(
                            {
                                "tweet_id": "t2",
                                "author_id": "u1",
                                "triplets": [],
                                "triplet_text": "compressed text",
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            documents = build_user_triplet_documents(path)
            self.assertEqual(documents[0]["author_id"], "u1")
            self.assertEqual(documents[0]["triplet_count"], 1)
            self.assertIn("compressed text", documents[0]["triplet_document"])

    def test_post_type_heuristics_cover_common_cases(self) -> None:
        retweet = classify_post_type_heuristic(
            {
                "text": "RT @acct hello",
                "referenced_tweets": [{"id": "1", "type": "retweeted"}],
            }
        )
        reply = classify_post_type_heuristic(
            {
                "text": "@user hi",
                "in_reply_to_user_id": "u2",
            }
        )
        link = classify_post_type_heuristic(
            {
                "text": "read this",
                "entities": {"urls": [{"expanded_url": "https://example.com/post"}]},
            }
        )
        self.assertEqual(retweet["coarse_type"], "retweet")
        self.assertEqual(reply["detail_type"], "reply")
        self.assertEqual(link["coarse_type"], "link_share")

    def test_post_type_distribution_aggregates_counts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "tweet_post_types.jsonl"
            path.write_text(
                "\n".join(
                    [
                        json.dumps({"tweet_id": "t1", "author_id": "u1", "coarse_type": "retweet", "detail_type": "retweet"}),
                        json.dumps(
                            {
                                "tweet_id": "t2",
                                "author_id": "u1",
                                "coarse_type": "comment_reply",
                                "detail_type": "reply",
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            distribution = build_user_post_type_distribution(path)
            self.assertEqual(distribution[0]["coarse_counts"]["retweet"], 1)
            self.assertEqual(distribution[0]["coarse_distribution"]["comment_reply"], 0.5)

    def test_post_type_normalization_defaults_invalid_values(self) -> None:
        payload = normalize_post_type_response({"label": "weird", "subtype": "odd"})
        self.assertEqual(payload["coarse_type"], "original")
        self.assertEqual(payload["detail_type"], "original")


if __name__ == "__main__":
    unittest.main()

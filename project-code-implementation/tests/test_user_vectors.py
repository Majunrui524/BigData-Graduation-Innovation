from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from twibot22_sampler.user_vectors import build_user_vectors


class _StubEmbeddingClient:
    class settings:  # noqa: D401
        model = "stub-embedding"

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text)), 1.0] for text in texts]


class UserVectorTests(unittest.TestCase):
    def test_build_user_vectors_uses_triplets_then_description_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            feature_root = root / "features"
            output_root = root / "vectors"
            feature_root.mkdir(parents=True)
            (feature_root / "user_feature_table.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "user_id": "u1",
                                "description": "desc one",
                                "triplet_document": "triplet one",
                                "followers_count": 10,
                                "following_count": 2,
                                "account_tweet_count": 3,
                                "tweets_total": 3,
                                "following_in_degree": 1,
                                "following_out_degree": 2,
                                "post_edge_count": 3,
                                "post_type_tweet_count": 3,
                                "triplet_tweet_count": 3,
                                "triplet_count": 4,
                                "tweets_with_created_at": 3,
                                "tweets_with_public_metrics": 3,
                                "tweets_with_references": 1,
                                "tweets_with_external_url": 1,
                                "post_type_coarse_ratio_original": 0.2,
                                "post_type_coarse_ratio_retweet": 0.3,
                                "post_type_coarse_ratio_comment_reply": 0.4,
                                "post_type_coarse_ratio_link_share": 0.1,
                                "post_type_detail_ratio_original": 0.2,
                                "post_type_detail_ratio_retweet": 0.3,
                                "post_type_detail_ratio_reply": 0.2,
                                "post_type_detail_ratio_quote_comment": 0.1,
                                "post_type_detail_ratio_link_share": 0.2,
                                "post_type_detail_ratio_other": 0.0,
                                "verified": 0,
                                "description_present": 1,
                                "profile_url_present": 1,
                                "can_time_feature": 1,
                                "can_behavior_feature": 1,
                                "can_network_feature": 1,
                                "can_full_pipeline": 1,
                                "post_type_incomplete_flag": 0,
                                "triplet_incomplete_flag": 0,
                            }
                        ),
                        json.dumps(
                            {
                                "user_id": "u2",
                                "description": "fallback text",
                                "triplet_document": "",
                                "followers_count": 5,
                                "following_count": 1,
                                "account_tweet_count": 2,
                                "tweets_total": 2,
                                "following_in_degree": 0,
                                "following_out_degree": 1,
                                "post_edge_count": 2,
                                "post_type_tweet_count": 2,
                                "triplet_tweet_count": 0,
                                "triplet_count": 0,
                                "tweets_with_created_at": 2,
                                "tweets_with_public_metrics": 2,
                                "tweets_with_references": 0,
                                "tweets_with_external_url": 0,
                                "post_type_coarse_ratio_original": 1.0,
                                "post_type_coarse_ratio_retweet": 0.0,
                                "post_type_coarse_ratio_comment_reply": 0.0,
                                "post_type_coarse_ratio_link_share": 0.0,
                                "post_type_detail_ratio_original": 1.0,
                                "post_type_detail_ratio_retweet": 0.0,
                                "post_type_detail_ratio_reply": 0.0,
                                "post_type_detail_ratio_quote_comment": 0.0,
                                "post_type_detail_ratio_link_share": 0.0,
                                "post_type_detail_ratio_other": 0.0,
                                "verified": 1,
                                "description_present": 1,
                                "profile_url_present": 0,
                                "can_time_feature": 1,
                                "can_behavior_feature": 1,
                                "can_network_feature": 0,
                                "can_full_pipeline": 0,
                                "post_type_incomplete_flag": 0,
                                "triplet_incomplete_flag": 1,
                            }
                        ),
                        json.dumps(
                            {
                                "user_id": "u3",
                                "description": "",
                                "triplet_document": "",
                                "followers_count": 1,
                                "following_count": 0,
                                "account_tweet_count": 1,
                                "tweets_total": 1,
                                "following_in_degree": 0,
                                "following_out_degree": 0,
                                "post_edge_count": 1,
                                "post_type_tweet_count": 0,
                                "triplet_tweet_count": 0,
                                "triplet_count": 0,
                                "tweets_with_created_at": 1,
                                "tweets_with_public_metrics": 1,
                                "tweets_with_references": 0,
                                "tweets_with_external_url": 0,
                                "post_type_coarse_ratio_original": 0.0,
                                "post_type_coarse_ratio_retweet": 0.0,
                                "post_type_coarse_ratio_comment_reply": 0.0,
                                "post_type_coarse_ratio_link_share": 0.0,
                                "post_type_detail_ratio_original": 0.0,
                                "post_type_detail_ratio_retweet": 0.0,
                                "post_type_detail_ratio_reply": 0.0,
                                "post_type_detail_ratio_quote_comment": 0.0,
                                "post_type_detail_ratio_link_share": 0.0,
                                "post_type_detail_ratio_other": 0.0,
                                "verified": 0,
                                "description_present": 0,
                                "profile_url_present": 0,
                                "can_time_feature": 0,
                                "can_behavior_feature": 0,
                                "can_network_feature": 0,
                                "can_full_pipeline": 0,
                                "post_type_incomplete_flag": 0,
                                "triplet_incomplete_flag": 0,
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            manifest = build_user_vectors(
                feature_root,
                output_root,
                client=_StubEmbeddingClient(),
                batch_size=2,
            )

            self.assertEqual(manifest["counts"]["users"], 3)
            self.assertEqual(manifest["counts"]["embedded_users"], 2)
            self.assertEqual(manifest["embedding_dim"], 2)

            with (output_root / "user_embedding_vectors.jsonl").open() as handle:
                embedding_rows = [json.loads(line) for line in handle if line.strip()]
            by_user = {row["user_id"]: row for row in embedding_rows}
            self.assertEqual(by_user["u1"]["text_source"], "triplet_document")
            self.assertEqual(by_user["u2"]["text_source"], "description_fallback")
            self.assertEqual(by_user["u3"]["embedding"], [0.0, 0.0])

            with (output_root / "user_fused_vectors.jsonl").open() as handle:
                fused_rows = [json.loads(line) for line in handle if line.strip()]
            self.assertEqual(len(fused_rows), 3)
            self.assertGreater(fused_rows[0]["fused_dim"], 2)


if __name__ == "__main__":
    unittest.main()

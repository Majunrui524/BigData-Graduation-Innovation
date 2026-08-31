from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from twibot22_sampler.readers import read_csv_rows, write_csv, write_jsonl
from twibot22_sampler.user_graph import build_user_graph


class UserGraphTests(unittest.TestCase):
    def test_build_user_graph_python_backend_early_fusion(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            vector_root = root / "vectors"
            output_root = root / "graph"
            vector_root.mkdir(parents=True)
            rows = [
                {
                    "user_id": "u1",
                    "text_source": "triplet_document",
                    "embedding_dim": 2,
                    "numeric_dim": 1,
                    "fused_dim": 3,
                    "fused_vector": [1.0, 0.0, 0.0],
                },
                {
                    "user_id": "u2",
                    "text_source": "triplet_document",
                    "embedding_dim": 2,
                    "numeric_dim": 1,
                    "fused_dim": 3,
                    "fused_vector": [0.9, 0.1, 0.0],
                },
                {
                    "user_id": "u3",
                    "text_source": "triplet_document",
                    "embedding_dim": 2,
                    "numeric_dim": 1,
                    "fused_dim": 3,
                    "fused_vector": [0.0, 1.0, 0.0],
                },
                {
                    "user_id": "u4",
                    "text_source": "missing",
                    "embedding_dim": 2,
                    "numeric_dim": 1,
                    "fused_dim": 3,
                    "fused_vector": [-1.0, 0.0, 0.0],
                },
            ]
            (vector_root / "user_fused_vectors.jsonl").write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )

            manifest = build_user_graph(
                vector_root,
                output_root,
                fusion_mode="early",
                k=1,
                min_similarity=0.05,
                backend="python",
            )

            self.assertEqual(manifest["fusion_mode"], "early")
            self.assertEqual(manifest["counts"]["users"], 4)
            self.assertEqual(manifest["counts"]["directed_edges"], 3)
            self.assertEqual(manifest["counts"]["undirected_edges"], 2)
            self.assertEqual(manifest["backend"], "python")

            directed_rows = list(read_csv_rows(output_root / "user_knn_directed_edges.csv"))
            directed_pairs = {(row["source_user_id"], row["target_user_id"]) for row in directed_rows}
            self.assertEqual(directed_pairs, {("u1", "u2"), ("u2", "u1"), ("u3", "u2")})

            undirected_rows = list(read_csv_rows(output_root / "user_knn_edges.csv"))
            by_pair = {
                (row["source_user_id"], row["target_user_id"]): row
                for row in undirected_rows
            }
            self.assertEqual(set(by_pair), {("u1", "u2"), ("u2", "u3")})
            self.assertEqual(by_pair[("u1", "u2")]["support"], "2")
            self.assertEqual(by_pair[("u2", "u3")]["support"], "1")

    def test_build_user_graph_mutual_mode_filters_one_way_edges(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            vector_root = root / "vectors"
            output_root = root / "graph"
            vector_root.mkdir(parents=True)
            rows = [
                {"user_id": "u1", "fused_vector": [1.0, 0.0]},
                {"user_id": "u2", "fused_vector": [0.9, 0.1]},
                {"user_id": "u3", "fused_vector": [0.0, 1.0]},
            ]
            (vector_root / "user_fused_vectors.jsonl").write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )

            manifest = build_user_graph(
                vector_root,
                output_root,
                fusion_mode="early",
                k=1,
                min_similarity=0.0,
                backend="python",
                symmetrize="mutual_max",
            )

            self.assertEqual(manifest["counts"]["undirected_edges"], 1)
            undirected_rows = list(read_csv_rows(output_root / "user_knn_edges.csv"))
            self.assertEqual(len(undirected_rows), 1)
            self.assertEqual(undirected_rows[0]["source_user_id"], "u1")
            self.assertEqual(undirected_rows[0]["target_user_id"], "u2")
            self.assertEqual(undirected_rows[0]["support"], "2")

    def test_build_user_graph_late_fusion_outputs_detailed_edges(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            sample_root = root / "sample"
            vector_root = sample_root / "analysis" / "user_vectors"
            feature_root = sample_root / "analysis" / "user_features"
            temporal_root = sample_root / "analysis" / "temporal_profiles"
            output_root = sample_root / "analysis" / "user_graph"
            sample_root.mkdir(parents=True)
            vector_root.mkdir(parents=True)
            feature_root.mkdir(parents=True)
            temporal_root.mkdir(parents=True)

            write_jsonl(
                sample_root / "user.jsonl",
                [
                    {"id": "u1"},
                    {"id": "u2"},
                    {"id": "u3"},
                ],
            )
            write_csv(
                sample_root / "edge.csv",
                ["source_id", "target_id", "relation"],
                [
                    {"source_id": "u1", "target_id": "u2", "relation": "following"},
                    {"source_id": "u2", "target_id": "u1", "relation": "following"},
                    {"source_id": "u3", "target_id": "u2", "relation": "following"},
                ],
            )
            write_jsonl(
                vector_root / "user_embedding_vectors.jsonl",
                [
                    {"user_id": "u1", "text_source": "triplet_document", "embedding_dim": 2, "embedding": [1.0, 0.0]},
                    {"user_id": "u2", "text_source": "triplet_document", "embedding_dim": 2, "embedding": [0.95, 0.05]},
                    {"user_id": "u3", "text_source": "triplet_document", "embedding_dim": 2, "embedding": [-1.0, 0.0]},
                ],
            )
            write_jsonl(
                feature_root / "user_feature_table.jsonl",
                [
                    _feature_row(
                        "u1",
                        followers=100,
                        following=50,
                        tweets_total=12,
                        post_type_tweets=8,
                        triplet_tweets=8,
                        verified=0,
                        coarse=[0.75, 0.10, 0.10, 0.05],
                    ),
                    _feature_row(
                        "u2",
                        followers=110,
                        following=55,
                        tweets_total=11,
                        post_type_tweets=8,
                        triplet_tweets=8,
                        verified=0,
                        coarse=[0.78, 0.08, 0.09, 0.05],
                    ),
                    _feature_row(
                        "u3",
                        followers=20,
                        following=200,
                        tweets_total=9,
                        post_type_tweets=8,
                        triplet_tweets=8,
                        verified=1,
                        coarse=[0.10, 0.70, 0.10, 0.10],
                    ),
                ],
            )
            write_jsonl(
                temporal_root / "user_temporal_profiles.jsonl",
                [
                    {
                        "user_id": "u1",
                        "created_at_tweets": 8,
                        "temporal_ready": 1,
                        "utc_hour_distribution": [0.25, 0.25, 0.25, 0.25] + [0.0] * 20,
                    },
                    {
                        "user_id": "u2",
                        "created_at_tweets": 8,
                        "temporal_ready": 1,
                        "utc_hour_distribution": [0.24, 0.26, 0.25, 0.25] + [0.0] * 20,
                    },
                    {
                        "user_id": "u3",
                        "created_at_tweets": 8,
                        "temporal_ready": 1,
                        "utc_hour_distribution": [0.0] * 20 + [0.25, 0.25, 0.25, 0.25],
                    },
                ],
            )

            manifest = build_user_graph(
                sample_root,
                output_root,
                fusion_mode="late",
                vector_root=vector_root,
                feature_root=feature_root,
                temporal_root=temporal_root,
                k=1,
                candidate_k=2,
                min_similarity=0.0,
                backend="python",
            )

            self.assertEqual(manifest["fusion_mode"], "late")
            self.assertEqual(manifest["counts"]["users"], 3)
            self.assertTrue((output_root / "user_knn_edges_detailed.csv").exists())
            self.assertGreater(manifest["channel_edge_coverage"]["content"]["directed_edges"], 0)
            self.assertGreater(manifest["channel_edge_coverage"]["behavior"]["directed_edges"], 0)
            self.assertGreater(manifest["channel_edge_coverage"]["temporal"]["directed_edges"], 0)
            self.assertGreater(manifest["channel_edge_coverage"]["network"]["directed_edges"], 0)

            undirected_rows = list(read_csv_rows(output_root / "user_knn_edges.csv"))
            detailed_rows = list(read_csv_rows(output_root / "user_knn_edges_detailed.csv"))
            pairs = {(row["source_user_id"], row["target_user_id"]) for row in undirected_rows}
            self.assertIn(("u1", "u2"), pairs)
            detailed_by_pair = {
                (row["source_user_id"], row["target_user_id"]): row
                for row in detailed_rows
            }
            self.assertNotEqual(detailed_by_pair[("u1", "u2")]["content_similarity"], "")
            self.assertNotEqual(detailed_by_pair[("u1", "u2")]["behavior_similarity"], "")
            self.assertNotEqual(detailed_by_pair[("u1", "u2")]["temporal_similarity"], "")
            self.assertNotEqual(detailed_by_pair[("u1", "u2")]["network_similarity"], "")


def _feature_row(
    user_id: str,
    *,
    followers: int,
    following: int,
    tweets_total: int,
    post_type_tweets: int,
    triplet_tweets: int,
    verified: int,
    coarse: list[float],
) -> dict[str, object]:
    return {
        "user_id": user_id,
        "followers_count": followers,
        "following_count": following,
        "tweets_total": tweets_total,
        "post_type_tweet_count": post_type_tweets,
        "triplet_tweet_count": triplet_tweets,
        "verified": verified,
        "tweets_with_created_at_ratio": 1.0,
        "tweets_with_public_metrics_ratio": 0.75,
        "tweets_with_references_ratio": 0.25,
        "tweets_with_external_url_ratio": 0.10,
        "post_type_coarse_ratio_original": coarse[0],
        "post_type_coarse_ratio_retweet": coarse[1],
        "post_type_coarse_ratio_comment_reply": coarse[2],
        "post_type_coarse_ratio_link_share": coarse[3],
    }


if __name__ == "__main__":
    unittest.main()

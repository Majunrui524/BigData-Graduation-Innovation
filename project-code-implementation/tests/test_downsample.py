from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from twibot22_sampler.downsample import downsample_exported_sample
from twibot22_sampler.validate import validate_sample


class DownsampleTests(unittest.TestCase):
    def test_downsample_exported_sample_builds_consistent_subset(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            sample_root = root / "sample"
            output_root = root / "small"
            sample_root.mkdir(parents=True)

            users = [
                {"id": "u1", "verified": False, "public_metrics": {"followers_count": 10, "following_count": 1}},
                {"id": "u2", "verified": True, "public_metrics": {"followers_count": 20, "following_count": 2}},
                {"id": "u3", "verified": False, "public_metrics": {"followers_count": 30, "following_count": 3}},
                {"id": "u4", "verified": False, "public_metrics": {"followers_count": 40, "following_count": 4}},
                {"id": "u5", "verified": True, "public_metrics": {"followers_count": 50, "following_count": 5}},
                {"id": "u6", "verified": False, "public_metrics": {"followers_count": 60, "following_count": 6}},
            ]
            (sample_root / "user.jsonl").write_text(
                "\n".join(json.dumps(record) for record in users) + "\n",
                encoding="utf-8",
            )
            tweets = [
                {"id": "t1", "author_id": 1, "text": "a"},
                {"id": "t2", "author_id": 1, "text": "b"},
                {"id": "t3", "author_id": 2, "text": "c"},
                {"id": "t4", "author_id": 3, "text": "d"},
                {"id": "t5", "author_id": 4, "text": "e"},
                {"id": "t6", "author_id": 5, "text": "f"},
                {"id": "t7", "author_id": 6, "text": "g"},
            ]
            (sample_root / "tweet_0.jsonl").write_text(
                "\n".join(json.dumps(record) for record in tweets) + "\n",
                encoding="utf-8",
            )
            (sample_root / "edge.csv").write_text(
                "\n".join(
                    [
                        "source_id,target_id,relation",
                        "u1,u2,following",
                        "u2,u3,following",
                        "u3,u4,following",
                        "u4,u5,following",
                        "u5,u6,following",
                        "u1,t1,post",
                        "u1,t2,post",
                        "u2,t3,post",
                        "u3,t4,post",
                        "u4,t5,post",
                        "u5,t6,post",
                        "u6,t7,post",
                        "t3,u1,mention",
                        "t4,t3,reply",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            with (sample_root / "split.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["id", "split"])
                writer.writeheader()
                for row in (
                    {"id": "u1", "split": "train"},
                    {"id": "u2", "split": "train"},
                    {"id": "u3", "split": "train"},
                    {"id": "u4", "split": "test"},
                    {"id": "u5", "split": "test"},
                    {"id": "u6", "split": "valid"},
                ):
                    writer.writerow(row)
            with (sample_root / "label.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["id", "label"])
                writer.writeheader()
                for row in (
                    {"id": "u1", "label": "human"},
                    {"id": "u2", "label": "human"},
                    {"id": "u3", "label": "bot"},
                    {"id": "u4", "label": "human"},
                    {"id": "u5", "label": "bot"},
                    {"id": "u6", "label": "human"},
                ):
                    writer.writerow(row)
            (sample_root / "sample_manifest.json").write_text("{}", encoding="utf-8")

            manifest = downsample_exported_sample(sample_root, output_root, target_users=4, seed=7)
            result = validate_sample(output_root)

            self.assertEqual(manifest["final_counts"]["users"], 4)
            self.assertTrue(result["tweet_budget_ok"])
            self.assertEqual(result["endpoint_error_count"], 0)
            with (output_root / "user.jsonl").open() as handle:
                self.assertEqual(sum(1 for _ in handle), 4)
            self.assertTrue((output_root / "sample_stats.md").exists())


if __name__ == "__main__":
    unittest.main()

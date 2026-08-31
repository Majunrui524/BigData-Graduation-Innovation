from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from twibot22_sampler.validate import validate_sample


class ValidateTests(unittest.TestCase):
    def test_validate_sample_passes_on_consistent_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "user.jsonl").write_text(
                '\n'.join(['{"id":"u1"}', '{"id":"u2"}']) + "\n",
                encoding="utf-8",
            )
            (root / "tweet_0.jsonl").write_text(
                '\n'.join(['{"id":"t1"}', '{"id":"t2"}']) + "\n",
                encoding="utf-8",
            )
            (root / "edge.csv").write_text(
                "\n".join(
                    [
                        "source_id,target_id,relation",
                        "u1,u2,following",
                        "u1,t1,post",
                        "u2,t2,post",
                        "t1,u2,mention",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (root / "split.csv").write_text("id,split\nu1,train\nu2,test\n", encoding="utf-8")
            (root / "label.csv").write_text("id,label\nu1,bot\nu2,human\n", encoding="utf-8")
            manifest = {
                "seed_sampling": {
                    "split_quotas": {"train": 1, "test": 1},
                    "label_quotas": {"train": {"bot": 1}, "test": {"human": 1}},
                },
                "tweet_budget": {"limit": 10},
            }
            (root / "sample_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

            report_path = root / "validation.md"
            result = validate_sample(root, report_path)

            self.assertTrue(result["passed"])
            self.assertTrue(report_path.exists())


if __name__ == "__main__":
    unittest.main()

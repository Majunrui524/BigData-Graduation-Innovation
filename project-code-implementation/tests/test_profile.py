from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from twibot22_sampler.profile import build_user_profile


class ProfileTests(unittest.TestCase):
    def test_build_user_profile_extracts_aliases_and_pool_flags(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_root = root / "raw"
            work_root = root / "work"
            data_root.mkdir(parents=True)

            users = [
                {
                    "id": "u1",
                    "public_metrics": {
                        "followers_count": 12,
                        "following_count": 5,
                        "tweet_count": 9,
                    },
                    "verified": True,
                },
                {
                    "id": "u2",
                    "followers_count": 3,
                    "friends_count": 7,
                    "statuses_count": 4,
                },
            ]
            (data_root / "user.json").write_text(json.dumps(users), encoding="utf-8")
            (data_root / "split.csv").write_text("id,split\nu1,train\nu2,test\n", encoding="utf-8")
            (data_root / "label.csv").write_text("id,label\nu1,bot\nu2,human\n", encoding="utf-8")

            output_path = build_user_profile(data_root, work_root)
            with output_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))

            self.assertEqual(len(rows), 2)
            first = rows[0]
            self.assertEqual(first["user_id"], "u1")
            self.assertEqual(first["followers_count"], "12")
            self.assertEqual(first["following_count"], "5")
            self.assertEqual(first["tweet_count_hint"], "9")
            self.assertEqual(first["degree_proxy"], "17")
            self.assertEqual(first["verified_bucket"], "true")
            self.assertEqual(first["primary_pool"], "1")
            self.assertEqual(first["sparse_pool"], "0")

            second = rows[1]
            self.assertEqual(second["verified_bucket"], "missing")
            self.assertEqual(second["primary_pool"], "0")
            self.assertEqual(second["sparse_pool"], "1")


if __name__ == "__main__":
    unittest.main()

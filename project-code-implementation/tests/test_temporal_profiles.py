from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from twibot22_sampler.readers import read_jsonl_records, write_csv, write_jsonl
from twibot22_sampler.temporal_profiles import build_temporal_profiles


class TemporalProfilesTests(unittest.TestCase):
    def test_build_temporal_profiles_counts_hours_and_readiness(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            sample_root = root / "sample"
            output_root = sample_root / "analysis" / "temporal_profiles"
            sample_root.mkdir(parents=True)

            write_jsonl(
                sample_root / "user.jsonl",
                [
                    {"id": "u1"},
                    {"id": "u2"},
                ],
            )
            write_csv(
                sample_root / "edge.csv",
                ["source_id", "target_id", "relation"],
                [
                    {"source_id": "u1", "target_id": "t1", "relation": "post"},
                    {"source_id": "u1", "target_id": "t2", "relation": "post"},
                    {"source_id": "u2", "target_id": "t3", "relation": "post"},
                ],
            )
            write_jsonl(
                sample_root / "tweet_0.jsonl",
                [
                    {"id": "t1", "created_at": "2024-01-01T00:00:00Z"},
                    {"id": "t2", "created_at": "2024-01-01T13:30:00Z"},
                    {"id": "t3", "created_at": ""},
                ],
            )

            manifest = build_temporal_profiles(sample_root, output_root, min_time_tweets=2)

            self.assertEqual(manifest["counts"]["users"], 2)
            self.assertEqual(manifest["counts"]["temporal_ready_users"], 1)
            rows = list(read_jsonl_records(output_root / "user_temporal_profiles.jsonl"))
            by_user = {row["user_id"]: row for row in rows}
            self.assertEqual(by_user["u1"]["created_at_tweets"], 2)
            self.assertEqual(by_user["u1"]["temporal_ready"], 1)
            self.assertEqual(by_user["u1"]["utc_hour_counts"][0], 1)
            self.assertEqual(by_user["u1"]["utc_hour_counts"][13], 1)
            self.assertEqual(by_user["u2"]["temporal_ready"], 0)


if __name__ == "__main__":
    unittest.main()

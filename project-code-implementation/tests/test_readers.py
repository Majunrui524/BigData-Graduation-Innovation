from __future__ import annotations

import json
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path

from twibot22_sampler.readers import write_json, write_jsonl


class ReaderSerializationTests(unittest.TestCase):
    def test_write_jsonl_converts_nested_decimal_values(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "tweet_0.jsonl"
            write_jsonl(
                path,
                [
                    {
                        "id": "t1",
                        "public_metrics": {
                            "like_count": Decimal("3"),
                            "ratio": Decimal("1.5"),
                        },
                        "values": [Decimal("2"), {"quote_count": Decimal("4")}],
                    }
                ],
            )

            payload = json.loads(path.read_text(encoding="utf-8").strip())
            self.assertEqual(payload["public_metrics"]["like_count"], 3)
            self.assertEqual(payload["public_metrics"]["ratio"], 1.5)
            self.assertEqual(payload["values"][0], 2)
            self.assertEqual(payload["values"][1]["quote_count"], 4)

    def test_write_json_converts_decimal_values(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "manifest.json"
            write_json(path, {"count": Decimal("5"), "score": Decimal("0.75")})

            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["count"], 5)
            self.assertEqual(payload["score"], 0.75)


if __name__ == "__main__":
    unittest.main()

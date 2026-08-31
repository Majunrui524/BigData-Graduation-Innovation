from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from twibot22_sampler.external_baseline_common import write_json
from twibot22_sampler.external_baseline_summary import summarize_external_baselines


class ExternalBaselineSummaryTests(unittest.TestCase):
    def test_summarize_external_baselines_collects_completed_methods(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            sample_root = root / "sample"
            baselines_root = sample_root / "analysis" / "external_baselines_10k"
            output_root = baselines_root / "summary"
            (baselines_root / "logreg").mkdir(parents=True)
            write_json(
                baselines_root / "logreg" / "manifest.json",
                {
                    "method_name": "Logistic Regression",
                    "model_family": "feature_supervised",
                    "graph_source": "",
                    "selection_split": "valid",
                    "selected_params": {"C": 1.0},
                },
            )
            write_json(
                baselines_root / "logreg" / "metrics.json",
                {
                    "test": {
                        "accuracy": 0.7,
                        "precision": 0.5,
                        "recall": 0.4,
                        "f1": 0.44,
                        "auc": 0.62,
                    }
                },
            )

            manifest = summarize_external_baselines(sample_root, baselines_root, output_root)
            self.assertEqual(manifest["counts"]["methods"], 1)
            self.assertTrue((output_root / "external_baseline_results.csv").exists())


if __name__ == "__main__":
    unittest.main()

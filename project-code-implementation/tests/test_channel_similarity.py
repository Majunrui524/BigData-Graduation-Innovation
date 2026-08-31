from __future__ import annotations

import unittest

from twibot22_sampler.channel_similarity import (
    dtw_average_distance,
    js_divergence,
    js_similarity,
    weighted_available_average,
)


class ChannelSimilarityTests(unittest.TestCase):
    def test_js_similarity_identical_distribution_is_one(self) -> None:
        distribution = [0.7, 0.2, 0.1, 0.0]
        self.assertAlmostEqual(js_divergence(distribution, distribution), 0.0, places=8)
        self.assertAlmostEqual(js_similarity(distribution, distribution), 1.0, places=8)

    def test_js_similarity_separates_different_peaks(self) -> None:
        left = [1.0, 0.0, 0.0, 0.0]
        right = [0.0, 1.0, 0.0, 0.0]
        similarity = js_similarity(left, right)
        self.assertIsNotNone(similarity)
        self.assertLess(float(similarity), 0.1)

    def test_dtw_average_distance_zero_for_identical_series(self) -> None:
        series = [0.0, 0.3, 0.4, 0.3]
        self.assertAlmostEqual(dtw_average_distance(series, series), 0.0, places=8)

    def test_dtw_average_distance_positive_for_shifted_series(self) -> None:
        left = [0.5, 0.5, 0.0, 0.0]
        right = [0.0, 0.5, 0.5, 0.0]
        distance = dtw_average_distance(left, right)
        self.assertIsNotNone(distance)
        self.assertGreater(float(distance), 0.0)

    def test_weighted_available_average_renormalizes_missing_channels(self) -> None:
        fused_score, available_weight_sum = weighted_available_average(
            {
                "content": 0.8,
                "behavior": None,
                "temporal": 0.6,
                "network": None,
            },
            {
                "content": 0.25,
                "behavior": 0.25,
                "temporal": 0.25,
                "network": 0.25,
            },
        )
        self.assertAlmostEqual(float(available_weight_sum), 0.5, places=8)
        self.assertAlmostEqual(float(fused_score), 0.7, places=8)


if __name__ == "__main__":
    unittest.main()

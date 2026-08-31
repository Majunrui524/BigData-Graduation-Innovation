"""Channel-level similarity helpers for late-fusion graph construction."""

from __future__ import annotations

import math
from typing import Iterable


def cosine_similarity(left: Iterable[float], right: Iterable[float]) -> float | None:
    """Return cosine similarity, or None when either vector has zero norm."""

    left_values = [float(value) for value in left]
    right_values = [float(value) for value in right]
    if len(left_values) != len(right_values):
        raise ValueError("Cosine similarity requires vectors of equal length")

    dot_product = sum(a * b for a, b in zip(left_values, right_values))
    left_norm = math.sqrt(sum(value * value for value in left_values))
    right_norm = math.sqrt(sum(value * value for value in right_values))
    if left_norm <= 0.0 or right_norm <= 0.0:
        return None
    return dot_product / (left_norm * right_norm)


def normalized_cosine_similarity(left: Iterable[float], right: Iterable[float]) -> float | None:
    """Map cosine similarity from [-1, 1] into [0, 1]."""

    similarity = cosine_similarity(left, right)
    if similarity is None:
        return None
    return max(0.0, min(1.0, (similarity + 1.0) / 2.0))


def jaccard_similarity(left: set[str], right: set[str]) -> float | None:
    """Return Jaccard similarity for two neighbor sets."""

    if not left and not right:
        return None
    union = left | right
    if not union:
        return None
    return len(left & right) / len(union)


def js_divergence(left: Iterable[float], right: Iterable[float]) -> float | None:
    """Return Jensen-Shannon divergence in [0, 1] for normalized distributions."""

    left_values = [max(float(value), 0.0) for value in left]
    right_values = [max(float(value), 0.0) for value in right]
    if len(left_values) != len(right_values):
        raise ValueError("JS divergence requires distributions of equal length")

    left_total = sum(left_values)
    right_total = sum(right_values)
    if left_total <= 0.0 or right_total <= 0.0:
        return None

    left_distribution = [value / left_total for value in left_values]
    right_distribution = [value / right_total for value in right_values]
    midpoint = [(a + b) / 2.0 for a, b in zip(left_distribution, right_distribution)]

    def _kl_divergence(source: list[float], target: list[float]) -> float:
        score = 0.0
        for source_value, target_value in zip(source, target):
            if source_value <= 0.0:
                continue
            if target_value <= 0.0:
                continue
            score += source_value * math.log2(source_value / target_value)
        return score

    divergence = 0.5 * _kl_divergence(left_distribution, midpoint)
    divergence += 0.5 * _kl_divergence(right_distribution, midpoint)
    return max(0.0, min(1.0, divergence))


def js_similarity(left: Iterable[float], right: Iterable[float]) -> float | None:
    """Convert JS divergence into a similarity score."""

    divergence = js_divergence(left, right)
    if divergence is None:
        return None
    return max(0.0, 1.0 - divergence)


def dtw_average_distance(left: Iterable[float], right: Iterable[float]) -> float | None:
    """Return length-normalized DTW distance for two equal-length sequences."""

    left_values = [float(value) for value in left]
    right_values = [float(value) for value in right]
    if not left_values or not right_values:
        return None
    if len(left_values) != len(right_values):
        raise ValueError("DTW average distance requires sequences of equal length")

    rows = len(left_values)
    cols = len(right_values)
    cost = [[math.inf for _ in range(cols + 1)] for _ in range(rows + 1)]
    steps = [[0 for _ in range(cols + 1)] for _ in range(rows + 1)]
    cost[0][0] = 0.0

    for row_index in range(1, rows + 1):
        for col_index in range(1, cols + 1):
            local_cost = abs(left_values[row_index - 1] - right_values[col_index - 1])
            candidates = (
                (cost[row_index - 1][col_index], steps[row_index - 1][col_index]),
                (cost[row_index][col_index - 1], steps[row_index][col_index - 1]),
                (cost[row_index - 1][col_index - 1], steps[row_index - 1][col_index - 1]),
            )
            best_cost, best_steps = min(candidates, key=lambda item: (item[0], item[1]))
            cost[row_index][col_index] = local_cost + best_cost
            steps[row_index][col_index] = best_steps + 1

    total_cost = cost[rows][cols]
    path_length = steps[rows][cols]
    if not math.isfinite(total_cost) or path_length <= 0:
        return None
    return total_cost / path_length


def weighted_available_average(
    scores: dict[str, float | None],
    weights: dict[str, float],
) -> tuple[float | None, float]:
    """Average only over channels with both a score and a positive weight."""

    weighted_sum = 0.0
    available_weight_sum = 0.0
    for channel_name, score in scores.items():
        weight = float(weights.get(channel_name, 0.0))
        if weight <= 0.0 or score is None or not math.isfinite(float(score)):
            continue
        weighted_sum += weight * float(score)
        available_weight_sum += weight
    if available_weight_sum <= 0.0:
        return None, 0.0
    return weighted_sum / available_weight_sum, available_weight_sum

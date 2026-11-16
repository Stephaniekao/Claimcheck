"""Unit tests for dual threshold selection."""

import itertools
import unittest

import numpy as np

from claimcheck_classical.evaluation.thresholds import (
    select_dual_thresholds,
)
from claimcheck_classical.evaluation.metrics import (
    evaluate_three_label,
)


# ============================================================
# 1. HELPERS
# ============================================================

def brute_force_best_recall(y, p, min_precision, max_rate):
    """Search every threshold pair and return the best strict recall.

    Returns None when no pair meets both conditions.
    """

    n = len(y)
    n_unsupported = int((y == 1).sum())
    max_uncertain = int(np.floor(max_rate * n))

    values = np.unique(p)
    midpoints = (values[:-1] + values[1:]) / 2

    candidates = sorted(
        set(values) | set(midpoints)
        | {np.nextafter(values[-1], 1.0), 1.0}
    )
    candidates = [c for c in candidates if 0 <= c <= 1]

    best = None

    for lower, upper in itertools.product(candidates, candidates):

        if not lower < upper:
            continue

        predicted_unsupported = p < lower
        predicted_supported = p > upper

        uncertain = ~predicted_unsupported & ~predicted_supported

        if uncertain.sum() > max_uncertain:
            continue

        true_positive = int((predicted_unsupported & (y == 1)).sum())
        predicted = int(predicted_unsupported.sum())

        precision = true_positive / predicted if predicted else 0.0

        if precision < min_precision:
            continue

        recall = true_positive / n_unsupported

        if best is None or recall > best:
            best = recall

    return best


# ============================================================
# 2. THRESHOLD TESTS
# ============================================================

class TestDualThresholds(unittest.TestCase):

    def test_matches_exhaustive_search(self):
        rng = np.random.default_rng(0)

        for _ in range(60):

            n = int(rng.integers(30, 100))

            y = (rng.random(n) < rng.choice([0.05, 0.2, 0.5])).astype(int)
            y[0] = 1

            # Few distinct levels create many tied probabilities.
            levels = int(rng.choice([4, 10, 1000]))

            p = rng.random(n) * 0.8 + 0.1 * (1 - y) * rng.random(n)
            p = np.clip(np.round(p * levels) / levels, 0, 0.999)

            for min_precision, max_rate in [(0.6, 1 / 3), (0.3, 1 / 3)]:

                result = select_dual_thresholds(
                    y,
                    p,
                    min_precision=min_precision,
                    max_uncertain_rate=max_rate,
                )

                expected = brute_force_best_recall(
                    y, p, min_precision, max_rate
                )

                self.assertEqual(result["feasible"], expected is not None)

                if expected is not None:
                    self.assertAlmostEqual(
                        result["strict_recall"], expected, places=12
                    )

    def test_selected_thresholds_match_evaluator(self):
        y = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
        p = np.array([0.1, 0.2, 0.3, 0.6, 0.7, 0.8, 0.9, 0.95, 0.9, 0.85])

        result = select_dual_thresholds(y, p)

        self.assertTrue(result["feasible"])

        labels = np.where(
            p < result["lower_threshold"],
            "Unsupported",
            np.where(p > result["upper_threshold"], "Supported", "Uncertain"),
        )

        metrics = evaluate_three_label(y, labels)

        self.assertAlmostEqual(
            metrics["strict_recall"], result["strict_recall"]
        )
        self.assertAlmostEqual(
            metrics["unsupported_precision"],
            result["unsupported_precision"],
        )

    def test_no_signal_is_infeasible(self):
        y = np.array([1, 0] * 10)
        p = np.full(20, 0.9)

        result = select_dual_thresholds(y, p)

        self.assertFalse(result["feasible"])

    def test_requires_unsupported_claims(self):
        with self.assertRaises(ValueError):
            select_dual_thresholds(np.zeros(10, dtype=int), np.linspace(0, 1, 10))


if __name__ == "__main__":
    unittest.main()

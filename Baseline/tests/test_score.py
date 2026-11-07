"""python -m unittest discover Baseline/tests  (support score 1.0.0)"""
import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from claimcheck_baseline import metrics, score  # noqa: E402
from claimcheck_baseline.rules import verify  # noqa: E402

PASSAGE = "\n".join([
    "Revenues for the fiscal year ended 2025-12-31: USD 4,100 million.",
    "Revenues for the fiscal year ended 2024-12-31: USD 3,900 million.",
    "Operating Income (Loss) for the fiscal year ended 2025-12-31: USD 800 million.",
])
CLAIMS = [
    "ACME's revenue for the fiscal period ended 2025-12-31 was USD 4,100 million.",   # Supported
    "ACME's revenue for the fiscal period ended 2025-12-31 was USD 4,920 million.",   # wrong number
    "ACME's revenue for the fiscal period ended 2024-12-31 was USD 4,100 million.",   # wrong year
    "ACME's operating income for the fiscal period ended 2025-12-31 was USD 4,100 million.",  # wrong item
    "Revenue grew because of strong demand.",                                          # no number
]


class SupportScore(unittest.TestCase):
    def test_label_is_unchanged_and_band_matches_it(self):
        for c in CLAIMS:
            s, v = score.support_score(c, PASSAGE)
            self.assertEqual(v.label, verify(c, PASSAGE).label, c)
            self.assertEqual(score.label_of(s), v.label, c)
            self.assertTrue(0.0 <= s <= 1.0)

    def test_right_number_wrong_item_or_year_is_never_supported(self):
        for c in CLAIMS[2:4]:
            s, _ = score.support_score(c, PASSAGE)
            self.assertLess(s, score.UPPER, c)

    def test_supported_ranks_above_contradicted(self):
        good, _ = score.support_score(CLAIMS[0], PASSAGE)
        bad, _ = score.support_score(CLAIMS[1], PASSAGE)
        self.assertGreater(good, bad)

    def test_no_passage_scores_in_uncertain_band(self):
        s, v = score.support_score(CLAIMS[0], "")
        self.assertEqual((v.label, score.label_of(s)), ("Uncertain", "Uncertain"))


class ScoreMetrics(unittest.TestCase):
    gold = pd.Series(["Unsupported", "Unsupported", "Supported", "Supported"])

    def test_auroc_perfect_reversed_and_ties(self):
        self.assertEqual(metrics.auroc(self.gold, pd.Series([0.9, 0.8, 0.2, 0.1])), 1.0)
        self.assertEqual(metrics.auroc(self.gold, pd.Series([0.1, 0.2, 0.8, 0.9])), 0.0)
        self.assertEqual(metrics.auroc(self.gold, pd.Series([0.5] * 4)), 0.5)

    def test_precision_at_recall(self):
        p = metrics.precision_at_recall(self.gold, pd.Series([0.9, 0.3, 0.5, 0.1]), 0.80)
        self.assertAlmostEqual(p["precision"], 2 / 3)


if __name__ == "__main__":
    unittest.main()

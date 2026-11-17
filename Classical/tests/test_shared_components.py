"""Tests for the shared normalizer, passages format and team threshold rules.

Cases follow Baseline/tests/test_baseline.py, so both methods read the
same numbers the same way (Experiment Card, Field 12).
"""

import unittest

import numpy as np
import pandas as pd

from claimcheck_classical.preprocessing import normalize_text
from claimcheck_classical.features.numerical_features import (
    compute_numerical_consistency,
)
from claimcheck_classical.features.scope_features import (
    compute_scope_consistency,
)
from claimcheck_classical.features.evidence_features import (
    select_answer_aggregation,
)
from claimcheck_classical.evaluation.thresholds import (
    select_dual_thresholds,
    select_thresholds_with_forced_uncertain,
    widen_upper_threshold,
)
from claimcheck_classical.evaluation.metrics import evaluate_three_label
from claimcheck_classical.inference.predict import (
    classify_support_probability,
)
from claimcheck_classical.shared import normalizer_cannot_read

SEGMENTS = (
    "ACME revenue by segment for the fiscal year ended 2025-12-31:\n"
    "CloudMember segment revenue for the fiscal year ended 2025-12-31: USD 2,000,000,000.\n"
    "Total revenues for the fiscal year ended 2025-12-31: USD 4,100,000,000."
)
NI = (
    "Net Income (Loss) Attributable to Parent for the fiscal year "
    "ended 2025-12-31: USD 7,085 million."
)


def features(claim, evidence):
    claim = normalize_text(claim)
    return (
        compute_numerical_consistency(claim, evidence),
        compute_scope_consistency(claim, evidence),
    )


class TestSharedNormalizerFeatures(unittest.TestCase):

    def test_storyboard_b_error_has_wrong_scope(self):
        claim = ("ACME's CloudMember revenue for the fiscal period ended 2025-12-31 "
                 "represented the company's entire revenue of USD 2,000,000,000.")
        self.assertEqual(features(claim, SEGMENTS), (1.0, 0.0))

    def test_storyboard_b_correct_segment_claim(self):
        claim = ("ACME's CloudMember segment revenue for the fiscal period "
                 "ended 2025-12-31 was USD 2,000,000,000.")
        self.assertEqual(features(claim, SEGMENTS), (1.0, 1.0))

    def test_dates_are_not_amounts(self):
        claim = "ACME's total revenue for the fiscal period ended 2025-12-31 was $4.1 billion."
        self.assertEqual(features(claim, SEGMENTS), (1.0, 1.0))

    def test_flipped_sign_and_unit_errors(self):
        flipped = "ACME's net income for 2025 was USD -7,085.000 million."
        unit = "ACME's net income for 2025 was USD 7,085 billion."
        same = "ACME's net income for 2025 was USD 7,085 million."
        self.assertEqual(features(flipped, NI)[0], 0.0)
        self.assertEqual(features(unit, NI)[0], 0.0)
        self.assertEqual(features(same, NI)[0], 1.0)

    def test_normalizer_failure_definition(self):
        self.assertTrue(normalizer_cannot_read("the ratio was 1.2x."))
        self.assertFalse(normalizer_cannot_read("revenue grew strongly."))
        self.assertFalse(normalizer_cannot_read("revenue was $5 million in 2019."))


class TestTeamThresholdRules(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(0)
        n = 3000
        cls.y = (rng.random(n) < 0.05).astype(int)
        z = -3.0 + 1.5 * cls.y + rng.normal(0, 1, n)
        cls.p = 1 - 1 / (1 + np.exp(-z))

    def test_widen_keeps_strict_metrics_and_cap(self):
        dual = select_dual_thresholds(self.y, self.p)
        wide = widen_upper_threshold(self.y, self.p, dual)

        labels = np.array([
            classify_support_probability(float(v), wide["lower_threshold"], wide["upper_threshold"])
            for v in self.p
        ])
        m = evaluate_three_label(self.y, labels)

        self.assertAlmostEqual(m["strict_recall"], dual["strict_recall"])
        self.assertAlmostEqual(m["unsupported_precision"], dual["unsupported_precision"])
        self.assertLessEqual(m["uncertain_rate"], 1 / 3)
        self.assertGreaterEqual(m["uncertain_rate"], dual["uncertain_rate"])
        self.assertAlmostEqual(m["lenient_recall"], wide["lenient_recall"])

    def test_forced_claims_are_uncertain_and_count_toward_cap(self):
        forced = np.zeros(len(self.y), dtype=bool)
        forced[:300] = True

        r = select_thresholds_with_forced_uncertain(self.y, self.p, forced)
        labels = r["labels"]

        self.assertTrue((labels[forced] == "Uncertain").all())
        self.assertLessEqual(np.mean(labels == "Uncertain"), 1 / 3)
        self.assertTrue(r["ac2_satisfiable"])

    def test_too_many_forced_claims_break_ac2(self):
        forced = np.zeros(len(self.y), dtype=bool)
        forced[:1500] = True

        r = select_thresholds_with_forced_uncertain(self.y, self.p, forced)
        self.assertFalse(r["ac2_satisfiable"])


class TestAnswerAggregation(unittest.TestCase):

    def test_adherence_gold_and_rule_choice(self):
        df = pd.DataFrame({
            "subset": ["FinQA"] * 6,
            "answer_row": [0, 0, 1, 1, 2, 2],
            "label": ["Unsupported", "Supported", "Supported", "Supported", "Supported", "Supported"],
            "adherence_score": [False, False, True, True, True, True],
        })
        p = np.array([0.2, 0.9, 0.95, 0.9, 0.8, 0.85])

        r = select_answer_aggregation(df, p)

        self.assertEqual(r["gold"], "adherence_score")
        self.assertEqual(r["gold_disagreements"], 0)
        self.assertIn(r["chosen_method"], ("min", "mean"))


if __name__ == "__main__":
    unittest.main()

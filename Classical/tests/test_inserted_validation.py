"""Tests for the Inserted Error Set v2 validation script."""

import unittest

import numpy as np
import pandas as pd

from claimcheck_classical.evaluation import inserted_validation as iv


def toy_results():
    """Eight claims: four general, four Storyboard B, half of each wrong."""

    return pd.DataFrame({
        "subset": ["general"] * 4 + ["storyboard_b"] * 4,
        "label": ["Supported", "Unsupported"] * 4,
        "error_type": ["Supported", "Small Numerical"] * 2
        + ["Supported Segment Revenue", "Segment vs Total Revenue"] * 2,
        "predicted_label": ["Supported", "Unsupported", "Uncertain", "Unsupported",
                            "Supported", "Supported", "Supported", "Unsupported"],
        "reason": [""] * 8,
        "p_unsupported": [0.1, 0.9, 0.5, 0.8, 0.2, 0.3, 0.1, 0.7],
    })


class TestInsertedValidation(unittest.TestCase):

    def test_test_split_is_refused(self):
        with self.assertRaises(ValueError):
            iv.run(splits=("test",))

    def test_summary_metrics(self):
        s = iv.summarise(toy_results())

        self.assertAlmostEqual(s["overall"]["strict_recall"], 3 / 4)
        self.assertAlmostEqual(s["overall"]["precision"], 1.0)
        self.assertAlmostEqual(s["storyboard_b"]["strict_recall"], 1 / 2)
        self.assertAlmostEqual(s["reference_flag_everything"]["precision"], 0.5)
        self.assertFalse(s["targets_met"]["storyboard_b_strict_recall>=0.80"])

        scores = s["scores_overall"]
        self.assertIn("ece", scores)
        self.assertIn("precision_at_recall_0.80", scores)

    def test_json_cleaning_removes_nan(self):
        cleaned = iv._clean({"a": float("nan"), "b": [np.float64(1.5), np.int64(2)]})
        self.assertEqual(cleaned, {"a": None, "b": [1.5, 2]})


if __name__ == "__main__":
    unittest.main()

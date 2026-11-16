"""Unit tests for the numeric plus claim text model pipeline."""

import unittest

import numpy as np
import pandas as pd

from claimcheck_classical.config import USE_CLAIM_TEXT
from claimcheck_classical.features.evidence_features import (
    FULL_FEATURE_COLUMNS,
    TEXT_COLUMN,
    with_claim_text,
)
from claimcheck_classical.training.train import (
    coefficient_table,
    model_numeric_columns,
    train_logistic_regression,
)


# ============================================================
# 1. MODEL PIPELINE TESTS
# ============================================================

class TestModelPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Build a small synthetic training set."""

        rng = np.random.default_rng(0)
        n = 200

        numeric = pd.DataFrame(
            rng.random((n, len(FULL_FEATURE_COLUMNS))),
            columns=FULL_FEATURE_COLUMNS,
        )

        # Unsupported claims use different wording.
        y = pd.Series(rng.integers(0, 2, n))

        source = pd.DataFrame({
            TEXT_COLUMN: np.where(
                y == 1,
                "revenue exceeded the limit",
                "calculate the revenue ratio",
            )
        })

        cls.X = with_claim_text(numeric, source)
        cls.y = y

    def test_with_claim_text_adds_column(self):
        self.assertIn(TEXT_COLUMN, self.X.columns)
        self.assertEqual(len(self.X), 200)

    def test_with_claim_text_does_not_modify_input(self):
        numeric = self.X[FULL_FEATURE_COLUMNS]

        with_claim_text(
            numeric,
            pd.DataFrame({TEXT_COLUMN: ["a"] * len(numeric)}),
        )

        self.assertNotIn(TEXT_COLUMN, numeric.columns)

    def test_training_and_probabilities(self):
        model = train_logistic_regression(
            self.X,
            self.y,
            C=1.0,
            ngram_range=(1, 2),
            max_features=1000,
        )

        probabilities = model.predict_proba(self.X)[:, 1]

        self.assertEqual(len(probabilities), 200)
        self.assertTrue(np.all(probabilities >= 0))
        self.assertTrue(np.all(probabilities <= 1))

    def test_coefficient_table_groups(self):
        model = train_logistic_regression(self.X, self.y, C=1.0)

        table = coefficient_table(model)

        numeric = table[table["group"] == "numeric"]

        self.assertEqual(
            numeric["feature"].tolist(),
            model_numeric_columns(),
        )

        if USE_CLAIM_TEXT:
            self.assertTrue((table["group"] == "claim_text").any())


if __name__ == "__main__":
    unittest.main()

"""Unit tests for ClaimCheck classical features."""

import unittest

import numpy as np
import pandas as pd

from claimcheck_classical.features.text_features import (
    create_tfidf_vectorizer,
    compute_cosine_similarity,
)
from claimcheck_classical.features.numerical_features import (
    compute_numerical_consistency,
)
from claimcheck_classical.features.period_features import (
    compute_period_consistency,
)
from claimcheck_classical.features.scope_features import (
    compute_scope_consistency,
    extract_number_scope_pairs,
)
from claimcheck_classical.features.feature_builder import (
    build_feature_matrix,
    FEATURE_COLUMNS,
)


# ============================================================
# 1. TEXT FEATURE TESTS
# ============================================================

class TestTextFeatures(unittest.TestCase):

    def test_cosine_similarity(self):
        claims = [
            "total revenue increased",
            "net income decreased",
        ]

        evidence = [
            "total revenue increased",
            "operating expenses increased",
        ]

        vectorizer = create_tfidf_vectorizer()

        vectorizer.fit(claims + evidence)

        scores = compute_cosine_similarity(
            vectorizer,
            claims,
            evidence,
        )

        self.assertEqual(len(scores), 2)
        self.assertAlmostEqual(scores[0], 1.0)
        self.assertTrue(np.all(scores >= 0.0))
        self.assertTrue(np.all(scores <= 1.0 + 1e-9))


# ============================================================
# 2. NUMERICAL FEATURE TESTS
# ============================================================

class TestNumericalFeatures(unittest.TestCase):

    def test_matching_financial_units(self):
        score = compute_numerical_consistency(
            "Revenue was $1.2 billion",
            "Revenue was $1,200 million",
        )

        self.assertEqual(score, 1.0)

    def test_different_values(self):
        score = compute_numerical_consistency(
            "Revenue was $500 million",
            "Revenue was $600 million",
        )

        self.assertEqual(score, 0.0)

    def test_missing_numbers(self):
        score = compute_numerical_consistency(
            "Revenue increased",
            "Revenue increased",
        )

        self.assertTrue(np.isnan(score))


# ============================================================
# 3. PERIOD FEATURE TESTS
# ============================================================

class TestPeriodFeatures(unittest.TestCase):

    def test_matching_period(self):
        score = compute_period_consistency(
            "Revenue increased in Q1 2024",
            "Revenue increased in first quarter of 2024",
        )

        self.assertEqual(score, 1.0)

    def test_different_year(self):
        score = compute_period_consistency(
            "Revenue increased in Q1 2024",
            "Revenue increased in Q1 2023",
        )

        self.assertEqual(score, 0.0)

    def test_missing_period(self):
        score = compute_period_consistency(
            "Revenue increased",
            "Revenue increased",
        )

        self.assertTrue(np.isnan(score))


# ============================================================
# 4. SCOPE FEATURE TESTS
# ============================================================

class TestScopeFeatures(unittest.TestCase):

    def test_matching_scope(self):
        score = compute_scope_consistency(
            "Total revenue was $20 billion",
            "Total revenue was $20 billion",
        )

        self.assertEqual(score, 1.0)

    def test_scope_mismatch(self):
        score = compute_scope_consistency(
            "Total revenue was $18.6 billion",
            "Total revenue was $20 billion, "
            "while segment revenue was $18.6 billion",
        )

        self.assertEqual(score, 0.0)

    def test_number_scope_association(self):
        pairs = extract_number_scope_pairs(
            "Total revenue was $1,200 million, "
            "while segment revenue was $800 million"
        )

        actual = [
            (item["value"], item["scope"])
            for item in pairs
        ]

        expected = [
            (1_200_000_000.0, "total"),
            (800_000_000.0, "segment"),
        ]

        self.assertEqual(actual, expected)


# ============================================================
# 5. FEATURE BUILDER TESTS
# ============================================================

class TestFeatureBuilder(unittest.TestCase):

    def test_feature_matrix(self):
        df = pd.DataFrame({
            "claim_clean": [
                "total revenue was $20 billion in 2024",
                "total revenue was $18.6 billion in 2024",
                "revenue increased",
            ],
            "evidence_clean": [
                "total revenue was $20 billion in 2024",
                "total revenue was $20 billion in 2024, "
                "while segment revenue was $18.6 billion",
                "revenue increased",
            ],
        })

        vectorizer = create_tfidf_vectorizer()

        vectorizer.fit(
            df["claim_clean"].tolist()
            + df["evidence_clean"].tolist()
        )

        features = build_feature_matrix(df, vectorizer)

        # Check matrix dimensions and column order.
        self.assertEqual(features.shape, (3, 7))
        self.assertEqual(
            features.columns.tolist(),
            FEATURE_COLUMNS,
        )

        # Logistic Regression cannot accept NaN values.
        self.assertTrue(
            np.isfinite(features.to_numpy()).all()
        )

        # Matching financial information.
        self.assertEqual(
            features.iloc[0]["numerical_consistency"],
            1.0,
        )
        self.assertEqual(
            features.iloc[0]["period_consistency"],
            1.0,
        )
        self.assertEqual(
            features.iloc[0]["scope_consistency"],
            1.0,
        )

        # Storyboard B: matching number, wrong scope.
        self.assertEqual(
            features.iloc[1]["numerical_consistency"],
            1.0,
        )
        self.assertEqual(
            features.iloc[1]["scope_consistency"],
            0.0,
        )

        # Missing information must be preserved.
        self.assertEqual(
            features.iloc[2]["numerical_missing"],
            1,
        )
        self.assertEqual(
            features.iloc[2]["period_missing"],
            1,
        )
        self.assertEqual(
            features.iloc[2]["scope_missing"],
            1,
        )


if __name__ == "__main__":
    unittest.main()


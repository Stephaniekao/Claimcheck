"""Integration tests using real RAGBench training data."""

import unittest

import numpy as np
import pandas as pd

from claimcheck_classical.data_loader import (
    load_ragbench_claims,
)
from claimcheck_classical.preprocessing import (
    preprocess_claims,
)
from claimcheck_classical.features.text_features import (
    create_tfidf_vectorizer,
)
from claimcheck_classical.features.feature_builder import (
    build_feature_matrix,
    FEATURE_COLUMNS,
)


# ============================================================
# 1. TEST CONFIGURATION
# ============================================================

SUBSET = "FinQA"
SPLIT = "train"
SAMPLE_SIZE = 100


# ============================================================
# 2. RAGBENCH INTEGRATION TESTS
# ============================================================

class TestRAGBenchIntegration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Load and prepare a small training sample."""

        # Load the original FinQA training split.
        cls.full_data = load_ragbench_claims(
            subset=SUBSET,
            split=SPLIT,
        )

        # Select a deterministic sample for the smoke test.
        cls.sample = (
            cls.full_data
            .head(SAMPLE_SIZE)
            .copy()
        )

        # Normalize the claim and evidence text.
        cls.processed = preprocess_claims(
            cls.sample
        )

        # Create a TF-IDF vectorizer.
        cls.vectorizer = create_tfidf_vectorizer()

        # Fit using training text only.
        training_text = (
            cls.processed["claim_clean"].tolist()
            + cls.processed["evidence_clean"].tolist()
        )

        cls.vectorizer.fit(training_text)

        # Build the final feature matrix.
        cls.features = build_feature_matrix(
            cls.processed,
            cls.vectorizer,
        )

    def test_data_loading(self):
        """Check that the requested training data exists."""

        self.assertEqual(len(self.sample), SAMPLE_SIZE)

        self.assertTrue(
            self.sample["split"].eq("train").all()
        )

        self.assertTrue(
            self.sample["subset"].eq(SUBSET).all()
        )

    def test_preprocessing(self):
        """Check that normalized text columns are created."""

        self.assertIn(
            "claim_clean",
            self.processed.columns,
        )

        self.assertIn(
            "evidence_clean",
            self.processed.columns,
        )

        self.assertEqual(
            len(self.processed),
            SAMPLE_SIZE,
        )

    def test_feature_matrix_shape(self):
        """Check the number of rows and feature columns."""

        self.assertEqual(
            self.features.shape,
            (SAMPLE_SIZE, len(FEATURE_COLUMNS)),
        )

        self.assertEqual(
            self.features.columns.tolist(),
            FEATURE_COLUMNS,
        )

    def test_feature_values(self):
        """Check that the matrix contains valid numeric values."""

        values = self.features.to_numpy()

        self.assertTrue(
            np.issubdtype(values.dtype, np.number)
        )

        self.assertTrue(
            np.isfinite(values).all()
        )

    def test_cosine_similarity_range(self):
        """Check TF-IDF cosine similarity values."""

        scores = self.features["cosine_similarity"]

        self.assertTrue(
            scores.ge(-1e-9).all()
        )

        self.assertTrue(
            scores.le(1.0 + 1e-9).all()
        )

    def test_missing_indicators(self):
        """Check that missing indicators are binary."""

        columns = [
            "numerical_missing",
            "period_missing",
            "scope_missing",
        ]

        for column in columns:
            self.assertTrue(
                self.features[column].isin([0, 1]).all()
            )

    def test_original_data_preserved(self):
        """Check that preprocessing does not modify original text."""

        pd.testing.assert_series_equal(
            self.sample["claim"],
            self.processed["claim"],
        )

        pd.testing.assert_series_equal(
            self.sample["evidence"],
            self.processed["evidence"],
        )


if __name__ == "__main__":
    unittest.main()

"""Regression tests: sentence-final years and 13-feature builder."""

import unittest

import numpy as np
import pandas as pd

from claimcheck_classical.preprocessing import preprocess_claims
from claimcheck_classical.features.text_features import (
    create_tfidf_vectorizer,
)
from claimcheck_classical.features.feature_builder import (
    build_feature_matrix,
    FEATURE_COLUMNS,
)
from claimcheck_classical.features.numerical_features import (
    compute_numerical_consistency,
)
from claimcheck_classical.features.period_features import (
    compute_period_consistency,
    extract_reporting_periods,
)
from claimcheck_classical.features.evidence_features import (
    FULL_FEATURE_COLUMNS,
    build_full_feature_matrix,
    fuzzy_number_match,
)


class TestSentenceFinalYear(unittest.TestCase):

    def test_year_before_period_is_detected(self):
        self.assertEqual(
            extract_reporting_periods("revenue was 5 million in 2019."),
            [(2019, None)],
        )

    def test_decimal_is_not_a_year(self):
        self.assertEqual(extract_reporting_periods("in 2019.5 units"), [])

    def test_claim_ending_with_year_is_consistent(self):
        claim = "revenue was $5 million in 2019."
        evidence = "revenue was $5 million in 2019, up from last year."

        self.assertEqual(
            compute_numerical_consistency(claim, evidence), 1.0
        )
        self.assertEqual(
            compute_period_consistency(claim, evidence), 1.0
        )


class TestFullFeatures(unittest.TestCase):

    def test_fuzzy_number_match_handles_scale_and_rounding(self):
        score, count = fuzzy_number_match(
            "revenue was $1.2 billion.",
            "revenue 1,205 total",
        )
        self.assertEqual(count, 1)
        self.assertEqual(score, 1.0)

    def test_full_matrix_shape_and_base_columns_unchanged(self):
        evidence = (
            "Total revenue was $20 billion in 2024.\n"
            "Net income rose to 55 million."
        )
        df = pd.DataFrame({
            "claim": ["Total revenue was $20 billion in 2024.",
                      "Net income was $99 million."],
            "evidence": [evidence, evidence],
        })
        processed = preprocess_claims(df)

        vectorizer = create_tfidf_vectorizer()
        vectorizer.fit(
            processed["claim_clean"].tolist()
            + processed["evidence_clean"].tolist()
        )

        full = build_full_feature_matrix(processed, vectorizer)
        base = build_feature_matrix(processed, vectorizer)

        self.assertEqual(full.columns.tolist(), FULL_FEATURE_COLUMNS)
        self.assertEqual(full.shape, (2, 13))
        self.assertTrue(np.isfinite(full.to_numpy()).all())
        pd.testing.assert_frame_equal(
            full[FEATURE_COLUMNS], base, check_dtype=False
        )


if __name__ == "__main__":
    unittest.main()

"""Unit tests for financial unit parsing."""

import unittest

import numpy as np

from claimcheck_classical.features.numerical_features import (
    compute_numerical_consistency,
    extract_numbers,
)


# ============================================================
# 1. UNIT PARSING TESTS
# ============================================================

class TestUnitParsing(unittest.TestCase):

    def test_words_and_abbreviations_give_same_value(self):
        word = extract_numbers("$4.1 billion")
        short = extract_numbers("$4.1b")
        bn = extract_numbers("4.1bn")

        self.assertEqual(len(short), 1)
        self.assertTrue(np.isclose(word[0], short[0]))
        self.assertTrue(np.isclose(word[0], bn[0]))

    def test_million_and_thousand_abbreviations(self):
        self.assertEqual(extract_numbers("$2.5m"), [2_500_000.0])
        self.assertEqual(extract_numbers("1.2mm"), [1_200_000.0])
        self.assertEqual(extract_numbers("$300k"), [300_000.0])

    def test_company_name_is_not_a_number(self):
        self.assertEqual(extract_numbers("3m reported"), [])

    def test_existing_units_unchanged(self):
        self.assertEqual(extract_numbers("199million"), [199_000_000.0])
        self.assertEqual(extract_numbers("4,100 million"), [4_100_000_000.0])
        self.assertEqual(extract_numbers("(5,860)"), [-5860.0])

    def test_abbreviation_matches_spelled_out_evidence(self):
        score = compute_numerical_consistency(
            "revenue was $4.1b",
            "revenue was 4,100 million",
        )

        self.assertEqual(score, 1.0)


if __name__ == "__main__":
    unittest.main()

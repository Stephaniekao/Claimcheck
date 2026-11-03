"""python -m unittest discover Baseline/tests"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from claimcheck_baseline.normalize import extract_periods, extract_quantities, quantities_match  # noqa: E402
from claimcheck_baseline.rules import verify  # noqa: E402


def q(text):
    out = extract_quantities(text)
    assert len(out) == 1, (text, out)
    return out[0]


class Normalizer(unittest.TestCase):
    def test_card_example_forms_are_equal(self):  # Field 3: "$4.1 billion", "4,100 million", "4.1B"
        a, b, c = q("$4.1 billion"), q("4,100 million"), q("4.1B")
        self.assertTrue(quantities_match(a, b) and quantities_match(b, c) and quantities_match(c, a))

    def test_rounded_claim_matches_exact_filing_value(self):
        self.assertTrue(quantities_match(q("$4.1 billion"), q("$4,123 million")))
        self.assertFalse(quantities_match(q("$4.1 billion"), q("$4,200 million")))

    def test_unit_error_is_not_a_match(self):
        self.assertFalse(quantities_match(q("USD 24,365.000 billion"), q("USD 24,365 million")))

    def test_bare_table_number_uses_implied_scale(self):
        self.assertTrue(quantities_match(q("$964 million"), q("964")))

    def test_accounting_negative_and_sign(self):
        self.assertEqual(q("(1,234)").value, -1234)
        self.assertEqual(q("USD -9,315.200 million").value, -9_315_200_000)

    def test_percent_never_matches_money(self):
        self.assertFalse(quantities_match(q("14%"), q("$14")))

    def test_years_dates_identifiers_and_markers_are_not_amounts(self):
        for text in ["in 2019", "2025-12-31", "december 31 , 2012", "Form 10-K", "accession 0000731766-26-000062",
                     "( 1 ) note", "dec-13", "2014pullmantur"]:
            self.assertEqual(extract_quantities(text), [], text)

    def test_periods(self):
        self.assertEqual(extract_periods("for the fiscal period ended 2025-12-31"), {"2025-12-31", "2025"})
        self.assertIn("2012", extract_periods("at december 31 , 2012"))


PASSAGE = """ACME Form 10-K, selected XBRL financial data:
Revenues for the fiscal year ended 2025-12-31: USD 4,100 million.
Revenues for the fiscal year ended 2024-12-31: USD 3,900 million.
Assets, Current as of 2025-12-31: USD 1,200 million.
Net Income (Loss) Attributable to Parent for the fiscal year ended 2025-12-31: USD 300 million."""

SEGMENTS = """ACME revenue by segment for the fiscal year ended 2025-12-31:
CloudMember segment revenue for the fiscal year ended 2025-12-31: USD 2,000,000,000.
Total revenues for the fiscal year ended 2025-12-31: USD 4,100,000,000."""


class Rules(unittest.TestCase):
    def label(self, claim, passage=PASSAGE):
        return verify(claim, passage).label

    def test_supported(self):
        self.assertEqual(self.label("ACME's revenue for the fiscal period ended 2025-12-31 was $4.1 billion."), "Supported")

    def test_numerical(self):
        self.assertEqual(self.label("ACME's revenue for the fiscal period ended 2025-12-31 was USD 4,920 million."), "Unsupported")

    def test_temporal(self):
        self.assertEqual(self.label("ACME's revenue for the fiscal period ended 2024-12-31 was USD 4,100 million."), "Unsupported")

    def test_metric_confusion(self):
        self.assertEqual(self.label("ACME's current assets for the fiscal period ended 2025-12-31 was USD 300 million."), "Unsupported")

    def test_unit(self):
        self.assertEqual(self.label("ACME's current assets for the fiscal period ended 2025-12-31 was USD 1,200 billion."), "Unsupported")

    def test_storyboard_b_scope_error(self):
        claim = "ACME's CloudMember revenue for the fiscal period ended 2025-12-31 represented the company's entire revenue of USD 2,000,000,000."
        self.assertEqual(self.label(claim, SEGMENTS), "Unsupported")

    def test_storyboard_b_scope_error_without_total_line_is_uncertain(self):
        claim = "ACME's CloudMember revenue for the fiscal period ended 2025-12-31 represented the company's entire revenue of USD 2,000,000,000."
        no_total = "\n".join(SEGMENTS.splitlines()[:2])
        self.assertEqual(self.label(claim, no_total), "Uncertain")

    def test_storyboard_b_correct_segment_claim(self):
        claim = "ACME's CloudMember segment revenue for the fiscal period ended 2025-12-31 was USD 2,000,000,000."
        self.assertEqual(self.label(claim, SEGMENTS), "Supported")

    def test_uncheckable_claims_are_uncertain(self):
        self.assertEqual(self.label("Revenue grew because of strong cloud demand."), "Uncertain")
        self.assertEqual(verify("Revenue was $4.1 billion.", "").label, "Uncertain")

    def test_table_cells_carry_their_column_period(self):
        table = 'Revenue table [["", "2019", "2018"], ["total revenue", "$ 5,000", "$ 4,000"]]'
        self.assertEqual(verify("Total revenue in 2019 was $5,000.", table).label, "Supported")
        self.assertEqual(verify("Total revenue in 2018 was $5,000.", table).label, "Unsupported")


if __name__ == "__main__":
    unittest.main()

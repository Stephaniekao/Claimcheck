"""python -m unittest discover ClaimSplitter/tests"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from claimcheck_splitter.split import split  # noqa: E402


def texts(answer):
    return [c.text for c in split(answer)]


class Splitter(unittest.TestCase):
    def test_one_claim_per_sentence(self):
        self.assertEqual(texts("Revenue was $52.4 billion. Net income was $8.1 billion."),
                         ["Revenue was $52.4 billion.", "Net income was $8.1 billion."])

    def test_decimals_and_amounts_do_not_split(self):
        self.assertEqual(texts("Margin rose to 4.1% in FY2023, from 3.9%."), ["Margin rose to 4.1% in FY2023, from 3.9%."])

    def test_company_and_country_abbreviations_do_not_split(self):
        self.assertEqual(len(split("Apple Inc. reported $383.3 billion in U.S. Dollars. It grew 2%.")), 2)

    def test_sentence_starting_with_a_number_or_dollar(self):
        self.assertEqual(texts("Revenue grew. $4.1B came from Cloud. 2023 was strong."),
                         ["Revenue grew.", "$4.1B came from Cloud.", "2023 was strong."])

    def test_list_items_are_separate_claims_without_markers(self):
        self.assertEqual(texts("Results:\n- Revenue was $10M\n- Net income was $2M\n1. Cash was $1M"),
                         ["Results:", "Revenue was $10M", "Net income was $2M", "Cash was $1M"])

    def test_offsets_point_back_into_the_answer(self):
        answer = "Intro line.\n  - Revenue was $10M.  Profit was $2M."
        for c in split(answer):
            self.assertEqual(answer[c.start:c.end], c.text)

    def test_lowercase_after_period_does_not_split(self):
        self.assertEqual(len(split("Revenue was approx. $5 billion and costs were lower.")), 1)

    def test_empty_answer(self):
        self.assertEqual(split("   \n  "), [])


if __name__ == "__main__":
    unittest.main()

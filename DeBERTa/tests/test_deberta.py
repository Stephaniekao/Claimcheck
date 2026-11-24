"""python -m unittest discover DeBERTa/tests   (no torch, no GPU needed)"""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from claimcheck_deberta import calibration as cal  # noqa: E402
from claimcheck_deberta import data, report  # noqa: E402
from claimcheck_deberta.windows import pack  # noqa: E402


class Windows(unittest.TestCase):
    def test_keeps_order_and_every_sentence(self):
        ws = pack([5, 5, 5, 5, 5], budget=10)
        self.assertEqual(ws, [[0, 1], [2, 3], [4]])

    def test_long_sentence_gets_its_own_window(self):
        self.assertEqual(pack([3, 50, 3], budget=10), [[0], [1], [2]])

    def test_empty_passage_gives_one_empty_window(self):
        self.assertEqual(pack([], budget=10), [[]])


class Labels(unittest.TestCase):
    def test_two_thresholds_give_three_labels(self):
        self.assertEqual(list(cal.to_labels(np.array([0.1, 0.5, 0.9]), 0.3, 0.7)),
                         ["Unsupported", "Uncertain", "Supported"])

    def test_thresholds_keep_uncertain_under_a_third(self):
        rng = np.random.default_rng(0)
        gold = pd.Series(["Unsupported"] * 100 + ["Supported"] * 100)
        score = np.concatenate([rng.uniform(0, 0.6, 100), rng.uniform(0.4, 1, 100)])
        th = cal.choose_thresholds(gold, score)
        pred = cal.to_labels(score, th["lower"], th["upper"])
        self.assertLessEqual(pred.eq("Uncertain").mean(), 1 / 3 + 1e-9)
        self.assertTrue(th["precision_target_met_on_validation"])
        self.assertGreaterEqual(th["validation"]["precision"], 0.60)

    def test_falls_back_and_says_so_when_precision_is_out_of_reach(self):
        gold = pd.Series(["Unsupported", "Supported"] * 50)
        score = np.full(100, 0.5)  # no signal at all
        th = cal.choose_thresholds(gold, score)
        self.assertFalse(th["precision_target_met_on_validation"])


class Calibration(unittest.TestCase):
    def test_ece_is_zero_when_scores_match_outcomes(self):
        gold = pd.Series(["Supported"] * 8 + ["Unsupported"] * 2)
        self.assertAlmostEqual(cal.ece(gold, np.full(10, 0.8)), 0.0)

    def test_ece_of_overconfident_scores(self):
        gold = pd.Series(["Supported"] * 5 + ["Unsupported"] * 5)
        self.assertAlmostEqual(cal.ece(gold, np.full(10, 0.95)), 0.45)

    def test_temperature_keeps_ranking(self):
        s = np.array([0.1, 0.4, 0.6, 0.99])
        t = cal.apply_temperature(s, 2.0)
        self.assertTrue(np.all(np.diff(t) > 0))

    def test_temperature_lowers_overconfidence(self):
        rng = np.random.default_rng(1)
        y = rng.random(2000) < 0.7
        gold = pd.Series(np.where(y, "Supported", "Unsupported"))
        score = np.where(y, 0.999, 0.6)  # far too sure
        t = cal.fit_temperature(gold, score)
        self.assertGreater(t, 1.0)
        self.assertLess(cal.ece(gold, cal.apply_temperature(score, t)), cal.ece(gold, score))


class Data(unittest.TestCase):
    def test_inserted_error_set_uses_the_baseline_passages(self):
        claims, passages = data.load_inserted_error_set()
        self.assertEqual(len(claims), 400)
        self.assertNotIn("evidence", claims.columns)  # the passage is only reachable through `answer`
        self.assertTrue(all(len(p["sentences"]) == len(p["keys"]) for p in passages))

    def test_answer_roll_up(self):
        claims = pd.DataFrame({"answer": [0, 0, 1], "answer_label": ["Unsupported", "Unsupported", "Supported"]})
        a = report.answer_scores(claims, np.array([0.9, 0.2, 0.8]), "lowest")
        self.assertEqual(list(a.score), [0.2, 0.8])


if __name__ == "__main__":
    unittest.main()

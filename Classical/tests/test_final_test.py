"""Tests for the seed summary and the one-time final test."""

import json
import tempfile
import unittest
from pathlib import Path

from claimcheck_classical.evaluation import final_test
from claimcheck_classical.evaluation.seed_summary import run_name


class TestSeedNames(unittest.TestCase):

    def test_main_seed_keeps_old_names(self):
        self.assertEqual(run_name(42), "full_best")
        self.assertEqual(run_name(43), "full_seed43_best")


class TestFinalTestLock(unittest.TestCase):

    def test_second_run_is_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            old = final_test.LOCK_PATH
            final_test.LOCK_PATH = Path(folder) / "final_test_lock.json"

            try:
                final_test.check_lock()  # no lock yet: allowed

                final_test.LOCK_PATH.write_text(
                    json.dumps({"status": "finished", "started_utc": "x"})
                )

                with self.assertRaises(SystemExit):
                    final_test.check_lock()
            finally:
                final_test.LOCK_PATH = old


class TestField10(unittest.TestCase):

    def test_targets(self):
        ragbench = {"answer_auroc": {"overall": 0.80}, "ece": 0.05}
        inserted = {
            "overall": {"strict_recall": 0.9, "precision": 0.7},
            "storyboard_b": {"strict_recall": 0.85},
            "scores_overall": {"ece": 0.20},
        }

        checks = final_test.field_10(ragbench, inserted)

        self.assertTrue(checks["ragbench_answer_auroc>=0.75"])
        self.assertTrue(checks["inserted_strict_recall>=0.80"])
        self.assertFalse(checks["inserted_ece<=0.10"])
        self.assertFalse(checks["meets_all_targets"])

    def test_missing_precision_is_not_met(self):
        ragbench = {"answer_auroc": {"overall": 0.6}, "ece": 0.01}
        inserted = {
            "overall": {"strict_recall": 0.0, "precision": float("nan")},
            "storyboard_b": {"strict_recall": 0.0},
            "scores_overall": {"ece": 0.46},
        }

        checks = final_test.field_10(ragbench, inserted)

        self.assertFalse(checks["inserted_precision>=0.60"])


if __name__ == "__main__":
    unittest.main()

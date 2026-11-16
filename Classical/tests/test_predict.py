"""Unit tests for three-label inference."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from claimcheck_classical.model_io import save_model
from claimcheck_classical.preprocessing import preprocess_claims
from claimcheck_classical.features.text_features import (
    create_tfidf_vectorizer,
)
from claimcheck_classical.features.evidence_features import (
    FULL_FEATURE_COLUMNS,
    build_full_feature_matrix,
    with_claim_text,
)
from claimcheck_classical.training.train import (
    train_logistic_regression,
)
from claimcheck_classical.inference.predict import (
    NO_PASSAGE_REASON,
    NO_THRESHOLD_REASON,
    predict_claims,
    predict_single_claim,
    predict_with_frozen_model,
)


EVIDENCE = (
    "Total revenue was $20 billion in 2024.\n"
    "Net income rose to $55 million in 2024."
)


# ============================================================
# 1. INFERENCE TESTS
# ============================================================

class TestPredict(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Train a small model on synthetic claims."""

        claims, labels = [], []

        for index in range(40):
            if index % 2 == 0:
                claims.append("Total revenue was $20 billion in 2024.")
                labels.append(0)
            else:
                claims.append(f"Total revenue was ${30 + index} billion in 2024.")
                labels.append(1)

        df = pd.DataFrame({
            "claim": claims,
            "evidence": [EVIDENCE] * len(claims),
        })

        processed = preprocess_claims(df)

        cls.vectorizer = create_tfidf_vectorizer()
        cls.vectorizer.fit(
            processed["claim_clean"].tolist()
            + processed["evidence_clean"].tolist()
        )

        X = build_full_feature_matrix(processed, cls.vectorizer)
        X = with_claim_text(X, processed)

        cls.model = train_logistic_regression(X, pd.Series(labels))

    def test_thresholds_are_required(self):
        df = pd.DataFrame({"claim": ["Revenue was $20 billion."], "evidence": [EVIDENCE]})

        with self.assertRaises(ValueError):
            predict_claims(df, self.model, self.vectorizer)

    def test_output_has_labels_probabilities_and_features(self):
        df = pd.DataFrame({
            "claim": ["Total revenue was $20 billion in 2024."],
            "evidence": [EVIDENCE],
        })

        result = predict_claims(
            df, self.model, self.vectorizer,
            lower_threshold=0.3, upper_threshold=0.7,
        )

        self.assertIn(
            result.loc[0, "predicted_label"],
            ("Supported", "Uncertain", "Unsupported"),
        )
        self.assertTrue(np.isfinite(result.loc[0, "p_supported"]))

        for column in FULL_FEATURE_COLUMNS:
            self.assertIn(column, result.columns)

    def test_missing_passage_is_uncertain(self):
        df = pd.DataFrame({
            "claim": [
                "Total revenue was $20 billion in 2024.",
                "Total revenue was $20 billion in 2024.",
                "Total revenue was $20 billion in 2024.",
            ],
            "evidence": [EVIDENCE, "", None],
        })

        result = predict_claims(
            df, self.model, self.vectorizer,
            lower_threshold=0.3, upper_threshold=0.7,
        )

        self.assertEqual(len(result), 3)

        for index in (1, 2):
            self.assertEqual(result.loc[index, "predicted_label"], "Uncertain")
            self.assertEqual(result.loc[index, "reason"], NO_PASSAGE_REASON)
            self.assertTrue(np.isnan(result.loc[index, "p_supported"]))

        self.assertTrue(np.isfinite(result.loc[0, "p_supported"]))

    def test_all_uncertain_fallback_keeps_probabilities(self):
        df = pd.DataFrame({
            "claim": ["Total revenue was $20 billion in 2024."] * 2,
            "evidence": [EVIDENCE, EVIDENCE],
        })

        result = predict_claims(
            df, self.model, self.vectorizer, all_uncertain=True,
        )

        self.assertTrue((result["predicted_label"] == "Uncertain").all())
        self.assertTrue((result["reason"] == NO_THRESHOLD_REASON).all())
        self.assertTrue(np.isfinite(result["p_supported"]).all())

    def test_single_claim_with_empty_evidence(self):
        result = predict_single_claim(
            "Total revenue was $20 billion in 2024.", "",
            self.model, self.vectorizer,
            lower_threshold=0.3, upper_threshold=0.7,
        )

        self.assertEqual(result["predicted_label"], "Uncertain")

    def test_frozen_model_round_trip(self):
        df = pd.DataFrame({
            "claim": [
                "Total revenue was $20 billion in 2024.",
                "Total revenue was $99 billion in 2024.",
            ],
            "evidence": [EVIDENCE, EVIDENCE],
        })

        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)

            save_model(self.model, self.vectorizer, folder / "model.joblib")

            metadata = {
                "model_file": "model.joblib",
                "thresholds": {
                    "feasible": True,
                    "lower_threshold": 0.3,
                    "upper_threshold": 0.7,
                },
            }

            (folder / "model.json").write_text(json.dumps(metadata))

            frozen = predict_with_frozen_model(df, folder / "model.json")

            metadata["thresholds"]["feasible"] = False
            (folder / "model.json").write_text(json.dumps(metadata))

            fallback = predict_with_frozen_model(df, folder / "model.json")

        direct = predict_claims(
            df, self.model, self.vectorizer,
            lower_threshold=0.3, upper_threshold=0.7,
        )

        self.assertEqual(
            frozen["predicted_label"].tolist(),
            direct["predicted_label"].tolist(),
        )
        self.assertTrue((fallback["predicted_label"] == "Uncertain").all())


if __name__ == "__main__":
    unittest.main()

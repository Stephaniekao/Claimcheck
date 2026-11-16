"""
ClaimCheck Classical Model - Feature Set Ablation

Compare feature sets with the same Logistic Regression settings:
1. Claim length only
2. Original seven features
3. All thirteen features
4. Claim TF-IDF text only
5. Thirteen features plus claim TF-IDF text

Models train on official training claims (all Unsupported claims and
a sample of Supported claims) and are scored on the full official
validation splits. The test splits are never loaded.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from claimcheck_classical.data_loader import load_ragbench_claims
from claimcheck_classical.preprocessing import preprocess_claims
from claimcheck_classical.features.text_features import (
    create_tfidf_vectorizer,
)
from claimcheck_classical.features.evidence_features import (
    BASE_COLUMNS,
    FULL_FEATURE_COLUMNS,
    aggregate_answer_scores,
    build_full_feature_matrix,
    build_static_features,
)
from claimcheck_classical.evaluation.feature_diagnostics import (
    SUBSETS,
    load_sample,
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_DIR / "outputs"

DEFAULT_SUPPORTED_SAMPLE = 20000
DEFAULT_C = 1.0
DEFAULT_SEED = 42

# Claim text TF-IDF settings.
TEXT_NGRAM_RANGE = (1, 2)
TEXT_MAX_FEATURES = 20000
TEXT_MIN_DF = 2

# Alphabetic words only, so the text model learns wording
# rather than memorizing individual numbers or years.
TEXT_TOKEN_PATTERN = r"(?u)\b[a-z][a-z]+\b"

TOP_NGRAMS = 15


# ============================================================
# 2. DATA
# ============================================================

def load_validation() -> pd.DataFrame:
    """Load the official validation splits."""

    frames = [
        load_ragbench_claims(subset=subset, split="validation")
        for subset in SUBSETS
    ]

    return pd.concat(frames, ignore_index=True)


def to_binary(processed: pd.DataFrame) -> np.ndarray:
    """Map Unsupported to 1 and Supported to 0."""

    return (
        processed["label"] == "Unsupported"
    ).astype(int).to_numpy()


# ============================================================
# 3. EVALUATION
# ============================================================

def evaluate(
    name: str,
    validation: pd.DataFrame,
    y_val: np.ndarray,
    p_unsupported: np.ndarray,
) -> dict:
    """Compute claim-level and answer-level validation metrics."""

    row = {
        "feature_set": name,
        "roc_auc": float(roc_auc_score(y_val, p_unsupported)),
        "pr_auc": float(average_precision_score(y_val, p_unsupported)),
    }

    # Answer score uses the lowest claim support probability.
    answers = aggregate_answer_scores(
        validation,
        1.0 - p_unsupported,
    )

    def auc(frame: pd.DataFrame) -> float:
        if frame["y_true"].nunique() < 2:
            return np.nan

        return float(
            roc_auc_score(frame["y_true"], frame["answer_score"])
        )

    row["answer_auroc"] = auc(answers)

    for subset in SUBSETS:
        row[f"answer_auroc_{subset}"] = auc(
            answers[answers["subset"] == subset]
        )

    return row


def fit_logistic(X, y: np.ndarray, C: float, seed: int):
    """Fit an L2 Logistic Regression."""

    model = LogisticRegression(
        solver="liblinear",
        C=C,
        max_iter=2000,
        random_state=seed,
    )

    model.fit(X, y)

    return model


# ============================================================
# 4. TEXT COEFFICIENTS
# ============================================================

def print_top_ngrams(text_vectorizer, model) -> None:
    """Show claim n-grams most tied to each label."""

    names = np.asarray(text_vectorizer.get_feature_names_out())
    coefficients = model.coef_[0]
    order = np.argsort(coefficients)

    print("\n========== CLAIM N-GRAMS: TOWARD UNSUPPORTED ==========")

    for index in order[::-1][:TOP_NGRAMS]:
        print(f"{coefficients[index]:+.3f}  {names[index]}")

    print("\n========== CLAIM N-GRAMS: TOWARD SUPPORTED ==========")

    for index in order[:TOP_NGRAMS]:
        print(f"{coefficients[index]:+.3f}  {names[index]}")


# ============================================================
# 5. COMMAND LINE INTERFACE
# ============================================================

def main() -> None:
    """Run the feature set ablation."""

    parser = argparse.ArgumentParser(
        description="Compare ClaimCheck feature sets."
    )

    parser.add_argument(
        "--supported-sample",
        type=int,
        default=DEFAULT_SUPPORTED_SAMPLE,
        help="Supported training claims sampled per subset.",
    )

    parser.add_argument("--C", type=float, default=DEFAULT_C)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)

    args = parser.parse_args()

    print("========== LOADING DATA ==========")

    train_df = load_sample(
        supported_sample=args.supported_sample,
        seed=args.seed,
    )

    validation_df = load_validation()

    train = preprocess_claims(train_df)
    validation = preprocess_claims(validation_df)

    y_train = to_binary(train)
    y_val = to_binary(validation)

    print(f"Training claims: {len(train)} ({int(y_train.sum())} Unsupported)")
    print(
        f"Validation claims: {len(validation)} "
        f"({int(y_val.sum())} Unsupported)"
    )

    print("\n========== BUILDING FEATURES ==========")

    # Fit the claim-evidence TF-IDF on training text only.
    vectorizer = create_tfidf_vectorizer()

    vectorizer.fit(
        train["claim_clean"].tolist()
        + train["evidence_clean"].tolist()
    )

    X_train = build_full_feature_matrix(
        train,
        vectorizer,
        static=build_static_features(train),
    )

    X_val = build_full_feature_matrix(
        validation,
        vectorizer,
        static=build_static_features(validation),
    )

    # Fit a separate TF-IDF on training claim text only.
    text_vectorizer = TfidfVectorizer(
        ngram_range=TEXT_NGRAM_RANGE,
        max_features=TEXT_MAX_FEATURES,
        min_df=TEXT_MIN_DF,
        token_pattern=TEXT_TOKEN_PATTERN,
        lowercase=False,
        norm="l2",
    )

    T_train = text_vectorizer.fit_transform(train["claim_clean"])
    T_val = text_vectorizer.transform(validation["claim_clean"])

    print("\n========== TRAINING AND EVALUATING ==========")

    rows = []

    # Dense feature sets are standardized before Logistic Regression.
    dense_sets = {
        "length_only": ["claim_length"],
        "original_7": BASE_COLUMNS,
        "all_13": FULL_FEATURE_COLUMNS,
    }

    scalers = {}

    for name, columns in dense_sets.items():

        scaler = StandardScaler().fit(X_train[columns])
        scalers[name] = scaler

        model = fit_logistic(
            scaler.transform(X_train[columns]),
            y_train,
            args.C,
            args.seed,
        )

        p_val = model.predict_proba(
            scaler.transform(X_val[columns])
        )[:, 1]

        rows.append(evaluate(name, validation, y_val, p_val))

    # Claim text only.
    text_model = fit_logistic(T_train, y_train, args.C, args.seed)

    rows.append(
        evaluate(
            "claim_text_only",
            validation,
            y_val,
            text_model.predict_proba(T_val)[:, 1],
        )
    )

    # Thirteen dense features plus claim text.
    scaler = scalers["all_13"]

    combined_train = hstack([
        csr_matrix(scaler.transform(X_train[FULL_FEATURE_COLUMNS])),
        T_train,
    ]).tocsr()

    combined_val = hstack([
        csr_matrix(scaler.transform(X_val[FULL_FEATURE_COLUMNS])),
        T_val,
    ]).tocsr()

    combined_model = fit_logistic(
        combined_train,
        y_train,
        args.C,
        args.seed,
    )

    rows.append(
        evaluate(
            "all_13_plus_claim_text",
            validation,
            y_val,
            combined_model.predict_proba(combined_val)[:, 1],
        )
    )

    results = pd.DataFrame(rows)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "feature_ablation.csv"
    results.to_csv(output_path, index=False)

    print("\n========== ABLATION RESULTS (VALIDATION) ==========")
    print(results.round(4).to_string(index=False))
    print(f"\nSaved: {output_path}")

    print_top_ngrams(text_vectorizer, text_model)


if __name__ == "__main__":
    main()

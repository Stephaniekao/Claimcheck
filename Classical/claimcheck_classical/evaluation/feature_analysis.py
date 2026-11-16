"""Analyze feature behavior on real RAGBench training claims."""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from claimcheck_classical.data_loader import load_ragbench_claims
from claimcheck_classical.preprocessing import preprocess_claims
from claimcheck_classical.features.text_features import (
    create_tfidf_vectorizer,
)
from claimcheck_classical.features.feature_builder import (
    build_feature_matrix,
    FEATURE_COLUMNS,
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

DEFAULT_SUBSET = "FinQA"
DEFAULT_SAMPLE_SIZE = 100
DEFAULT_EXAMPLES = 5
MAX_EVIDENCE_CHARS = 1200


# ============================================================
# 2. PREPARE DATA AND FEATURES
# ============================================================

def prepare_features(
    subset: str = DEFAULT_SUBSET,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
) -> pd.DataFrame:
    """Load training claims and calculate their features."""

    if sample_size <= 0:
        raise ValueError("Sample size must be positive.")

    # Load only the original training split.
    df = load_ragbench_claims(
        subset=subset,
        split="train",
    )

    # Select a deterministic sample.
    df = df.head(sample_size).copy()

    if df.empty:
        raise ValueError("No training claims were loaded.")

    # Normalize claim and evidence text.
    processed = preprocess_claims(df)

    # Fit TF-IDF using training data only.
    vectorizer = create_tfidf_vectorizer()

    training_text = (
        processed["claim_clean"].tolist()
        + processed["evidence_clean"].tolist()
    )

    vectorizer.fit(training_text)

    # Calculate all seven features.
    features = build_feature_matrix(
        processed,
        vectorizer,
    )

    # Preserve the original text and labels for analysis.
    result = processed.reset_index(drop=True).copy()

    features = features.reset_index(drop=True)

    for column in FEATURE_COLUMNS:
        result[column] = features[column]

    return result


# ============================================================
# 3. SELECT CASES FOR ERROR ANALYSIS
# ============================================================

def select_numerical_cases(
    df: pd.DataFrame,
    max_examples: int = DEFAULT_EXAMPLES,
) -> pd.DataFrame:
    """Find supported claims with numerical inconsistency."""

    if max_examples <= 0:
        raise ValueError("Number of examples must be positive.")

    # Select cases where the numerical rule returns zero.
    # Exclude missing values because they mean no decision.
    mask = (
        df["label"].eq("Supported")
        & df["numerical_consistency"].eq(0.0)
        & df["numerical_missing"].eq(0)
    )

    return df.loc[mask].head(max_examples).copy()


# ============================================================
# 4. DISPLAY FEATURE SUMMARY
# ============================================================

def print_feature_summary(df: pd.DataFrame) -> None:
    """Print feature statistics and label distribution."""

    print("\n========== DATA SUMMARY ==========")
    print(f"Total claims: {len(df)}")

    print("\nLabel distribution:")
    print(df["label"].value_counts().to_string())

    print("\n========== FEATURE SUMMARY ==========")
    print(
        df[FEATURE_COLUMNS]
        .describe()
        .round(4)
        .to_string()
    )

    print("\n========== MISSING RATES ==========")

    for column in [
        "numerical_missing",
        "period_missing",
        "scope_missing",
    ]:
        rate = df[column].mean() * 100
        print(f"{column}: {rate:.1f}%")

    print("\n========== SUPPORTED NUMERICAL CASES ==========")

    mask = (
        df["label"].eq("Supported")
        & df["numerical_consistency"].eq(0.0)
        & df["numerical_missing"].eq(0)
    )

    print(f"Cases to review: {int(mask.sum())}")


# ============================================================
# 5. DISPLAY INDIVIDUAL CASES
# ============================================================

def print_cases(df: pd.DataFrame) -> None:
    """Display claim, evidence, and feature values."""

    if df.empty:
        print("\nNo matching cases found.")
        return

    for index, (_, row) in enumerate(df.iterrows(), start=1):

        print("\n" + "=" * 70)
        print(f"CASE {index}")
        print("=" * 70)

        print(f"\nSubset: {row['subset']}")
        print(f"Split: {row['split']}")
        print(f"Label: {row['label']}")
        print(f"Answer ID: {row['answer_id']}")

        print("\nCLAIM:")
        print(row["claim"])

        print("\nEVIDENCE:")
        evidence = str(row["evidence"])

        print(evidence[:MAX_EVIDENCE_CHARS])

        if len(evidence) > MAX_EVIDENCE_CHARS:
            print("\n[Evidence truncated for display]")

        print("\nFEATURES:")

        for column in FEATURE_COLUMNS:
            print(f"{column}: {row[column]:.4f}")


# ============================================================
# 6. COMMAND LINE INTERFACE
# ============================================================

def main() -> None:
    """Run feature analysis from the command line."""

    parser = argparse.ArgumentParser(
        description="Analyze ClaimCheck financial features."
    )

    parser.add_argument(
        "--subset",
        choices=["FinQA", "TATQA"],
        default=DEFAULT_SUBSET,
    )

    parser.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
    )

    parser.add_argument(
        "--examples",
        type=int,
        default=DEFAULT_EXAMPLES,
    )

    args = parser.parse_args()

    df = prepare_features(
        subset=args.subset,
        sample_size=args.sample_size,
    )

    print_feature_summary(df)

    cases = select_numerical_cases(
        df,
        max_examples=args.examples,
    )

    print_cases(cases)


if __name__ == "__main__":
    main()

"""Build a feature matrix from claim and evidence pairs."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .text_features import compute_cosine_similarity
from .numerical_features import compute_numerical_consistency
from .period_features import compute_period_consistency
from .scope_features import compute_scope_consistency


# ============================================================
# 1. FEATURE COLUMNS
# ============================================================

FEATURE_COLUMNS = [
    "cosine_similarity",
    "numerical_consistency",
    "period_consistency",
    "scope_consistency",
    "numerical_missing",
    "period_missing",
    "scope_missing",
]


# ============================================================
# 2. BUILD FEATURE MATRIX
# ============================================================

def build_feature_matrix(
    df: pd.DataFrame,
    vectorizer,
    claim_column: str = "claim_clean",
    evidence_column: str = "evidence_clean",
) -> pd.DataFrame:
    """Convert claim and evidence pairs into numerical features."""

    # Check required columns.
    required_columns = [claim_column, evidence_column]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # Read preprocessed text without modifying the input dataframe.
    claims = df[claim_column].fillna("").astype(str).tolist()
    evidence = df[evidence_column].fillna("").astype(str).tolist()

    # Compute text similarity using the fitted vectorizer.
    # The function already performs TF-IDF transformation.
    # Never fit the vectorizer on validation or test data.
    similarities = compute_cosine_similarity(
        vectorizer,
        claims,
        evidence,
    )

    # Calculate financial consistency features.
    numerical_scores = [
        compute_numerical_consistency(claim, source)
        for claim, source in zip(claims, evidence)
    ]

    period_scores = [
        compute_period_consistency(claim, source)
        for claim, source in zip(claims, evidence)
    ]

    scope_scores = [
        compute_scope_consistency(claim, source)
        for claim, source in zip(claims, evidence)
    ]

    # Build the initial feature matrix.
    features = pd.DataFrame(
        {
            "cosine_similarity": np.asarray(similarities).ravel(),
            "numerical_consistency": numerical_scores,
            "period_consistency": period_scores,
            "scope_consistency": scope_scores,
        },
        index=df.index,
    )

    # Add missing indicators before replacing NaN values.
    for name in ["numerical", "period", "scope"]:
        score_column = f"{name}_consistency"
        missing_column = f"{name}_missing"

        features[missing_column] = (
            features[score_column].isna().astype(int)
        )

        # Replace NaN with zero for Logistic Regression.
        # Missing indicators preserve the original information.
        features[score_column] = (
            features[score_column].fillna(0.0)
        )

    # Keep the same feature order for all dataset splits.
    features = features[FEATURE_COLUMNS]

    return features

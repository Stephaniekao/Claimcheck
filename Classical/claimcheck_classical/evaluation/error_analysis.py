"""Analyze ClaimCheck claim-level predictions and errors.

Supports:
1. Binary classification error analysis.
2. Three-label classification error analysis.

Gold labels:
    Supported = 0
    Unsupported = 1

Three-label predictions:
    Supported, Uncertain, Unsupported
"""

from __future__ import annotations

import numpy as np
import pandas as pd


LABEL_MAPPING = {
    0: "Supported",
    1: "Unsupported",
}

ERROR_TYPES = (
    "FP",
    "FN",
    "TP",
    "TN",
    "U_UNSUPPORTED",
    "U_SUPPORTED",
)


# ============================================================
# 1. BUILD PREDICTION TABLE
# ============================================================

def build_prediction_table(
    df: pd.DataFrame,
    X: pd.DataFrame,
    y_true,
    probabilities,
    threshold: float = 0.5,
    lower_threshold: float | None = None,
    upper_threshold: float | None = None,
) -> pd.DataFrame:
    """Combine claims, evidence, predictions, and feature values.

    Binary mode:
        Uses P(Unsupported) and a single threshold.

    Three-label mode:
        Uses P(Supported) and two thresholds.

    Three-label mode is enabled when both lower_threshold
    and upper_threshold are provided.
    """

    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities, dtype=float)

    if y_true.ndim != 1 or probabilities.ndim != 1:
        raise ValueError("Inputs must be one-dimensional.")

    if (
        len(df) != len(X)
        or len(df) != len(y_true)
        or len(df) != len(probabilities)
    ):
        raise ValueError("Input lengths must match.")

    if len(df) == 0:
        raise ValueError("Input data cannot be empty.")

    if not np.isin(y_true, [0, 1]).all():
        raise ValueError("True labels must be 0 or 1.")

    if (
        not np.isfinite(probabilities).all()
        or ((probabilities < 0) | (probabilities > 1)).any()
    ):
        raise ValueError(
            "Probabilities must be finite and in [0, 1]."
        )

    if not {"claim", "evidence"}.issubset(df.columns):
        raise ValueError(
            "Data must contain claim and evidence columns."
        )

    three_label_mode = (
        lower_threshold is not None
        or upper_threshold is not None
    )

    if three_label_mode:
        if lower_threshold is None or upper_threshold is None:
            raise ValueError(
                "Both lower and upper thresholds are required."
            )

        if not (
            np.isfinite(lower_threshold)
            and np.isfinite(upper_threshold)
            and 0 <= lower_threshold < upper_threshold <= 1
        ):
            raise ValueError(
                "Thresholds must satisfy 0 <= lower < upper <= 1."
            )

        # In three-label mode, probabilities are P(Supported).
        p_supported = probabilities
        p_unsupported = 1.0 - probabilities

        predicted_labels = np.select(
            [
                p_supported < lower_threshold,
                p_supported > upper_threshold,
            ],
            [
                "Unsupported",
                "Supported",
            ],
            default="Uncertain",
        )

    else:
        if not np.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError(
                "Threshold must be between 0 and 1."
            )

        # In binary mode, probabilities are P(Unsupported).
        p_unsupported = probabilities
        p_supported = 1.0 - probabilities

        y_pred = (
            p_unsupported >= threshold
        ).astype(int)

        predicted_labels = np.array([
            LABEL_MAPPING[int(label)]
            for label in y_pred
        ])

    # --------------------------------------------------------
    # Build result table
    # --------------------------------------------------------

    result = df[
        ["claim", "evidence"]
    ].reset_index(drop=True).copy()

    result["true_label"] = [
        LABEL_MAPPING[int(label)]
        for label in y_true
    ]

    result["predicted_label"] = predicted_labels

    result["p_supported"] = p_supported
    result["p_unsupported"] = p_unsupported

    if three_label_mode:
        result["lower_threshold"] = lower_threshold
        result["upper_threshold"] = upper_threshold
    else:
        result["threshold"] = threshold

    # --------------------------------------------------------
    # Classify prediction outcomes
    # --------------------------------------------------------

    gold_supported = y_true == 0
    gold_unsupported = y_true == 1

    pred_supported = predicted_labels == "Supported"
    pred_unsupported = predicted_labels == "Unsupported"
    pred_uncertain = predicted_labels == "Uncertain"

    result["error_type"] = np.select(
        [
            gold_supported & pred_unsupported,
            gold_unsupported & pred_supported,
            gold_unsupported & pred_unsupported,
            gold_supported & pred_supported,
            gold_unsupported & pred_uncertain,
            gold_supported & pred_uncertain,
        ],
        [
            "FP",
            "FN",
            "TP",
            "TN",
            "U_UNSUPPORTED",
            "U_SUPPORTED",
        ],
        default="UNKNOWN",
    )

    if (result["error_type"] == "UNKNOWN").any():
        raise ValueError(
            "Unknown prediction outcome detected."
        )

    # --------------------------------------------------------
    # Add feature values
    # --------------------------------------------------------

    features = X.reset_index(drop=True).copy()

    overlapping = (
        set(result.columns)
        & set(features.columns)
    )

    if overlapping:
        raise ValueError(
            "Feature columns overlap with prediction columns: "
            f"{overlapping}"
        )

    result = pd.concat(
        [result, features],
        axis=1,
    )

    return result


# ============================================================
# 2. GET ERROR CASES
# ============================================================

def get_error_cases(
    prediction_table: pd.DataFrame,
    error_type: str,
    top_n: int = 20,
) -> pd.DataFrame:
    """Return the most confident cases of a prediction outcome."""

    if error_type not in ERROR_TYPES:
        raise ValueError(
            f"Error type must be one of {ERROR_TYPES}."
        )

    if top_n <= 0:
        raise ValueError(
            "top_n must be positive."
        )

    required_columns = {
        "error_type",
        "p_unsupported",
    }

    if not required_columns.issubset(
        prediction_table.columns
    ):
        raise ValueError(
            "Prediction table is missing required columns."
        )

    cases = prediction_table[
        prediction_table["error_type"] == error_type
    ].copy()

    # FP / TP: most confident Unsupported predictions.
    # FN / TN: most confident Supported predictions.
    # U_UNSUPPORTED / U_SUPPORTED:
    # sort by distance from P(Supported)=0.5,
    # with the most uncertain probabilities first.

    if error_type in {
        "U_UNSUPPORTED",
        "U_SUPPORTED",
    }:
        cases["_distance_to_half"] = (
            cases["p_unsupported"] - 0.5
        ).abs()

        cases = cases.sort_values(
            "_distance_to_half",
            ascending=True,
            kind="mergesort",
        )

        cases = cases.drop(
            columns="_distance_to_half"
        )

    else:
        ascending = error_type in {"FN", "TN"}

        cases = cases.sort_values(
            "p_unsupported",
            ascending=ascending,
            kind="mergesort",
        )

    return (
        cases.head(top_n)
        .reset_index(drop=True)
    )


# ============================================================
# 3. SUMMARIZE ERRORS
# ============================================================

def summarize_errors(
    prediction_table: pd.DataFrame,
) -> pd.Series:
    """Count all binary or three-label prediction outcomes."""

    if "error_type" not in prediction_table.columns:
        raise ValueError(
            "Prediction table must contain error_type."
        )

    return (
        prediction_table["error_type"]
        .value_counts()
        .reindex(ERROR_TYPES, fill_value=0)
    )

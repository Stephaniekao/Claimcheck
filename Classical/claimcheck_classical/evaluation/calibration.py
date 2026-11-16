"""Probability calibration diagnostics for ClaimCheck."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss


def calibration_table(
    y_true,
    probabilities,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Summarize predicted and observed probabilities by bin."""

    y = np.asarray(y_true)
    p = np.asarray(probabilities, dtype=float)

    if y.ndim != 1 or p.ndim != 1:
        raise ValueError("Inputs must be one-dimensional.")

    if len(y) == 0 or len(y) != len(p):
        raise ValueError("Inputs must have equal nonzero lengths.")

    if not np.isin(y, [0, 1]).all():
        raise ValueError("Labels must be binary: 0 or 1.")

    if not np.isfinite(p).all():
        raise ValueError("Probabilities must be finite.")

    if ((p < 0) | (p > 1)).any():
        raise ValueError("Probabilities must be between 0 and 1.")

    if not isinstance(n_bins, int) or n_bins < 1:
        raise ValueError("n_bins must be a positive integer.")

    # Equal-width bins, including probability 1.0.
    bin_ids = np.minimum(
        (p * n_bins).astype(int),
        n_bins - 1,
    )

    rows = []

    for i in range(n_bins):
        mask = bin_ids == i
        count = int(mask.sum())

        mean_predicted = (
            float(p[mask].mean()) if count else np.nan
        )
        observed_rate = (
            float(y[mask].mean()) if count else np.nan
        )

        rows.append({
            "bin": i + 1,
            "lower_bound": i / n_bins,
            "upper_bound": (i + 1) / n_bins,
            "count": count,
            "mean_predicted": mean_predicted,
            "observed_rate": observed_rate,
            "absolute_gap": (
                abs(mean_predicted - observed_rate)
                if count else np.nan
            ),
        })

    return pd.DataFrame(rows)


def expected_calibration_error(
    y_true,
    probabilities,
    n_bins: int = 10,
) -> float:
    """Compute weighted expected calibration error."""

    table = calibration_table(
        y_true,
        probabilities,
        n_bins=n_bins,
    )

    total = int(table["count"].sum())

    return float(
        (
            table["count"]
            * table["absolute_gap"].fillna(0)
        ).sum() / total
    )


def evaluate_calibration(
    y_true,
    probabilities,
    n_bins: int = 10,
) -> dict:
    """Return Brier Score, ECE, and calibration table."""

    table = calibration_table(
        y_true,
        probabilities,
        n_bins=n_bins,
    )

    return {
        "brier_score": float(
            brier_score_loss(y_true, probabilities)
        ),
        "ece": expected_calibration_error(
            y_true,
            probabilities,
            n_bins=n_bins,
        ),
        "table": table,
    }

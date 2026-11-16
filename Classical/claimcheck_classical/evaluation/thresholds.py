"""Threshold selection for ClaimCheck classical classification.

Supports:
1. Binary threshold selection using P(Unsupported).
2. Dual threshold selection using P(Supported).

Three-label decisions:
    P(Supported) < lower  -> Unsupported
    P(Supported) > upper  -> Supported
    Otherwise             -> Uncertain
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve


# ============================================================
# 1. EXISTING BINARY THRESHOLD FUNCTIONS
# ============================================================

def threshold_table(
    y_true,
    probabilities,
) -> pd.DataFrame:
    """Evaluate all distinct binary decision thresholds."""

    y = np.asarray(y_true)
    p = np.asarray(probabilities, dtype=float)

    if y.ndim != 1 or p.ndim != 1:
        raise ValueError("Inputs must be one-dimensional.")

    if len(y) == 0 or len(y) != len(p):
        raise ValueError("Inputs must have equal nonzero lengths.")

    if not np.isin(y, [0, 1]).all():
        raise ValueError("Labels must be binary.")

    if not np.isfinite(p).all():
        raise ValueError("Probabilities must be finite.")

    if ((p < 0) | (p > 1)).any():
        raise ValueError("Probabilities must be between 0 and 1.")

    if not np.any(y == 1):
        raise ValueError("At least one Unsupported claim is required.")

    precision, recall, thresholds = precision_recall_curve(
        y,
        p,
    )

    table = pd.DataFrame({
        "threshold": thresholds,
        "precision": precision[:-1],
        "recall": recall[:-1],
    })

    table["f1"] = np.divide(
        2 * table["precision"] * table["recall"],
        table["precision"] + table["recall"],
        out=np.zeros(len(table), dtype=float),
        where=(
            table["precision"] + table["recall"]
        ) > 0,
    )

    return table


def select_threshold(
    y_true,
    probabilities,
    target_recall: float = 0.80,
) -> dict:
    """Select highest-precision binary threshold meeting recall target."""

    if not 0 < target_recall <= 1:
        raise ValueError(
            "target_recall must be in (0, 1]."
        )

    table = threshold_table(
        y_true,
        probabilities,
    )

    candidates = table.loc[
        table["recall"] >= target_recall
    ].copy()

    if candidates.empty:
        raise ValueError(
            "No threshold satisfies the recall target."
        )

    candidates = candidates.sort_values(
        by=["precision", "f1", "threshold"],
        ascending=[False, False, False],
        kind="mergesort",
    )

    best = candidates.iloc[0]

    return {
        "threshold": float(best["threshold"]),
        "precision": float(best["precision"]),
        "recall": float(best["recall"]),
        "f1": float(best["f1"]),
        "target_recall": float(target_recall),
        "table": table,
    }


# ============================================================
# 2. DUAL THRESHOLD OPTIMIZATION
# ============================================================

def select_dual_thresholds(
    y_true,
    p_supported,
    min_precision: float = 0.60,
    max_uncertain_rate: float = 1 / 3,
) -> dict:
    """Select optimal dual thresholds on validation data.

    Objective:
        Maximize strict Unsupported recall.

    Constraints:
        Unsupported precision >= min_precision.
        Uncertain rate <= max_uncertain_rate.

    Tie-breakers:
        1. Higher Unsupported precision.
        2. Lower Uncertain rate.
        3. Lower threshold values.

    Gold labels:
        Supported = 0
        Unsupported = 1

    Returns an explicit infeasible result if no valid pair exists.
    """

    y = np.asarray(y_true)
    p = np.asarray(p_supported, dtype=float)

    # --------------------------------------------------------
    # Input validation
    # --------------------------------------------------------

    if y.ndim != 1 or p.ndim != 1:
        raise ValueError("Inputs must be one-dimensional.")

    if len(y) == 0 or len(y) != len(p):
        raise ValueError("Inputs must have equal nonzero lengths.")

    if not np.isin(y, [0, 1]).all():
        raise ValueError("Gold labels must be 0 or 1.")

    if not np.isfinite(p).all():
        raise ValueError("Probabilities must be finite.")

    if ((p < 0) | (p > 1)).any():
        raise ValueError("Probabilities must be between 0 and 1.")

    if not np.isfinite(min_precision) or not 0 <= min_precision <= 1:
        raise ValueError(
            "min_precision must be between 0 and 1."
        )

    if (
        not np.isfinite(max_uncertain_rate)
        or not 0 <= max_uncertain_rate <= 1
    ):
        raise ValueError(
            "max_uncertain_rate must be between 0 and 1."
        )

    n = len(y)
    n_unsupported = int(np.sum(y == 1))

    if n_unsupported == 0:
        raise ValueError(
            "At least one Unsupported claim is required."
        )

    # --------------------------------------------------------
    # Sort probabilities and group identical values
    # --------------------------------------------------------

    order = np.argsort(p, kind="mergesort")

    sorted_p = p[order]
    sorted_y = y[order]

    unique_p, starts, counts = np.unique(
        sorted_p,
        return_index=True,
        return_counts=True,
    )

    unsupported_per_group = np.add.reduceat(
        (sorted_y == 1).astype(int),
        starts,
    )

    cumulative_counts = np.cumsum(counts)
    cumulative_unsupported = np.cumsum(
        unsupported_per_group
    )

    # --------------------------------------------------------
    # Generate complete lower decision boundaries
    # --------------------------------------------------------

    # Every observed probability is a possible lower threshold.
    #
    # Because the rule uses p < lower, a threshold equal to
    # an observed score excludes that score from Unsupported.
    #
    # A threshold immediately above the maximum observed
    # score is also needed to allow all claims to be
    # classified Unsupported, provided max(p) < 1.

    lower_values = unique_p.copy()

    if unique_p[-1] < 1.0:
        extra_lower = np.nextafter(
            unique_p[-1],
            1.0,
        )

        lower_values = np.append(
            lower_values,
            extra_lower,
        )

    # A valid upper threshold must be strictly greater
    # than lower and no greater than 1.
    lower_values = lower_values[
        lower_values < 1.0
    ]

    if len(lower_values) == 0:
        return {
            "feasible": False,
            "reason": (
                "No valid dual thresholds exist because "
                "all support probabilities equal 1."
            ),
            "lower_threshold": None,
            "upper_threshold": None,
        }

    # --------------------------------------------------------
    # Count Unsupported predictions for every lower boundary
    # --------------------------------------------------------

    left_indices = np.searchsorted(
        sorted_p,
        lower_values,
        side="left",
    )

    cumulative_positive = np.concatenate([
        np.array([0], dtype=int),
        cumulative_unsupported,
    ])

    # Map a row count to the corresponding number of
    # true Unsupported claims using grouped boundaries.
    group_indices = np.searchsorted(
        cumulative_counts,
        left_indices,
        side="right",
    )

    true_unsupported = cumulative_positive[
        group_indices
    ]

    predicted_unsupported = left_indices

    precision = np.divide(
        true_unsupported,
        predicted_unsupported,
        out=np.zeros(
            len(lower_values),
            dtype=float,
        ),
        where=predicted_unsupported > 0,
    )

    strict_recall = (
        true_unsupported / n_unsupported
    )

    # --------------------------------------------------------
    # Choose the smallest valid upper threshold
    # --------------------------------------------------------

    # For a fixed lower threshold, changing upper does not
    # affect Unsupported precision or strict recall.
    #
    # Therefore the smallest representable upper threshold
    # minimizes Uncertain predictions.
    #
    # np.nextafter handles floating-point boundaries without
    # assuming that a midpoint is representable.

    upper_values = np.nextafter(
        lower_values,
        1.0,
    )

    right_indices = np.searchsorted(
        sorted_p,
        upper_values,
        side="right",
    )

    uncertain_counts = (
        right_indices - left_indices
    )

    uncertain_rate = uncertain_counts / n

    # Use integer counts for the uncertain-rate constraint.
    max_uncertain_count = int(
        np.floor(max_uncertain_rate * n)
    )

    eligible = (
        (precision >= min_precision)
        & (uncertain_counts <= max_uncertain_count)
    )

    eligible_indices = np.flatnonzero(eligible)

    if len(eligible_indices) == 0:
        return {
            "feasible": False,
            "reason": (
                "No dual-threshold pair satisfies "
                "Unsupported precision >= "
                f"{min_precision:.4f} and "
                "Uncertain rate <= "
                f"{max_uncertain_rate:.4f}."
            ),
            "lower_threshold": None,
            "upper_threshold": None,
            "min_precision": float(min_precision),
            "max_uncertain_rate": float(
                max_uncertain_rate
            ),
            "candidates_checked": int(
                len(lower_values)
            ),
        }

    # --------------------------------------------------------
    # Select the optimal feasible candidate
    # --------------------------------------------------------

    best_index = max(
        eligible_indices,
        key=lambda i: (
            strict_recall[i],
            precision[i],
            -uncertain_rate[i],
            -lower_values[i],
            -upper_values[i],
        ),
    )

    lower = float(lower_values[best_index])
    upper = float(upper_values[best_index])

    selected_precision = float(
        precision[best_index]
    )

    selected_recall = float(
        strict_recall[best_index]
    )

    selected_uncertain_rate = float(
        uncertain_rate[best_index]
    )

    selected_tp = int(
        true_unsupported[best_index]
    )

    selected_predicted_unsupported = int(
        predicted_unsupported[best_index]
    )

    # Lenient recall counts true Unsupported claims
    # predicted either Unsupported or Uncertain.
    right = int(right_indices[best_index])

    lenient_tp = int(
        np.sum(sorted_y[:right] == 1)
    )

    lenient_recall = (
        lenient_tp / n_unsupported
    )

    strict_f1 = (
        2 * selected_precision * selected_recall
        / (selected_precision + selected_recall)
        if selected_precision + selected_recall > 0
        else 0.0
    )

    return {
        "feasible": True,
        "reason": None,
        "lower_threshold": lower,
        "upper_threshold": upper,
        "strict_recall": selected_recall,
        "lenient_recall": float(lenient_recall),
        "unsupported_precision": selected_precision,
        "strict_f1": float(strict_f1),
        "uncertain_rate": selected_uncertain_rate,
        "coverage": float(
            1 - selected_uncertain_rate
        ),
        "predicted_unsupported": (
            selected_predicted_unsupported
        ),
        "true_unsupported": selected_tp,
        "predicted_uncertain": int(
            uncertain_counts[best_index]
        ),
        "min_precision": float(min_precision),
        "max_uncertain_rate": float(
            max_uncertain_rate
        ),
        "candidates_checked": int(
            len(lower_values)
        ),
        "feasible_candidates": int(
            len(eligible_indices)
        ),
    }

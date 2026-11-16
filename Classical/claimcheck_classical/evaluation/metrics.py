"""
ClaimCheck Classical Model - Evaluation Metrics

1. Binary classification metrics
2. Three-label decision metrics
3. ROC-AUC and PR-AUC
4. Confusion matrix
5. Probability analysis

Gold labels:
    Supported = 0
    Unsupported = 1

Predicted labels:
    Supported, Uncertain, Unsupported
"""

from __future__ import annotations

import numpy as np

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

POSITIVE_LABEL = 1
DEFAULT_THRESHOLD = 0.5

THREE_LABELS = (
    "Supported",
    "Uncertain",
    "Unsupported",
)


# ============================================================
# 2. BINARY CLASSIFICATION METRICS
# ============================================================

def evaluate_binary(
    y_true,
    probabilities,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict:
    """Evaluate binary predictions with Unsupported as positive."""

    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)

    if y_true.ndim != 1 or probabilities.ndim != 1:
        raise ValueError("Inputs must be one-dimensional.")

    if len(y_true) != len(probabilities):
        raise ValueError(
            "Labels and probabilities must have the same length."
        )

    if len(y_true) == 0:
        raise ValueError("Evaluation data cannot be empty.")

    if not np.isin(y_true, [0, 1]).all():
        raise ValueError("Labels must be 0 or 1.")

    if not np.isfinite(probabilities).all():
        raise ValueError("Probabilities must be finite.")

    if not np.logical_and(
        probabilities >= 0,
        probabilities <= 1,
    ).all():
        raise ValueError("Probabilities must be between 0 and 1.")

    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Threshold must be between 0 and 1.")

    y_pred = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    results = {
        "n": int(len(y_true)),
        "supported_gold": int((y_true == 0).sum()),
        "unsupported_gold": int((y_true == 1).sum()),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(
                y_true,
                y_pred,
                pos_label=POSITIVE_LABEL,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                y_pred,
                pos_label=POSITIVE_LABEL,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                y_pred,
                pos_label=POSITIVE_LABEL,
                zero_division=0,
            )
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "roc_auc": None,
        "pr_auc": None,
    }

    if len(np.unique(y_true)) == 2:
        results["roc_auc"] = float(
            roc_auc_score(y_true, probabilities)
        )
        results["pr_auc"] = float(
            average_precision_score(y_true, probabilities)
        )

    return results


# ============================================================
# 3. THREE-LABEL EVALUATION
# ============================================================

def evaluate_three_label(
    y_true,
    predicted_labels,
) -> dict:
    """Evaluate three-label decisions against binary gold labels.

    Strict recall:
        True Unsupported predicted Unsupported.

    Lenient recall:
        True Unsupported predicted Unsupported or Uncertain.

    Unsupported precision:
        True Unsupported among claims predicted Unsupported.

    Uncertain is an abstention decision, not a gold class.
    """

    y_true = np.asarray(y_true)
    predicted_labels = np.asarray(predicted_labels)

    if y_true.ndim != 1 or predicted_labels.ndim != 1:
        raise ValueError("Inputs must be one-dimensional.")

    if len(y_true) != len(predicted_labels):
        raise ValueError(
            "Labels and predictions must have the same length."
        )

    if len(y_true) == 0:
        raise ValueError("Evaluation data cannot be empty.")

    if not np.isin(y_true, [0, 1]).all():
        raise ValueError("Gold labels must be 0 or 1.")

    if not np.isin(predicted_labels, THREE_LABELS).all():
        raise ValueError(
            "Predictions must be Supported, Uncertain, or Unsupported."
        )

    gold_unsupported = y_true == 1
    gold_supported = y_true == 0

    pred_supported = predicted_labels == "Supported"
    pred_uncertain = predicted_labels == "Uncertain"
    pred_unsupported = predicted_labels == "Unsupported"

    n = len(y_true)
    n_unsupported = int(gold_unsupported.sum())
    n_pred_unsupported = int(pred_unsupported.sum())

    strict_tp = int(
        (gold_unsupported & pred_unsupported).sum()
    )

    false_unsupported = int(
        (gold_supported & pred_unsupported).sum()
    )

    missed_unsupported = int(
        (gold_unsupported & pred_supported).sum()
    )

    uncertain_unsupported = int(
        (gold_unsupported & pred_uncertain).sum()
    )

    uncertain_supported = int(
        (gold_supported & pred_uncertain).sum()
    )

    strict_recall = (
        strict_tp / n_unsupported
        if n_unsupported > 0
        else 0.0
    )

    lenient_recall = (
        (strict_tp + uncertain_unsupported) / n_unsupported
        if n_unsupported > 0
        else 0.0
    )

    unsupported_precision = (
        strict_tp / n_pred_unsupported
        if n_pred_unsupported > 0
        else 0.0
    )

    strict_f1 = (
        2 * unsupported_precision * strict_recall
        / (unsupported_precision + strict_recall)
        if unsupported_precision + strict_recall > 0
        else 0.0
    )

    uncertain_rate = float(pred_uncertain.mean())
    coverage = 1.0 - uncertain_rate

    # A 2x3 table, since gold labels have only two classes.
    decision_counts = {
        "supported_to_supported": int(
            (gold_supported & pred_supported).sum()
        ),
        "supported_to_uncertain": uncertain_supported,
        "supported_to_unsupported": false_unsupported,
        "unsupported_to_supported": missed_unsupported,
        "unsupported_to_uncertain": uncertain_unsupported,
        "unsupported_to_unsupported": strict_tp,
    }

    return {
        "n": int(n),
        "supported_gold": int(gold_supported.sum()),
        "unsupported_gold": n_unsupported,
        "predicted_supported": int(pred_supported.sum()),
        "predicted_uncertain": int(pred_uncertain.sum()),
        "predicted_unsupported": n_pred_unsupported,
        "strict_recall": float(strict_recall),
        "lenient_recall": float(lenient_recall),
        "unsupported_precision": float(unsupported_precision),
        "strict_f1": float(strict_f1),
        "uncertain_rate": uncertain_rate,
        "coverage": float(coverage),
        "strict_tp": strict_tp,
        "false_unsupported": false_unsupported,
        "missed_unsupported": missed_unsupported,
        "uncertain_unsupported": uncertain_unsupported,
        "uncertain_supported": uncertain_supported,
        "decision_counts": decision_counts,
    }


# ============================================================
# 4. PROBABILITY SUMMARY
# ============================================================

def summarize_probabilities(probabilities) -> dict:
    """Summarize predicted Unsupported probabilities."""

    probabilities = np.asarray(probabilities)

    if probabilities.ndim != 1 or len(probabilities) == 0:
        raise ValueError("Probabilities must be a nonempty vector.")

    if not np.isfinite(probabilities).all():
        raise ValueError("Probabilities must be finite.")

    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Probabilities must be between 0 and 1.")

    return {
        "min": float(np.min(probabilities)),
        "mean": float(np.mean(probabilities)),
        "median": float(np.median(probabilities)),
        "max": float(np.max(probabilities)),
        "std": float(np.std(probabilities)),
    }


# ============================================================
# 5. DISPLAY BINARY METRICS
# ============================================================

def print_binary_metrics(results: dict) -> None:
    """Display binary classification evaluation results."""

    print("\n========== BINARY EVALUATION ==========")

    print(f"Total claims: {results['n']}")
    print(f"Supported: {results['supported_gold']}")
    print(f"Unsupported: {results['unsupported_gold']}")
    print(f"Threshold: {results['threshold']:.2f}")

    print("\nClassification metrics:")

    for metric in [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
    ]:
        value = results[metric]

        if value is None:
            print(f"{metric}: N/A")
        else:
            print(f"{metric}: {value:.4f}")

    print("\nConfusion matrix:")
    print(f"TN: {results['tn']}")
    print(f"FP: {results['fp']}")
    print(f"FN: {results['fn']}")
    print(f"TP: {results['tp']}")


# ============================================================
# 6. DISPLAY THREE-LABEL METRICS
# ============================================================

def print_three_label_metrics(results: dict) -> None:
    """Display three-label evaluation results."""

    print("\n========== THREE-LABEL EVALUATION ==========")

    print(f"Total claims: {results['n']}")
    print(f"Supported gold: {results['supported_gold']}")
    print(f"Unsupported gold: {results['unsupported_gold']}")

    print("\nPredicted labels:")
    print(f"Supported: {results['predicted_supported']}")
    print(f"Uncertain: {results['predicted_uncertain']}")
    print(f"Unsupported: {results['predicted_unsupported']}")

    print("\nEvaluation metrics:")

    for metric in [
        "strict_recall",
        "lenient_recall",
        "unsupported_precision",
        "strict_f1",
        "uncertain_rate",
        "coverage",
    ]:
        print(f"{metric}: {results[metric]:.4f}")

    print("\nDecision counts (gold -> predicted):")

    for name, count in results["decision_counts"].items():
        print(f"{name}: {count}")


# ============================================================
# 7. DISPLAY PROBABILITY SUMMARY
# ============================================================

def print_probability_summary(summary: dict) -> None:
    """Display predicted probability statistics."""

    print("\n========== P(UNSUPPORTED) ==========")

    for name in [
        "min",
        "mean",
        "median",
        "max",
        "std",
    ]:
        print(f"{name}: {summary[name]:.4f}")

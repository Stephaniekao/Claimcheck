"""ClaimCheck Classical Model - Main Pipeline.

1. Load RAGBench FinQA and TATQA training data.
2. Build features and train Logistic Regression.
3. Save model and TF-IDF vectorizer.
4. Evaluate binary baseline on validation data.
5. Evaluate probability calibration.
6. Select dual thresholds using validation data only.
7. Evaluate three-label predictions.
8. Export metrics, thresholds, and error analysis.

Gold labels:
    Supported = 0
    Unsupported = 1

Three-label decisions use P(Supported).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from claimcheck_classical.config import RANDOM_SEED
from claimcheck_classical.data_loader import load_ragbench_claims
from claimcheck_classical.preprocessing import preprocess_claims

from claimcheck_classical.features.evidence_features import (
    FULL_FEATURE_COLUMNS,
    build_full_feature_matrix,
    with_claim_text,
)

from claimcheck_classical.training.train import (
    SUBSETS,
    LABEL_MAPPING,
    load_training_data,
    prepare_training_features,
    train_logistic_regression,
)

from claimcheck_classical.evaluation.metrics import (
    evaluate_binary,
    evaluate_three_label,
    summarize_probabilities,
    print_binary_metrics,
    print_three_label_metrics,
    print_probability_summary,
)

from claimcheck_classical.evaluation.error_analysis import (
    build_prediction_table,
    summarize_errors,
)

from claimcheck_classical.evaluation.calibration import (
    evaluate_calibration,
)

from claimcheck_classical.evaluation.thresholds import (
    select_threshold,
    select_dual_thresholds,
)

from claimcheck_classical.inference.predict import (
    classify_support_probability,
)

from claimcheck_classical.model_io import save_model


# ============================================================
# 1. CONFIGURATION
# ============================================================

DEFAULT_SAMPLE_SIZE = 1000
DEFAULT_SAMPLING = "random"
DEFAULT_UNSUPPORTED_RATIO = 0.5

DEFAULT_THRESHOLD = 0.5
DEFAULT_TARGET_RECALL = 0.80

DEFAULT_MIN_PRECISION = 0.60
DEFAULT_MAX_UNCERTAIN_RATE = 1 / 3
DEFAULT_CALIBRATION_BINS = 10

PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "outputs"
MODEL_DIR = PROJECT_DIR / "models"


def format_number(value: float) -> str:
    """Format a float for filenames."""
    return f"{value:.2f}".replace(".", "p")


def experiment_name(
    sample_size: int,
    sampling: str,
    unsupported_ratio: float,
) -> str:
    """Build a reproducible experiment identifier."""

    name = f"{sampling}_n{sample_size}"

    if sampling == "stratified":
        name += f"_r{format_number(unsupported_ratio)}"

    return name


def save_json(data: dict, path: Path) -> None:
    """Save JSON, converting NumPy scalars when necessary."""

    def convert(value):
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.bool_):
            return bool(value)
        raise TypeError(
            f"Cannot serialize {type(value).__name__}"
        )

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            indent=2,
            default=convert,
            allow_nan=False,
        )


# ============================================================
# 2. LOAD VALIDATION DATA
# ============================================================

def load_validation_data() -> pd.DataFrame:
    """Load original RAGBench validation splits."""

    frames = []

    for subset in SUBSETS:
        df = load_ragbench_claims(
            subset=subset,
            split="validation",
        )

        df = df.copy()
        df["subset"] = subset

        print(f"{subset}: {len(df)} validation claims")
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


# ============================================================
# 3. PREPARE VALIDATION FEATURES
# ============================================================

def prepare_validation_features(
    df: pd.DataFrame,
    vectorizer,
):
    """Transform validation data without refitting TF-IDF."""

    processed = preprocess_claims(df)

    X = build_full_feature_matrix(
        processed,
        vectorizer,
    )

    y = processed["label"].map(LABEL_MAPPING)

    if y.isna().any():
        raise ValueError(
            "Unknown labels found in validation data."
        )

    y = y.astype(int)

    if not np.isfinite(X.to_numpy()).all():
        raise ValueError(
            "Validation features contain NaN or infinity."
        )

    # Attach claim text after the numeric check.
    X = with_claim_text(X, processed)

    return X, y


# ============================================================
# 4. TRAINING
# ============================================================

def run_training(
    sample_size: int,
    sampling: str,
    unsupported_ratio: float,
):
    """Train Logistic Regression using training data only."""

    print("\n========== LOADING TRAINING DATA ==========")

    train_df = load_training_data(
        sample_size=sample_size,
        sampling=sampling,
        unsupported_ratio=unsupported_ratio,
    )

    X_train, y_train, vectorizer = (
        prepare_training_features(train_df)
    )

    print(f"\nTraining matrix: {X_train.shape}")

    print("\nTraining label distribution:")
    print(y_train.value_counts().sort_index().to_string())

    print("\n========== TRAINING MODEL ==========")

    model = train_logistic_regression(
        X_train,
        y_train,
    )

    print("Logistic Regression training completed.")

    return model, vectorizer


# ============================================================
# 5. MODEL SAVING
# ============================================================

def save_trained_model(
    model,
    vectorizer,
    sample_size: int,
    sampling: str,
    unsupported_ratio: float,
) -> Path:
    """Save classifier and fitted TF-IDF vectorizer."""

    name = experiment_name(
        sample_size,
        sampling,
        unsupported_ratio,
    )

    model_path = (
        MODEL_DIR / f"classical_model_{name}.joblib"
    )

    saved_path = save_model(
        model=model,
        vectorizer=vectorizer,
        path=model_path,
    )

    print("\n========== MODEL SAVING ==========")
    print(f"Saved model: {saved_path}")

    return saved_path


# ============================================================
# 6. CALIBRATION
# ============================================================

def run_calibration(
    y_val: pd.Series,
    probabilities: np.ndarray,
    sample_size: int,
    sampling: str,
    unsupported_ratio: float,
    n_bins: int = DEFAULT_CALIBRATION_BINS,
) -> dict:
    """Evaluate P(Unsupported) calibration."""

    print("\n========== CALIBRATION ==========")

    results = evaluate_calibration(
        y_true=y_val,
        probabilities=probabilities,
        n_bins=n_bins,
    )

    print(f"Brier Score: {results['brier_score']:.4f}")
    print(f"ECE: {results['ece']:.4f}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    name = experiment_name(
        sample_size,
        sampling,
        unsupported_ratio,
    )

    output_path = (
        OUTPUT_DIR
        / f"calibration_{name}_bins{n_bins}.csv"
    )

    results["table"].to_csv(
        output_path,
        index=False,
    )

    print(f"Saved calibration table: {output_path}")

    return results


# ============================================================
# 7. LEGACY BINARY THRESHOLD SELECTION
# ============================================================

def run_threshold_selection(
    y_val: pd.Series,
    probabilities: np.ndarray,
    sample_size: int,
    sampling: str,
    unsupported_ratio: float,
    target_recall: float = DEFAULT_TARGET_RECALL,
) -> dict:
    """Preserve binary threshold analysis for comparison."""

    print("\n========== BINARY THRESHOLD ANALYSIS ==========")

    results = select_threshold(
        y_true=y_val,
        probabilities=probabilities,
        target_recall=target_recall,
    )

    print(f"Target Recall: {target_recall:.4f}")
    print(f"Selected Threshold: {results['threshold']:.6f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1: {results['f1']:.4f}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    name = experiment_name(
        sample_size,
        sampling,
        unsupported_ratio,
    )

    output_path = (
        OUTPUT_DIR
        / (
            f"thresholds_{name}"
            f"_target{format_number(target_recall)}.csv"
        )
    )

    results["table"].to_csv(
        output_path,
        index=False,
    )

    print(f"Saved binary threshold table: {output_path}")

    return results


# ============================================================
# 8. DUAL THRESHOLD SELECTION
# ============================================================

def run_dual_threshold_selection(
    y_val,
    p_supported: np.ndarray,
    sample_size: int,
    sampling: str,
    unsupported_ratio: float,
    min_precision: float = DEFAULT_MIN_PRECISION,
    max_uncertain_rate: float = DEFAULT_MAX_UNCERTAIN_RATE,
) -> dict:
    """Optimize three-label thresholds on validation data."""

    print("\n========== DUAL THRESHOLD SELECTION ==========")

    results = select_dual_thresholds(
        y_true=y_val,
        p_supported=p_supported,
        min_precision=min_precision,
        max_uncertain_rate=max_uncertain_rate,
    )

    name = experiment_name(
        sample_size,
        sampling,
        unsupported_ratio,
    )

    output_path = (
        OUTPUT_DIR / f"dual_thresholds_{name}.json"
    )

    save_json(results, output_path)

    print(f"Feasible: {results['feasible']}")

    if results["feasible"]:
        print(
            "Lower Threshold: "
            f"{results['lower_threshold']:.17g}"
        )
        print(
            "Upper Threshold: "
            f"{results['upper_threshold']:.17g}"
        )
        print(
            f"Strict Recall: {results['strict_recall']:.4f}"
        )
        print(
            "Unsupported Precision: "
            f"{results['unsupported_precision']:.4f}"
        )
        print(
            f"Uncertain Rate: {results['uncertain_rate']:.4f}"
        )
    else:
        print(f"Reason: {results['reason']}")
        print(
            "Fallback: all claims will be labeled Uncertain."
        )

    print(f"Saved dual thresholds: {output_path}")

    return results


# ============================================================
# 9. THREE-LABEL PREDICTIONS
# ============================================================

def make_three_label_predictions(
    p_supported: np.ndarray,
    threshold_results: dict,
) -> np.ndarray:
    """Apply selected thresholds or all-Uncertain fallback."""

    probabilities = np.asarray(
        p_supported,
        dtype=float,
    )

    if not threshold_results["feasible"]:
        return np.full(
            len(probabilities),
            "Uncertain",
            dtype=object,
        )

    lower = threshold_results["lower_threshold"]
    upper = threshold_results["upper_threshold"]

    return np.array(
        [
            classify_support_probability(
                float(probability),
                lower,
                upper,
            )
            for probability in probabilities
        ],
        dtype=object,
    )


# ============================================================
# 10. ERROR ANALYSIS
# ============================================================

def export_error_analysis(
    validation_df: pd.DataFrame,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    p_unsupported: np.ndarray,
    p_supported: np.ndarray,
    predicted_labels: np.ndarray,
    threshold_results: dict,
    sample_size: int,
    sampling: str,
    unsupported_ratio: float,
) -> Path:
    """Export three-label predictions, errors, and features."""

    print("\n========== ERROR ANALYSIS ==========")

    name = experiment_name(
        sample_size,
        sampling,
        unsupported_ratio,
    )

    if threshold_results["feasible"]:
        prediction_table = build_prediction_table(
            df=validation_df,
            X=X_val[FULL_FEATURE_COLUMNS],
            y_true=y_val,
            probabilities=p_supported,
            lower_threshold=threshold_results[
                "lower_threshold"
            ],
            upper_threshold=threshold_results[
                "upper_threshold"
            ],
        )

    else:
        # Build the table without using invented thresholds.
        prediction_table = build_prediction_table(
            df=validation_df,
            X=X_val[FULL_FEATURE_COLUMNS],
            y_true=y_val,
            probabilities=p_unsupported,
            threshold=0.5,
        )

        # Replace binary decisions with the required fallback.
        prediction_table["predicted_label"] = "Uncertain"
        prediction_table["error_type"] = np.where(
            np.asarray(y_val) == 1,
            "U_UNSUPPORTED",
            "U_SUPPORTED",
        )

        prediction_table["threshold"] = np.nan
        prediction_table["lower_threshold"] = np.nan
        prediction_table["upper_threshold"] = np.nan

    if not np.array_equal(
        prediction_table["predicted_label"].to_numpy(),
        predicted_labels,
    ):
        raise ValueError(
            "Error analysis predictions do not match "
            "three-label evaluation predictions."
        )

    counts = summarize_errors(prediction_table)
    print(counts.to_string())

    prediction_table.insert(
        0,
        "subset",
        validation_df["subset"].to_numpy()
        if "subset" in validation_df.columns
        else ["unknown"] * len(validation_df),
    )

    prediction_table["fallback_all_uncertain"] = (
        not threshold_results["feasible"]
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = (
        OUTPUT_DIR / f"error_analysis_three_label_{name}.csv"
    )

    prediction_table.to_csv(
        output_path,
        index=False,
    )

    print(f"Saved error analysis: {output_path}")
    print(f"Exported claims: {len(prediction_table)}")

    return output_path


# ============================================================
# 11. VALIDATION
# ============================================================

def run_validation(
    model,
    vectorizer,
    threshold: float,
    sample_size: int,
    sampling: str,
    unsupported_ratio: float,
    target_recall: float = DEFAULT_TARGET_RECALL,
    min_precision: float = DEFAULT_MIN_PRECISION,
    max_uncertain_rate: float = DEFAULT_MAX_UNCERTAIN_RATE,
) -> dict:
    """Run binary diagnostics and three-label validation."""

    print("\n========== LOADING VALIDATION DATA ==========")

    validation_df = load_validation_data()

    X_val, y_val = prepare_validation_features(
        validation_df,
        vectorizer,
    )

    print(f"Validation matrix: {X_val.shape}")

    if 1 not in model.classes_:
        raise ValueError(
            "Model does not contain Unsupported class."
        )

    unsupported_index = list(
        model.classes_
    ).index(1)

    p_unsupported = model.predict_proba(
        X_val
    )[:, unsupported_index]

    p_supported = 1.0 - p_unsupported

    # --------------------------------------------------------
    # Binary diagnostics
    # --------------------------------------------------------

    print("\n========== BINARY VALIDATION ==========")

    binary_results = evaluate_binary(
        y_true=y_val,
        probabilities=p_unsupported,
        threshold=threshold,
    )

    probability_summary = summarize_probabilities(
        p_unsupported
    )

    print_binary_metrics(binary_results)
    print_probability_summary(probability_summary)

    # --------------------------------------------------------
    # Calibration
    # --------------------------------------------------------

    calibration_results = run_calibration(
        y_val=y_val,
        probabilities=p_unsupported,
        sample_size=sample_size,
        sampling=sampling,
        unsupported_ratio=unsupported_ratio,
    )

    # --------------------------------------------------------
    # Legacy binary threshold analysis
    # --------------------------------------------------------

    binary_threshold_results = run_threshold_selection(
        y_val=y_val,
        probabilities=p_unsupported,
        sample_size=sample_size,
        sampling=sampling,
        unsupported_ratio=unsupported_ratio,
        target_recall=target_recall,
    )

    # --------------------------------------------------------
    # Three-label threshold optimization
    # --------------------------------------------------------

    dual_results = run_dual_threshold_selection(
        y_val=y_val,
        p_supported=p_supported,
        sample_size=sample_size,
        sampling=sampling,
        unsupported_ratio=unsupported_ratio,
        min_precision=min_precision,
        max_uncertain_rate=max_uncertain_rate,
    )

    # --------------------------------------------------------
    # Three-label predictions and evaluation
    # --------------------------------------------------------

    predicted_labels = make_three_label_predictions(
        p_supported,
        dual_results,
    )

    three_label_results = evaluate_three_label(
        y_true=y_val,
        predicted_labels=predicted_labels,
    )

    print("\n========== THREE-LABEL VALIDATION ==========")

    print_three_label_metrics(three_label_results)

    # Verify the optimizer's reported metrics.
    if dual_results["feasible"]:
        for key in (
            "strict_recall",
            "lenient_recall",
            "unsupported_precision",
            "uncertain_rate",
        ):
            if not np.isclose(
                three_label_results[key],
                dual_results[key],
            ):
                raise ValueError(
                    f"Threshold selection mismatch: {key}"
                )

    else:
        if not np.all(
            predicted_labels == "Uncertain"
        ):
            raise ValueError(
                "Infeasible threshold selection must "
                "use all-Uncertain fallback."
            )

    # --------------------------------------------------------
    # Save evaluation results
    # --------------------------------------------------------

    name = experiment_name(
        sample_size,
        sampling,
        unsupported_ratio,
    )

    summary = {
        "experiment": name,
        "random_seed": RANDOM_SEED,
        "training": {
            "sample_size_per_subset": sample_size,
            "sampling": sampling,
            "unsupported_ratio": unsupported_ratio,
        },
        "binary_validation": binary_results,
        "binary_threshold": {
            "threshold": binary_threshold_results["threshold"],
            "precision": binary_threshold_results["precision"],
            "recall": binary_threshold_results["recall"],
            "f1": binary_threshold_results["f1"],
            "target_recall": target_recall,
        },
        "calibration": {
            "brier_score": calibration_results["brier_score"],
            "ece": calibration_results["ece"],
            "n_bins": DEFAULT_CALIBRATION_BINS,
        },
        "dual_threshold_selection": dual_results,
        "three_label_validation": three_label_results,
        "fallback_all_uncertain": not dual_results["feasible"],
    }

    summary_path = (
        OUTPUT_DIR / f"validation_summary_{name}.json"
    )

    save_json(summary, summary_path)

    print(f"\nSaved validation summary: {summary_path}")

    # --------------------------------------------------------
    # Export claim-level analysis
    # --------------------------------------------------------

    export_error_analysis(
        validation_df=validation_df,
        X_val=X_val,
        y_val=y_val,
        p_unsupported=p_unsupported,
        p_supported=p_supported,
        predicted_labels=predicted_labels,
        threshold_results=dual_results,
        sample_size=sample_size,
        sampling=sampling,
        unsupported_ratio=unsupported_ratio,
    )

    return summary


# ============================================================
# 12. COMMAND LINE INTERFACE
# ============================================================

def main() -> None:
    """Run the Classical Model pipeline."""

    parser = argparse.ArgumentParser(
        description="Run ClaimCheck Classical Model."
    )

    parser.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help="Maximum training claims per subset.",
    )

    parser.add_argument(
        "--sampling",
        choices=["random", "stratified"],
        default=DEFAULT_SAMPLING,
        help="Training sampling strategy.",
    )

    parser.add_argument(
        "--unsupported-ratio",
        type=float,
        default=DEFAULT_UNSUPPORTED_RATIO,
        help="Unsupported fraction for stratified sampling.",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Fixed threshold for legacy binary evaluation.",
    )

    parser.add_argument(
        "--target-recall",
        type=float,
        default=DEFAULT_TARGET_RECALL,
        help="Target recall for legacy binary threshold analysis.",
    )

    parser.add_argument(
        "--min-precision",
        type=float,
        default=DEFAULT_MIN_PRECISION,
        help="Minimum Unsupported precision for dual thresholds.",
    )

    parser.add_argument(
        "--max-uncertain-rate",
        type=float,
        default=DEFAULT_MAX_UNCERTAIN_RATE,
        help="Maximum allowed Uncertain fraction.",
    )

    args = parser.parse_args()

    print("========== CLAIMCHECK CLASSICAL ==========")
    print(f"Random seed: {RANDOM_SEED}")
    print(f"Sampling: {args.sampling}")
    print(f"Unsupported ratio: {args.unsupported_ratio}")
    print(f"Minimum precision: {args.min_precision}")
    print(f"Maximum uncertain rate: {args.max_uncertain_rate}")

    model, vectorizer = run_training(
        sample_size=args.sample_size,
        sampling=args.sampling,
        unsupported_ratio=args.unsupported_ratio,
    )

    save_trained_model(
        model=model,
        vectorizer=vectorizer,
        sample_size=args.sample_size,
        sampling=args.sampling,
        unsupported_ratio=args.unsupported_ratio,
    )

    run_validation(
        model=model,
        vectorizer=vectorizer,
        threshold=args.threshold,
        sample_size=args.sample_size,
        sampling=args.sampling,
        unsupported_ratio=args.unsupported_ratio,
        target_recall=args.target_recall,
        min_precision=args.min_precision,
        max_uncertain_rate=args.max_uncertain_rate,
    )

    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()

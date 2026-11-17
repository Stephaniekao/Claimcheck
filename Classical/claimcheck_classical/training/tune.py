"""Tune the ClaimCheck classical model (TF-IDF + Logistic Regression).

Follows the experiment card (Fields 3, 5, 8):
    * At most MAX_TRIALS (20) random trials over
      TF-IDF n-gram range x vocabulary size x L2 strength C.
    * Trials are scored on the official RAGBench VALIDATION splits only.
      The test splits are never loaded.
    * The best trial then gets the full evaluation: calibration,
      dual thresholds, three-label metrics, saved model.
    * Thresholds (Field 8): the lower threshold gives the highest strict
      recall with precision >= 0.60; the upper threshold then fills
      Uncertain up to 1/3 of claims (team rule). Claims the shared
      normalizer cannot read, or with no passage, are Uncertain whatever
      their score (Field 12) and count toward the 1/3.
    * Answer-level AUROC (Field 6): gold is adherence_score, and the
      rule that combines claim scores is chosen on validation. The
      candidate rules count toward the 20-trial limit (Field 5), as in
      the baseline, so the search uses MAX_TRIALS - 2 trials.
    * Precision at recall 0.80 uses the baseline's definition.

Gold labels:
    Supported = 0
    Unsupported = 1
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from claimcheck_classical.config import (
    DATA_DIR,
    LR_C_VALUES,
    MAX_TRIALS,
    MODEL_CONFIG,
    RANDOM_SEED,
    TFIDF_MAX_FEATURES,
    TFIDF_NGRAM_RANGE,
    USE_CLAIM_TEXT,
    USE_EXTRA_FEATURES,
)
from claimcheck_classical.data_loader import load_ragbench_claims
from claimcheck_classical.preprocessing import preprocess_claims

from claimcheck_classical.features.text_features import (
    create_tfidf_vectorizer,
)
from claimcheck_classical.features.evidence_features import (
    ANSWER_AGGREGATIONS,
    aggregate_answer_scores,
    select_answer_aggregation,
    build_full_feature_matrix,
    build_static_features,
    sample_trials,
    with_claim_text,
)

from claimcheck_classical.training.train import (
    SUBSETS,
    LABEL_MAPPING,
    coefficient_table,
    load_training_data,
    model_numeric_columns,
    train_logistic_regression,
)

from claimcheck_classical.evaluation.metrics import (
    evaluate_binary,
    evaluate_three_label,
)

from claimcheck_classical.evaluation.calibration import (
    evaluate_calibration,
)

from claimcheck_classical.evaluation.thresholds import (
    select_threshold,
    select_thresholds_with_forced_uncertain,
)

from claimcheck_classical.inference.predict import (
    classify_support_probability,
)

from claimcheck_classical.model_io import save_model
from claimcheck_classical.preprocessing import normalize_text
from claimcheck_classical.provenance import build_metadata, hardware_info
from claimcheck_classical.shared import (
    normalizer_failure_mask,
    precision_at_recall,
    shared_versions,
)

# Field 5: 20 trials in total. The answer aggregation rules chosen on
# validation count as trials, as in the baseline.
SEARCH_TRIALS = MAX_TRIALS - len(ANSWER_AGGREGATIONS)


PROJECT_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_DIR / "outputs"
MODEL_DIR = PROJECT_DIR / "models"

DEFAULT_TARGET_RECALL = 0.80
DEFAULT_MIN_PRECISION = 0.60
DEFAULT_MAX_UNCERTAIN_RATE = 1 / 3
DEFAULT_CALIBRATION_BINS = 10

# Claim-level PR AUC is the default selection metric: Unsupported is
# ~2% of claims, so F1 at a fixed 0.5 threshold is almost always 0.
SELECTION_METRICS = ("pr_auc", "roc_auc", "answer_auroc")


# ============================================================
# 1. HELPERS
# ============================================================

def save_json(data: dict, path: Path) -> None:
    """Save experiment results as JSON."""

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

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open("w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            indent=2,
            default=convert,
            allow_nan=False,
        )


def run_mode_name(
    full_data: bool,
    sample_size: int,
    sampling: str,
    unsupported_ratio: float,
    seed: int = RANDOM_SEED,
) -> str:
    """Name the data configuration of a tuning run.

    The main run uses seed 42 and keeps the old names. The extra
    seed runs (Field 5) add the seed, such as "full_seed43".
    """

    name = "full" if full_data else (
        f"{sampling}_n{sample_size}"
        f"_r{unsupported_ratio:.2f}"
    )

    if seed != RANDOM_SEED:
        name += f"_seed{seed}"

    return name


def ragbench_input_paths() -> dict:
    """List the RAGBench files used for training and validation."""

    return {
        f"{subset}_{split}": DATA_DIR / "RAGBench" / subset / f"{split}.csv"
        for subset in SUBSETS
        for split in ("train", "validation")
    }


# ============================================================
# 2. DATA
# ============================================================

def load_validation_data() -> pd.DataFrame:
    """Load official validation splits only."""

    frames = []

    for subset in SUBSETS:
        df = load_ragbench_claims(
            subset=subset,
            split="validation",
        )

        print(
            f"{subset}: {len(df)} validation claims"
        )

        frames.append(df)

    return pd.concat(
        frames,
        ignore_index=True,
    )


def map_labels(processed: pd.DataFrame) -> pd.Series:
    """Map Supported/Unsupported to 0/1."""

    y = processed["label"].map(LABEL_MAPPING)

    if y.isna().any():
        raise ValueError("Unknown gold labels detected.")

    return y.astype(int)


def check_finite(X: pd.DataFrame, name: str) -> None:
    if not np.isfinite(X.to_numpy()).all():
        raise ValueError(f"Invalid {name} features.")


def forced_uncertain_mask(df: pd.DataFrame) -> dict:
    """Claims labeled Uncertain whatever their score (Field 12).

    Returns the combined mask and its two parts:
    claims the shared normalizer cannot read, and claims with
    no cited passage.
    """

    claims = df["claim"].fillna("").astype(str).map(normalize_text)
    unreadable = normalizer_failure_mask(claims.tolist())

    no_passage = (
        df["evidence"].fillna("").astype(str).str.strip().eq("").to_numpy()
    )

    return {
        "mask": unreadable | no_passage,
        "normalizer_failures": int(unreadable.sum()),
        "normalizer_failure_rate": float(unreadable.mean()),
        "no_passage": int(no_passage.sum()),
    }


# ============================================================
# 4. ANSWER-LEVEL AUROC
# ============================================================

def answer_level_auroc(
    validation_df: pd.DataFrame,
    p_supported: np.ndarray,
    method: str = "min",
) -> dict:
    """Answer-level AUROC (card Field 6).

    Gold: an answer is Unsupported when adherence_score is False.
    method: the rule that combines claim scores ("min" or "mean").
    Returned keys: overall, plus one entry per subset.
    """

    answers = aggregate_answer_scores(
        validation_df,
        p_supported,
        method=method,
    )

    def auc(frame: pd.DataFrame):
        if frame["y_true"].nunique() < 2:
            return None
        return float(
            roc_auc_score(
                frame["y_true"],
                frame["answer_score"],
            )
        )

    result = {
        "method": method,
        "overall": auc(answers),
        "answers": int(len(answers)),
        "unsupported_answers": int(answers["y_true"].sum()),
    }

    for subset in SUBSETS:
        part = answers[answers["subset"] == subset]
        result[subset] = auc(part)

    return result


# ============================================================
# 5. TRIAL SEARCH
# ============================================================

def run_search(
    train_processed: pd.DataFrame,
    validation_processed: pd.DataFrame,
    validation_df: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    selection_metric: str,
    min_precision: float,
    max_uncertain_rate: float,
    forced_val: np.ndarray,
    seed: int = RANDOM_SEED,
):
    """Run SEARCH_TRIALS random trials; keep the best in memory."""

    trials = sample_trials(
        TFIDF_NGRAM_RANGE,
        TFIDF_MAX_FEATURES,
        LR_C_VALUES,
        max_trials=SEARCH_TRIALS,
        seed=seed,
    )

    print(
        f"\nTuning trials: {len(trials)} "
        f"(+ {len(ANSWER_AGGREGATIONS)} answer rules = "
        f"{len(trials) + len(ANSWER_AGGREGATIONS)}, maximum {MAX_TRIALS})"
    )

    # Slow, vectorizer-independent features: computed once per split.
    print("Computing vectorizer-independent features ...")
    start = time.perf_counter()
    static_train = build_static_features(train_processed)
    static_train_seconds = time.perf_counter() - start

    static_val = build_static_features(validation_processed)

    # Group trials so each (ngram, vocabulary) is fitted only once.
    groups: dict[tuple, list] = {}
    for trial_id, (ngram, vocab, c_value) in enumerate(trials, 1):
        groups.setdefault((ngram, vocab), []).append(
            (trial_id, c_value)
        )

    training_text = (
        train_processed["claim_clean"].tolist()
        + train_processed["evidence_clean"].tolist()
    )

    rows = []
    best = None

    for (ngram, vocab), members in groups.items():

        print(f"\n--- TF-IDF ngram={ngram} max_features={vocab} ---")

        vectorizer = create_tfidf_vectorizer(
            ngram_range=ngram,
            max_features=vocab,
        )
        group_start = time.perf_counter()
        vectorizer.fit(training_text)

        X_train = build_full_feature_matrix(
            train_processed,
            vectorizer,
            static=static_train,
        )
        # Training cost: TF-IDF fit and training features only.
        group_feature_seconds = time.perf_counter() - group_start

        X_val = build_full_feature_matrix(
            validation_processed,
            vectorizer,
            static=static_val,
        )

        check_finite(X_train, "training")
        check_finite(X_val, "validation")

        # Attach claim text after the numeric check.
        X_train = with_claim_text(X_train, train_processed)
        X_val = with_claim_text(X_val, validation_processed)

        for trial_id, c_value in members:

            fit_start = time.perf_counter()

            model = train_logistic_regression(
                X_train,
                y_train,
                C=c_value,
                ngram_range=ngram,
                max_features=vocab,
                random_state=seed,
            )

            fit_seconds = time.perf_counter() - fit_start

            unsupported_index = list(model.classes_).index(1)
            p_unsupported = model.predict_proba(X_val)[
                :, unsupported_index
            ]
            p_supported = 1.0 - p_unsupported

            binary = evaluate_binary(
                y_true=y_val,
                probabilities=p_unsupported,
                threshold=0.5,
            )

            answers = answer_level_auroc(
                validation_df,
                p_supported,
            )

            dual = select_thresholds_with_forced_uncertain(
                y_true=y_val,
                p_supported=p_supported,
                forced_uncertain=forced_val,
                min_precision=min_precision,
                max_uncertain_rate=max_uncertain_rate,
            )

            # Metrics over all validation claims, forced ones included.
            trial_labels = evaluate_three_label(
                y_true=y_val,
                predicted_labels=dual["labels"],
            )

            row = {
                "trial": trial_id,
                "ngram_range": str(ngram),
                "max_features": vocab,
                "C": c_value,
                "roc_auc": binary["roc_auc"],
                "pr_auc": binary["pr_auc"],
                "answer_auroc": answers["overall"],
                "answer_auroc_FinQA": answers.get("FinQA"),
                "answer_auroc_TATQA": answers.get("TATQA"),
                "dual_feasible": dual["feasible"],
                "strict_recall": trial_labels["strict_recall"],
                "unsupported_precision": trial_labels[
                    "unsupported_precision"
                ],
                "uncertain_rate": trial_labels["uncertain_rate"],
            }
            rows.append(row)

            score = row[selection_metric]
            score = -1.0 if score is None else float(score)

            print(
                f"trial {trial_id:>2} C={c_value:<5} "
                f"roc={row['roc_auc']:.4f} pr={row['pr_auc']:.4f} "
                f"ans_auroc={row['answer_auroc']} "
                f"feasible={row['dual_feasible']}"
            )

            # Ties on the selection metric: higher ROC AUC wins.
            key = (score, row["roc_auc"])

            if best is None or key > best["key"]:
                best = {
                    "key": key,
                    "row": row,
                    "model": model,
                    "vectorizer": vectorizer,
                    "X_val": X_val,
                    "X_train_shape": X_train.shape,
                    "training_seconds": {
                        "static_features": static_train_seconds,
                        "tfidf_and_features": group_feature_seconds,
                        "logistic_regression_fit": fit_seconds,
                        "total": static_train_seconds
                        + group_feature_seconds
                        + fit_seconds,
                    },
                    "ngram": ngram,
                    "max_features": vocab,
                    "C": c_value,
                }

    return pd.DataFrame(rows), best


# ============================================================
# 6. FULL EVALUATION OF THE BEST TRIAL
# ============================================================

def evaluate_best(
    best: dict,
    validation_df: pd.DataFrame,
    y_val: pd.Series,
    mode: str,
    target_recall: float,
    min_precision: float,
    max_uncertain_rate: float,
    forced: dict,
    seed: int = RANDOM_SEED,
) -> dict:
    """Calibration, thresholds, three-label metrics and saved model."""

    model = best["model"]
    vectorizer = best["vectorizer"]
    X_val = best["X_val"]

    name = f"{mode}_best"

    model_path = MODEL_DIR / f"classical_model_{name}.joblib"

    save_model(
        model=model,
        vectorizer=vectorizer,
        path=model_path,
    )

    print(f"\nSaved best model: {model_path}")

    unsupported_index = list(model.classes_).index(1)

    predict_start = time.perf_counter()
    p_unsupported = model.predict_proba(X_val)[:, unsupported_index]
    model_ms_per_claim = (
        (time.perf_counter() - predict_start) * 1000 / max(len(X_val), 1)
    )

    p_supported = 1.0 - p_unsupported

    binary_metrics = evaluate_binary(
        y_true=y_val,
        probabilities=p_unsupported,
        threshold=0.5,
    )

    binary_threshold = select_threshold(
        y_true=y_val,
        probabilities=p_unsupported,
        target_recall=target_recall,
    )

    calibration = evaluate_calibration(
        y_true=y_val,
        probabilities=p_unsupported,
        n_bins=DEFAULT_CALIBRATION_BINS,
    )

    calibration["table"].to_csv(
        OUTPUT_DIR / f"calibration_{name}.csv",
        index=False,
    )

    dual_threshold = select_thresholds_with_forced_uncertain(
        y_true=y_val,
        p_supported=p_supported,
        forced_uncertain=forced["mask"],
        min_precision=min_precision,
        max_uncertain_rate=max_uncertain_rate,
    )

    predicted_labels = dual_threshold.pop("labels")

    three_label_metrics = evaluate_three_label(
        y_true=y_val,
        predicted_labels=predicted_labels,
    )

    # Check the labels against the rule inference uses.
    if dual_threshold["feasible"]:
        expected = np.array(
            [
                "Uncertain" if is_forced else classify_support_probability(
                    float(p),
                    dual_threshold["lower_threshold"],
                    dual_threshold["upper_threshold"],
                )
                for p, is_forced in zip(p_supported, forced["mask"])
            ],
            dtype=object,
        )

        if not np.array_equal(expected, predicted_labels):
            raise ValueError("Threshold labels differ from inference.")

        if not np.isclose(
            dual_threshold["unsupported_precision"],
            three_label_metrics["unsupported_precision"],
        ):
            raise ValueError("Threshold mismatch: unsupported_precision")

    elif not np.all(predicted_labels == "Uncertain"):
        raise ValueError("Invalid all-Uncertain fallback.")

    # Field 6: precision at recall 0.80, baseline definition.
    p_at_r80 = precision_at_recall(y_val, p_unsupported, target_recall)

    # Field 6: answer rule chosen on validation, then answer AUROC.
    aggregation = select_answer_aggregation(validation_df, p_supported)
    answers = answer_level_auroc(
        validation_df,
        p_supported,
        method=aggregation["chosen_method"],
    )

    coefficients = coefficient_table(model)

    coefficients.to_csv(
        OUTPUT_DIR / f"coefficients_{name}.csv",
        index=False,
    )

    prediction_table = pd.DataFrame({
        "subset": validation_df["subset"].to_numpy(),
        "answer_row": validation_df["answer_row"].to_numpy(),
        "true_label": y_val.map({
            0: "Supported",
            1: "Unsupported",
        }).to_numpy(),
        "p_supported": p_supported,
        "p_unsupported": p_unsupported,
        "predicted_label": predicted_labels,
    })

    prediction_table.to_csv(
        OUTPUT_DIR / f"predictions_{name}.csv",
        index=False,
    )

    detail = {
        "experiment": name,
        "random_seed": seed,
        "best_config": {
            "ngram_range": list(best["ngram"]),
            "max_features": best["max_features"],
            "C": best["C"],
        },
        "validation_claims": len(X_val),
        "binary_metrics": binary_metrics,
        "binary_threshold": {
            "threshold": binary_threshold["threshold"],
            "precision": binary_threshold["precision"],
            "recall": binary_threshold["recall"],
            "f1": binary_threshold["f1"],
            "target_recall": target_recall,
        },
        "calibration": {
            "brier_score": calibration["brier_score"],
            "ece": calibration["ece"],
            "n_bins": DEFAULT_CALIBRATION_BINS,
        },
        "precision_at_recall_0.80": p_at_r80,
        "answer_aggregation": aggregation,
        "answer_level_auroc": answers,
        "forced_uncertain": {
            key: value for key, value in forced.items() if key != "mask"
        },
        "dual_threshold": dual_threshold,
        "cost": {
            "training_seconds": best["training_seconds"],
            "model_only_ms_per_claim": model_ms_per_claim,
            "note": (
                "Training seconds cover the training features, the TF-IDF "
                "fit and the best trial's fit on CPU. End-to-end latency "
                "with feature building is measured by final_test.py."
            ),
        },
        "three_label_metrics": three_label_metrics,
        "fallback_all_uncertain": not dual_threshold["feasible"],
    }

    save_json(detail, OUTPUT_DIR / f"evaluation_{name}.json")

    # Save the thresholds and the run record next to the model.
    # Inference reads the thresholds from this file.
    metadata = build_metadata(
        model_path=model_path,
        input_paths=ragbench_input_paths(),
        project_dir=PROJECT_DIR,
        details={
            "experiment": name,
            "random_seed": seed,
            "shared_components": shared_versions(),
            "hardware": hardware_info(),
            "search_trials": SEARCH_TRIALS,
            "training_seconds": best["training_seconds"],
            "answer_aggregation_trials": len(ANSWER_AGGREGATIONS),
            "card_version": not (USE_EXTRA_FEATURES or USE_CLAIM_TEXT),
            "use_extra_features": USE_EXTRA_FEATURES,
            "use_claim_text": USE_CLAIM_TEXT,
            "numeric_features": model_numeric_columns(),
            "model_config": {
                key: value for key, value in MODEL_CONFIG.items()
            },
            "best_config": detail["best_config"],
            "train_claims": int(best["X_train_shape"][0]),
            "validation_claims": int(len(X_val)),
            "thresholds": {
                "feasible": bool(dual_threshold["feasible"]),
                "lower_threshold": dual_threshold["lower_threshold"],
                "upper_threshold": dual_threshold["upper_threshold"],
                "min_precision": min_precision,
                "max_uncertain_rate": max_uncertain_rate,
                "upper_rule": "fill Uncertain up to max_uncertain_rate",
                "forced_uncertain": "normalizer cannot read, or no passage",
            },
            "answer_aggregation": aggregation["chosen_method"],
            "validation_metrics": {
                "roc_auc": binary_metrics["roc_auc"],
                "pr_auc": binary_metrics["pr_auc"],
                "answer_auroc": answers["overall"],
                "precision_at_recall_0.80": p_at_r80["precision"],
                "normalizer_failure_rate": forced["normalizer_failure_rate"],
                "ece": calibration["ece"],
                "strict_recall": three_label_metrics["strict_recall"],
                "unsupported_precision": three_label_metrics[
                    "unsupported_precision"
                ],
                "uncertain_rate": three_label_metrics["uncertain_rate"],
            },
        },
    )

    metadata_path = MODEL_DIR / f"classical_model_{name}.json"

    save_json(metadata, metadata_path)

    print(f"Saved run record and thresholds: {metadata_path}")

    print("\n========== BEST TRIAL ==========")
    print(f"Config: ngram={best['ngram']} "
          f"max_features={best['max_features']} C={best['C']}")
    print(f"Claim ROC AUC: {binary_metrics['roc_auc']:.4f}")
    print(f"Claim PR AUC:  {binary_metrics['pr_auc']:.4f}")
    print(f"Answer AUROC:  {answers['overall']}  "
          f"FinQA={answers.get('FinQA')}  TATQA={answers.get('TATQA')}")
    print(f"ECE: {calibration['ece']:.4f}")
    print(f"Dual feasible: {dual_threshold['feasible']}")
    print(f"Strict recall: {three_label_metrics['strict_recall']:.4f}")
    print("Unsupported precision: "
          f"{three_label_metrics['unsupported_precision']:.4f}")
    print(f"Uncertain rate: {three_label_metrics['uncertain_rate']:.4f}")
    print(f"Lenient recall: {three_label_metrics['lenient_recall']:.4f}")
    print(f"Precision at recall 0.80: {p_at_r80['precision']:.4f}")
    print(f"Forced Uncertain: {forced['normalizer_failures']} unreadable, "
          f"{forced['no_passage']} no passage "
          f"(AC2 satisfiable: {dual_threshold['ac2_satisfiable']})")
    print(f"Answer rule: {aggregation['chosen_method']} "
          f"{aggregation['validation_answer_auroc']}  "
          f"gold disagreements: {aggregation['gold_disagreements']}")
    numeric = coefficients[coefficients["group"] == "numeric"]

    print("\nNumeric feature coefficients (standardized):")
    print(
        numeric[["feature", "coefficient"]]
        .round(4)
        .to_string(index=False)
    )

    text = coefficients[coefficients["group"] == "claim_text"]

    if not text.empty:
        ranked = text.sort_values("coefficient")

        print("\nClaim n-grams toward Unsupported:")
        print(
            ranked.tail(15)[::-1][["feature", "coefficient"]]
            .round(3)
            .to_string(index=False)
        )

        print("\nClaim n-grams toward Supported:")
        print(
            ranked.head(15)[["feature", "coefficient"]]
            .round(3)
            .to_string(index=False)
        )

    return detail


# ============================================================
# 7. MAIN EXPERIMENT
# ============================================================

def run_tuning(
    sample_size: int = 1000,
    sampling: str = "stratified",
    unsupported_ratio: float = 0.5,
    target_recall: float = DEFAULT_TARGET_RECALL,
    full_data: bool = False,
    min_precision: float = DEFAULT_MIN_PRECISION,
    max_uncertain_rate: float = DEFAULT_MAX_UNCERTAIN_RATE,
    selection_metric: str = "pr_auc",
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Search, then evaluate the best trial on validation data."""

    if selection_metric not in SELECTION_METRICS:
        raise ValueError(
            f"selection_metric must be one of {SELECTION_METRICS}"
        )

    print("========== TUNING (TF-IDF + LOGISTIC REGRESSION) ==========")
    print(f"Random seed: {seed}")
    print(f"Full training data: {full_data}")
    print(f"Selection metric: {selection_metric}")

    train_df = load_training_data(
        sample_size=sample_size,
        sampling=sampling,
        unsupported_ratio=unsupported_ratio,
        full_data=full_data,
    )

    validation_df = load_validation_data()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    train_processed = preprocess_claims(train_df)
    validation_processed = preprocess_claims(validation_df)

    y_train = map_labels(train_processed)
    y_val = map_labels(validation_processed)

    forced = forced_uncertain_mask(validation_df)

    trials_df, best = run_search(
        train_processed=train_processed,
        validation_processed=validation_processed,
        validation_df=validation_df,
        y_train=y_train,
        y_val=y_val,
        selection_metric=selection_metric,
        min_precision=min_precision,
        max_uncertain_rate=max_uncertain_rate,
        forced_val=forced["mask"],
        seed=seed,
    )

    mode = run_mode_name(
        full_data,
        sample_size,
        sampling,
        unsupported_ratio,
        seed=seed,
    )

    trials_path = OUTPUT_DIR / f"tuning_trials_{mode}.csv"
    trials_df.to_csv(trials_path, index=False)

    print("\n========== ALL TRIALS ==========")
    print(
        trials_df.sort_values(selection_metric, ascending=False)
        .round(4)
        .to_string(index=False)
    )
    print(f"\nSaved trials: {trials_path}")

    evaluate_best(
        best=best,
        validation_df=validation_df,
        y_val=y_val,
        mode=mode,
        target_recall=target_recall,
        min_precision=min_precision,
        max_uncertain_rate=max_uncertain_rate,
        forced=forced,
        seed=seed,
    )

    return trials_df


# ============================================================
# 8. COMMAND LINE INTERFACE
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tune TF-IDF + Logistic Regression (20 trials in total)."
    )

    parser.add_argument("--sample-size", type=int, default=1000)

    parser.add_argument(
        "--sampling",
        choices=["random", "stratified"],
        default="stratified",
    )

    parser.add_argument("--unsupported-ratio", type=float, default=0.5)

    parser.add_argument(
        "--target-recall",
        type=float,
        default=DEFAULT_TARGET_RECALL,
    )

    parser.add_argument(
        "--full-data",
        action="store_true",
        help="Use all official training claims.",
    )

    parser.add_argument(
        "--min-precision",
        type=float,
        default=DEFAULT_MIN_PRECISION,
    )

    parser.add_argument(
        "--max-uncertain-rate",
        type=float,
        default=DEFAULT_MAX_UNCERTAIN_RATE,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help="Trial sampling and solver seed. 42 is the main run; "
             "43 and 44 are the extra seed runs (Field 5).",
    )

    parser.add_argument(
        "--selection-metric",
        choices=SELECTION_METRICS,
        default="pr_auc",
    )

    args = parser.parse_args()

    run_tuning(
        sample_size=args.sample_size,
        sampling=args.sampling,
        unsupported_ratio=args.unsupported_ratio,
        target_recall=args.target_recall,
        full_data=args.full_data,
        min_precision=args.min_precision,
        max_uncertain_rate=args.max_uncertain_rate,
        selection_metric=args.selection_metric,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

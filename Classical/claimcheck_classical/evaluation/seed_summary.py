"""
ClaimCheck Classical Model - Seed Summary (Experiment Card, Field 5)

Reads the validation results of the three tuning runs (seeds 42, 43
and 44) and reports the mean and the standard deviation of each metric.

Logistic Regression gives the same result for the same data, so the
seed changes which trials are sampled and the solver seed. Seed 42 is
the main model; the other seeds only measure the spread.

    python -m claimcheck_classical.evaluation.seed_summary
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

CLASSICAL_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = CLASSICAL_DIR / "outputs"
MODEL_DIR = CLASSICAL_DIR / "models"

SEEDS = (42, 43, 44)
MAIN_SEED = 42


def run_name(seed: int) -> str:
    """Name used by tune.py for a full-data run with this seed."""

    return "full_best" if seed == MAIN_SEED else f"full_seed{seed}_best"


def evaluation_path(seed: int) -> Path:
    return OUTPUT_DIR / f"evaluation_{run_name(seed)}.json"


def metadata_path(seed: int) -> Path:
    return MODEL_DIR / f"classical_model_{run_name(seed)}.json"


def read_row(seed: int) -> dict:
    """Main validation metrics of one seed run."""

    detail = json.loads(evaluation_path(seed).read_text(encoding="utf-8"))

    config = detail["best_config"]
    answers = detail["answer_level_auroc"]
    three = detail["three_label_metrics"]

    return {
        "seed": seed,
        "ngram_range": str(tuple(config["ngram_range"])),
        "max_features": config["max_features"],
        "C": config["C"],
        "answer_rule": detail["answer_aggregation"]["chosen_method"],
        "roc_auc": detail["binary_metrics"]["roc_auc"],
        "pr_auc": detail["binary_metrics"]["pr_auc"],
        "answer_auroc": answers["overall"],
        "answer_auroc_FinQA": answers.get("FinQA"),
        "answer_auroc_TATQA": answers.get("TATQA"),
        "precision_at_recall_0.80": detail["precision_at_recall_0.80"]["precision"],
        "ece": detail["calibration"]["ece"],
        "strict_recall": three["strict_recall"],
        "unsupported_precision": three["unsupported_precision"],
        "lenient_recall": three["lenient_recall"],
        "uncertain_rate": three["uncertain_rate"],
        "training_seconds": detail.get("cost", {})
        .get("training_seconds", {})
        .get("total"),
    }


METRICS = [
    "roc_auc", "pr_auc", "answer_auroc", "answer_auroc_FinQA",
    "answer_auroc_TATQA", "precision_at_recall_0.80", "ece",
    "strict_recall", "unsupported_precision", "lenient_recall",
    "uncertain_rate", "training_seconds",
]


def summarise(rows: list[dict]) -> pd.DataFrame:
    """Mean and sample standard deviation of each metric."""

    table = pd.DataFrame(rows)

    stats = pd.DataFrame({
        "mean": table[METRICS].astype(float).mean(),
        "std": table[METRICS].astype(float).std(ddof=1),
    })

    return stats


def main() -> None:
    missing = [seed for seed in SEEDS if not evaluation_path(seed).is_file()]

    if missing:
        raise SystemExit(
            f"Missing results for seeds {missing}. Run run_seeds.sh first."
        )

    rows = [read_row(seed) for seed in SEEDS]
    per_seed = pd.DataFrame(rows)
    stats = summarise(rows)

    per_seed.to_csv(OUTPUT_DIR / "seed_runs_validation.csv", index=False)
    stats.to_csv(OUTPUT_DIR / "seed_summary_validation.csv")

    print("========== VALIDATION RESULTS FOR EACH SEED ==========")
    print(per_seed.round(4).to_string(index=False))

    print("\n========== MEAN AND STANDARD DEVIATION (3 SEEDS) ==========")
    print(stats.round(4).to_string())

    same_config = per_seed[["ngram_range", "max_features", "C"]].nunique().max() == 1
    print(f"\nSame best configuration for all seeds: {same_config}")
    print(f"Saved: {OUTPUT_DIR / 'seed_summary_validation.csv'}")


if __name__ == "__main__":
    main()

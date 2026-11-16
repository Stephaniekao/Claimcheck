"""
ClaimCheck Classical Model - Inserted Error Set v2 Validation

Evaluates the frozen model saved by tune.py on Inserted Error Set v2
train and validation, overall and on the Storyboard B slice.

Experiment Card rules followed here:
* Field 4 / team decision 1: the thresholds were chosen on RAGBench
  validation and are only applied here, never refitted.
* Field 4: the test split is never loaded. Only train and validation
  are accepted; the test split is for final_test.py, run once.
* Field 12: passages come from the team's shared evidence.py, and
  numbers from the shared normalizer.
* Fields 6, 7, 8: strict and lenient recall, precision, F1, false
  alarms, Uncertain rate, AUROC, precision at recall 0.80, ECE (10
  equal-width bins), per error type, and the flag-everything reference.
  Label metrics use the baseline's metrics.py, so both rows of the
  comparison table are computed the same way.

The train split was used to develop the scope keywords ("entire"),
so its numbers are development results, not an independent check.

    python -m claimcheck_classical.evaluation.inserted_validation
"""

from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from claimcheck_classical.evaluation.calibration import evaluate_calibration
from claimcheck_classical.inference.predict import (
    load_frozen_model,
    predict_claims,
)
from claimcheck_classical.provenance import (
    git_info,
    hardware_info,
    library_versions,
    sha256_file,
)
from claimcheck_classical import shared


# ============================================================
# 1. CONFIGURATION
# ============================================================

CLASSICAL_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = CLASSICAL_DIR / "outputs" / "inserted_error_set"
DEFAULT_METADATA = CLASSICAL_DIR / "models" / "classical_model_full_best.json"

ALLOWED_SPLITS = ("train", "validation")
STORYBOARD_B = "storyboard_b"
CALIBRATION_BINS = 10


# ============================================================
# 2. PREDICTION
# ============================================================

def predict_split(data: pd.DataFrame, metadata_path: Path) -> tuple[pd.DataFrame, float]:
    """Label one split with the frozen model and saved thresholds.

    The model reads claim and evidence only. Labels and error types
    are joined back afterwards for evaluation.
    """

    model, vectorizer, thresholds = load_frozen_model(metadata_path)

    inputs = data[["claim", "evidence"]].reset_index(drop=True)

    start = time.perf_counter()

    predicted = predict_claims(
        claims=inputs,
        model=model,
        vectorizer=vectorizer,
        lower_threshold=thresholds.get("lower_threshold"),
        upper_threshold=thresholds.get("upper_threshold"),
        all_uncertain=not thresholds["feasible"],
    )

    ms_per_claim = (time.perf_counter() - start) * 1000 / max(len(inputs), 1)

    result = data.reset_index(drop=True).copy()

    for column in predicted.columns:
        if column not in ("claim", "evidence"):
            result[column] = predicted[column].to_numpy()

    return result, ms_per_claim


# ============================================================
# 3. METRICS
# ============================================================

def score_block(res: pd.DataFrame) -> dict:
    """Score-based metrics on P(Unsupported): AUROC, AP, P@R0.80, ECE."""

    scored = res[np.isfinite(res["p_unsupported"].to_numpy(dtype=float))]

    block = {
        "n_scored": int(len(scored)),
        "n_without_score": int(len(res) - len(scored)),
    }

    if scored.empty or scored["label"].nunique() < 2:
        block["note"] = "AUROC and ECE need both labels among scored claims."
        return block

    gold = scored["label"].reset_index(drop=True)
    risk = scored["p_unsupported"].astype(float).reset_index(drop=True)

    block.update({
        "auroc": shared.shared_metrics.auroc(gold, risk),
        "average_precision": shared.shared_metrics.average_precision(gold, risk),
        "precision_at_recall_0.80": shared.shared_metrics.precision_at_recall(gold, risk, 0.80),
        "base_rate": float(gold.eq("Unsupported").mean()),
        "n_distinct_scores": int(risk.nunique()),
    })

    calibration = evaluate_calibration(
        y_true=gold.eq("Unsupported").astype(int).to_numpy(),
        probabilities=risk.to_numpy(),
        n_bins=CALIBRATION_BINS,
    )

    block["ece"] = calibration["ece"]
    block["brier_score"] = calibration["brier_score"]
    block["calibration_table"] = calibration["table"].to_dict(orient="records")

    return block


def summarise(res: pd.DataFrame) -> dict:
    """All Field 6-8 metrics for one split, overall and Storyboard B."""

    m = shared.shared_metrics
    pred = res["predicted_label"]

    storyboard = res[res["subset"] == STORYBOARD_B]
    general = res[res["subset"] != STORYBOARD_B]

    overall = m.binary_metrics(res["label"], pred)
    storyboard_metrics = (
        m.binary_metrics(storyboard["label"], storyboard["predicted_label"])
        if len(storyboard) else None
    )

    flag_all = m.binary_metrics(
        res["label"],
        pd.Series("Unsupported", index=res.index),
    )

    reasons = res["reason"].replace("", "scored").value_counts().to_dict()

    return {
        "overall": overall,
        "general_claims": (
            m.binary_metrics(general["label"], general["predicted_label"])
            if len(general) else None
        ),
        "storyboard_b": storyboard_metrics,
        "reference_flag_everything": flag_all,
        "targets_met": m.meets_targets(overall, storyboard_metrics),
        "scores_overall": score_block(res),
        "scores_storyboard_b": score_block(storyboard) if len(storyboard) else None,
        "per_error_type": (
            m.per_error_type(res, pred_col="predicted_label")
            .round(3)
            .reset_index()
            .to_dict(orient="records")
        ),
        "label_reasons": {str(k): int(v) for k, v in reasons.items()},
    }


# ============================================================
# 4. OUTPUT
# ============================================================

def _clean(value):
    """Make a result JSON-safe: NaN and infinity become null."""

    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def print_split(split: str, s: dict, ms: float) -> None:
    o, sb = s["overall"], s["storyboard_b"]
    sc = s["scores_overall"]

    print(f"\n=== {split} (n={o['n']}, {o['n_unsupported_gold']} wrong claims) ===")
    print(
        f"overall      strict R {o['strict_recall']:.3f}  lenient R {o['lenient_recall']:.3f}  "
        f"P {o['precision']:.3f}  F1 {o['f1']:.3f}  false alarms {o['false_alarm_rate']:.3f}  "
        f"uncertain {o['uncertain_rate']:.3f}"
    )

    if sb:
        print(
            f"storyboard B strict R {sb['strict_recall']:.3f}  lenient R {sb['lenient_recall']:.3f}  "
            f"P {sb['precision']:.3f}  false alarms {sb['false_alarm_rate']:.3f}  "
            f"uncertain {sb['uncertain_rate']:.3f}"
        )

    if "auroc" in sc:
        print(
            f"scores       AUROC {sc['auroc']:.3f}  AP {sc['average_precision']:.3f} "
            f"(base rate {sc['base_rate']:.3f})  "
            f"P@R0.80 {sc['precision_at_recall_0.80']['precision']:.3f}  ECE {sc['ece']:.3f}"
        )

    ssb = s["scores_storyboard_b"]
    if ssb and "auroc" in ssb:
        print(f"storyboard B AUROC {ssb['auroc']:.3f}  ECE {ssb['ece']:.3f}")

    fa = s["reference_flag_everything"]
    print(f"flag-everything reference: R 1.000  P {fa['precision']:.3f}  F1 {fa['f1']:.3f}")
    print("label reasons:", s["label_reasons"])

    table = pd.DataFrame(s["per_error_type"]).set_index("error_type")
    print(table.to_string())
    print("targets:", s["targets_met"], f"| {ms:.2f} ms/claim")


# ============================================================
# 5. MAIN
# ============================================================

def run(
    metadata_path: Path = DEFAULT_METADATA,
    splits: tuple[str, ...] = ALLOWED_SPLITS,
    output_dir: Path = OUTPUT_DIR,
) -> dict:
    """Evaluate the frozen model on Inserted Error Set train/validation."""

    for split in splits:
        if split not in ALLOWED_SPLITS:
            raise ValueError(
                f"Split '{split}' is not allowed here. The test split is "
                "evaluated once by final_test.py."
            )

    metadata_path = Path(metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    output_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "method": "TF-IDF + Logistic Regression (Classical)",
        "run_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "note": (
            "Thresholds were chosen on RAGBench validation and are only "
            "applied here. The train split was used to develop the scope "
            "keywords, so train results are development results."
        ),
        "model": {
            "metadata_file": metadata_path.name,
            "model_sha256": metadata.get("model_sha256"),
            "best_config": metadata.get("best_config"),
            "thresholds": metadata.get("thresholds"),
        },
        "shared_components": shared.shared_versions(),
        "inserted_input_sha256": shared.inserted_input_hashes(),
        "git": git_info(CLASSICAL_DIR),
        "libraries": library_versions(),
        "hardware": hardware_info(),
        "splits": {},
    }

    # The shared loader returns every split; only the requested
    # train/validation rows are ever passed to the model.
    data = shared.load_inserted_error_set()

    for split in splits:
        part = data[data["split"] == split]

        if part.empty:
            raise ValueError(f"No Inserted Error Set claims in split '{split}'.")

        res, ms = predict_split(part, metadata_path)

        keep = [
            "claim_id", "subset", "split", "error_type", "label",
            "predicted_label", "reason", "p_supported", "p_unsupported",
            "numerical_consistency", "period_consistency", "scope_consistency",
            "numerical_missing", "period_missing", "scope_missing",
            "cosine_similarity", "claim", "evidence",
        ]
        res[[c for c in keep if c in res.columns]].to_csv(
            output_dir / f"predictions_{split}.csv",
            index=False,
        )

        summary = summarise(res)
        summary["latency_ms_per_claim"] = round(ms, 3)
        report["splits"][split] = summary

        print_split(split, summary, ms)

    out_path = output_dir / "inserted_validation.json"
    out_path.write_text(
        json.dumps(_clean(report), indent=2, allow_nan=False),
        encoding="utf-8",
    )

    print(f"\nWrote {out_path}")

    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the frozen Classical model on Inserted Error Set v2 "
                    "train and validation (never test)."
    )
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=ALLOWED_SPLITS,
        default=list(ALLOWED_SPLITS),
    )
    args = parser.parse_args()

    run(metadata_path=args.metadata, splits=tuple(args.splits))


if __name__ == "__main__":
    main()

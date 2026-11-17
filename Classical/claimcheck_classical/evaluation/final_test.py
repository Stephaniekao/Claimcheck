"""
ClaimCheck Classical Model - Final Test (run once)

Experiment Card, Field 5: the test set is used only once. This script
evaluates the frozen models on the RAGBench test splits and the
Inserted Error Set v2 test split, then writes a lock file. If the lock
file exists, the script refuses to run again.

Nothing is chosen here. The models, the thresholds and the answer rule
were all chosen on validation and are read from models/.

* Seed 42 is the main model. Seeds 43 and 44 are evaluated in the same
  run only to report the spread across seeds (Field 5).
* RAGBench test: claim and answer AUROC, precision at recall 0.80,
  three labels, ECE (Fields 6, 8).
* Inserted Error Set v2 test and the Storyboard B slice: strict recall,
  precision, ECE (Fields 6, 7).
* Reliability diagrams on the test split and the Storyboard B slice
  (Field 8), for the main model.
* Field 10 targets for the main model.
* Field 9 samples for the error analysis: every Storyboard B error,
  30 missed errors and 20 false alarms per RAGBench subset (seed 42).

    python -m claimcheck_classical.evaluation.final_test --confirm
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from claimcheck_classical import shared
from claimcheck_classical.config import DATA_DIR
from claimcheck_classical.data_loader import load_ragbench_claims
from claimcheck_classical.evaluation.calibration import evaluate_calibration
from claimcheck_classical.evaluation.inserted_validation import (
    _clean,
    predict_split,
    summarise as summarise_inserted,
)
from claimcheck_classical.evaluation.metrics import (
    evaluate_binary,
    evaluate_three_label,
)
from claimcheck_classical.evaluation.seed_summary import (
    MAIN_SEED,
    SEEDS,
    metadata_path,
)
from claimcheck_classical.features.evidence_features import (
    aggregate_answer_scores,
)
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


# ============================================================
# 1. CONFIGURATION
# ============================================================

CLASSICAL_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = CLASSICAL_DIR / "outputs" / "final_test"
LOCK_PATH = OUTPUT_DIR / "final_test_lock.json"

SUBSETS = ("FinQA", "TATQA")
SAMPLE_SEED = 42
MISSED_SAMPLE = 30
FALSE_ALARM_SAMPLE = 20

TARGETS = {
    "strict_recall": 0.80,
    "precision": 0.60,
    "storyboard_b_strict_recall": 0.80,
    "answer_auroc": 0.75,
    "max_ece": 0.10,
}


# ============================================================
# 2. LOCK
# ============================================================

def check_lock() -> None:
    """Stop if the test split was already used."""

    if LOCK_PATH.exists():
        lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
        raise SystemExit(
            "The final test was already run "
            f"({lock.get('status')}, {lock.get('started_utc')}). "
            "The Experiment Card allows only one test run (Field 5). "
            f"The lock file is {LOCK_PATH}."
        )


def write_lock(data: dict) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOCK_PATH.write_text(json.dumps(_clean(data), indent=2), encoding="utf-8")


# ============================================================
# 3. RAGBENCH TEST
# ============================================================

def load_ragbench_test() -> pd.DataFrame:
    frames = [load_ragbench_claims(subset=s, split="test") for s in SUBSETS]
    return pd.concat(frames, ignore_index=True)


def ragbench_test_files() -> dict:
    paths = {}
    for subset in SUBSETS:
        for suffix in ("csv", "parquet"):
            path = DATA_DIR / "RAGBench" / subset / f"test.{suffix}"
            if path.is_file():
                paths[f"{subset}_test.{suffix}"] = sha256_file(path)
    return paths


def evaluate_ragbench(test_df: pd.DataFrame, meta_path: Path) -> tuple[dict, pd.DataFrame]:
    """Score RAGBench test with one frozen model and its thresholds."""

    model, vectorizer, thresholds = load_frozen_model(meta_path)
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    answer_rule = metadata.get("answer_aggregation", "min")

    start = time.perf_counter()

    predicted = predict_claims(
        claims=test_df[["claim", "evidence"]],
        model=model,
        vectorizer=vectorizer,
        lower_threshold=thresholds.get("lower_threshold"),
        upper_threshold=thresholds.get("upper_threshold"),
        all_uncertain=not thresholds["feasible"],
    )

    ms_per_claim = (time.perf_counter() - start) * 1000 / max(len(test_df), 1)

    y = (test_df["label"] == "Unsupported").astype(int).to_numpy()
    p_unsupported = predicted["p_unsupported"].to_numpy(dtype=float)
    p_supported = predicted["p_supported"].to_numpy(dtype=float)
    labels = predicted["predicted_label"].to_numpy()

    scored = np.isfinite(p_unsupported)

    result = {
        "answer_rule": answer_rule,
        "thresholds": thresholds,
        "end_to_end_ms_per_claim": ms_per_claim,
        "n_claims": int(len(test_df)),
        "n_without_score": int((~scored).sum()),
        "label_reasons": (
            predicted["reason"].replace("", "scored").value_counts().to_dict()
        ),
        "claim_level": evaluate_binary(y[scored], p_unsupported[scored], 0.5),
        "precision_at_recall_0.80": shared.precision_at_recall(
            y[scored], p_unsupported[scored], 0.80
        ),
        "three_label": evaluate_three_label(y, labels),
    }

    calibration = evaluate_calibration(y[scored], p_unsupported[scored], 10)
    result["ece"] = calibration["ece"]
    result["brier_score"] = calibration["brier_score"]
    result["calibration_table"] = calibration["table"].to_dict(orient="records")

    # Answer level (Field 6): gold adherence_score, rule from validation.
    answers = aggregate_answer_scores(test_df, p_supported, method=answer_rule)

    def auc(frame: pd.DataFrame):
        if frame["y_true"].nunique() < 2:
            return None
        return shared.shared_metrics.auroc(
            frame["y_true"].map({1: "Unsupported", 0: "Supported"}).reset_index(drop=True),
            frame["answer_score"].reset_index(drop=True),
        )

    result["answer_auroc"] = {
        "overall": auc(answers),
        **{s: auc(answers[answers["subset"] == s]) for s in SUBSETS},
        "n_answers": int(len(answers)),
        "n_unsupported_answers": int(answers["y_true"].sum()),
    }

    # Per subset claim metrics, as the baseline reports them.
    result["per_subset"] = {}
    for subset in SUBSETS:
        part = (test_df["subset"] == subset).to_numpy()
        ok = part & scored
        result["per_subset"][subset] = {
            "claim_level": evaluate_binary(y[ok], p_unsupported[ok], 0.5),
            "three_label": evaluate_three_label(y[part], labels[part]),
        }

    table = test_df[["subset", "answer_row", "answer_id", "sentence_key", "claim", "label"]].copy()
    for column in predicted.columns:
        if column not in ("claim", "evidence"):
            table[column] = predicted[column].to_numpy()

    return result, table


# ============================================================
# 4. RELIABILITY DIAGRAMS (FIELD 8)
# ============================================================

def reliability_diagram(calibration_table: list[dict], title: str, path: Path) -> bool:
    """Draw predicted against observed Unsupported rate per bin."""

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; reliability diagram skipped.")
        return False

    table = pd.DataFrame(calibration_table)
    table = table[table["count"] > 0]

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot([0, 1], [0, 1], linestyle=":", color="gray", label="Perfect calibration")
    ax.plot(table["mean_predicted"], table["observed_rate"], marker="o", label="Model")

    for _, row in table.iterrows():
        ax.annotate(
            str(int(row["count"])),
            (row["mean_predicted"], row["observed_rate"]),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=7,
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted P(Unsupported)")
    ax.set_ylabel("Observed share of Unsupported claims")
    ax.set_title(title, fontsize=10)
    ax.legend(loc="upper left", fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

    return True


# ============================================================
# 5. FIELD 9 SAMPLES
# ============================================================

def add_review_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["reviewer_1_category"] = ""
    df["reviewer_2_category"] = ""
    return df


def export_error_samples(ragbench: pd.DataFrame, inserted: pd.DataFrame) -> dict:
    """Error samples from the main model, for the team's manual review."""

    folder = OUTPUT_DIR / "error_analysis"
    folder.mkdir(parents=True, exist_ok=True)

    counts = {}

    for subset in SUBSETS:
        part = ragbench[ragbench["subset"] == subset]

        missed = part[(part["label"] == "Unsupported") & (part["predicted_label"] == "Supported")]
        false_alarm = part[(part["label"] == "Supported") & (part["predicted_label"] == "Unsupported")]

        for name, pool, size in (
            ("missed_30", missed, MISSED_SAMPLE),
            ("false_alarms_20", false_alarm, FALSE_ALARM_SAMPLE),
        ):
            sample = pool.sample(min(size, len(pool)), random_state=SAMPLE_SEED).sort_index()
            add_review_columns(sample).to_csv(
                folder / f"ragbench_{subset.lower()}_{name}.csv", index=False
            )
            counts[f"{subset}_{name}"] = {"pool": int(len(pool)), "sampled": int(len(sample))}

    wrong = inserted[
        ((inserted["label"] == "Unsupported") & (inserted["predicted_label"] != "Unsupported"))
        | ((inserted["label"] == "Supported") & (inserted["predicted_label"] != "Supported"))
    ]
    add_review_columns(wrong).to_csv(folder / "inserted_test_errors.csv", index=False)

    storyboard = wrong[wrong["subset"] == "storyboard_b"]
    add_review_columns(storyboard).to_csv(folder / "storyboard_b_test_errors.csv", index=False)

    counts["inserted_test_errors"] = int(len(wrong))
    counts["storyboard_b_test_errors"] = int(len(storyboard))

    return counts


# ============================================================
# 6. FIELD 10 AND THE COMPARISON TABLE
# ============================================================

def field_10(ragbench: dict, inserted: dict) -> dict:
    overall = inserted["overall"]
    storyboard = inserted["storyboard_b"] or {}
    answer_auroc = ragbench["answer_auroc"]["overall"]
    ece_rag = ragbench["ece"]
    ece_ins = inserted["scores_overall"].get("ece")

    def at_least(value, target):
        return value is not None and value == value and value >= target

    checks = {
        "inserted_strict_recall>=0.80": at_least(overall["strict_recall"], TARGETS["strict_recall"]),
        "inserted_precision>=0.60": at_least(overall["precision"], TARGETS["precision"]),
        "storyboard_b_strict_recall>=0.80": at_least(
            storyboard.get("strict_recall"), TARGETS["storyboard_b_strict_recall"]
        ),
        "ragbench_answer_auroc>=0.75": at_least(answer_auroc, TARGETS["answer_auroc"]),
        "ragbench_ece<=0.10": ece_rag is not None and ece_rag <= TARGETS["max_ece"],
        "inserted_ece<=0.10": ece_ins is not None and ece_ins <= TARGETS["max_ece"],
    }
    checks["meets_all_targets"] = all(checks.values())

    return checks


def comparison_row(main: dict, spread: dict, metadata: dict) -> str:
    """Markdown row for the Experiment Card comparison table."""

    ins = main["inserted"]["overall"]
    rag = main["ragbench"]
    sb = main["inserted"]["storyboard_b"] or {}

    def f(value, digits=2):
        if value is None or value != value:
            return "N/A"
        return f"{value:.{digits}f}"

    seconds = metadata.get("training_seconds", {}).get("total")
    hours = "N/A" if seconds is None else f"0 GPU ({seconds / 3600:.2f} CPU hours)"

    return (
        "| Model | Overall (strict recall / precision / F1 / AUROC) | Storyboard B slice (strict recall) "
        "| Spread across 3 seeds (± std) | Calibration (ECE) | Training cost (GPU hours) "
        "| Inference latency (ms per claim) | Inspectability |\n"
        "|---|---|---|---|---|---|---|---|\n"
        f"| TF-IDF + Logistic Regression "
        f"| {f(ins['strict_recall'])} / {f(ins['precision'])} / {f(ins['f1'])} / {f(rag['answer_auroc']['overall'])} "
        f"| {f(sb.get('strict_recall'))} "
        f"| AUROC ± {f(spread.get('ragbench_answer_auroc', {}).get('std'), 3)} "
        f"| {f(rag['ece'], 3)} (RAGBench) / {f(main['inserted']['scores_overall'].get('ece'), 3)} (Inserted) "
        f"| {hours} "
        f"| {f(rag['end_to_end_ms_per_claim'], 1)} "
        "| 7 standardized coefficients |\n\n"
        "Strict recall, precision and F1: Inserted Error Set v2 test. "
        "AUROC: RAGBench test, answer level. Spread: RAGBench test answer AUROC over seeds 42, 43, 44.\n"
    )


# ============================================================
# 7. MAIN
# ============================================================

def run() -> dict:
    check_lock()

    meta_paths = {seed: metadata_path(seed) for seed in SEEDS}
    missing = [seed for seed, path in meta_paths.items() if not path.is_file()]

    if missing:
        raise SystemExit(
            f"Missing models for seeds {missing}. Run run_seeds.sh first, "
            "so the spread across seeds comes from the same test run."
        )

    metadata = {
        seed: json.loads(path.read_text(encoding="utf-8"))
        for seed, path in meta_paths.items()
    }

    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    lock = {
        "status": "started",
        "started_utc": started,
        "git": git_info(CLASSICAL_DIR),
        "models": {
            str(seed): {"file": meta_paths[seed].name, "model_sha256": m.get("model_sha256")}
            for seed, m in metadata.items()
        },
    }

    # The lock is written before any test data is read. If the run
    # stops halfway, the test split still counts as used.
    write_lock(lock)

    report = {
        "method": "TF-IDF + Logistic Regression (Classical)",
        "run_at_utc": started,
        "main_seed": MAIN_SEED,
        "seeds": list(SEEDS),
        "note": "Everything was chosen on validation. Nothing is chosen on test.",
        "shared_components": shared.shared_versions(),
        "inserted_input_sha256": shared.inserted_input_hashes(),
        "ragbench_test_sha256": ragbench_test_files(),
        "git": lock["git"],
        "libraries": library_versions(),
        "hardware": hardware_info(),
        "by_seed": {},
    }

    print("========== LOADING TEST DATA ==========")
    rag_test = load_ragbench_test()
    inserted_all = shared.load_inserted_error_set()
    inserted_test = inserted_all[inserted_all["split"] == "test"].reset_index(drop=True)
    print(f"RAGBench test claims: {len(rag_test)}")
    print(f"Inserted Error Set v2 test claims: {len(inserted_test)}")

    tables = {}

    for seed in SEEDS:
        print(f"\n========== SEED {seed} ==========")

        rag_result, rag_table = evaluate_ragbench(rag_test, meta_paths[seed])
        ins_table, ins_ms = predict_split(inserted_test, meta_paths[seed])
        ins_result = summarise_inserted(ins_table)
        ins_result["latency_ms_per_claim"] = ins_ms

        report["by_seed"][str(seed)] = {"ragbench": rag_result, "inserted": ins_result}
        tables[seed] = (rag_table, ins_table)

        a = rag_result["answer_auroc"]
        t = rag_result["three_label"]
        o = ins_result["overall"]
        sb = ins_result["storyboard_b"] or {}
        print(
            f"RAGBench  answer AUROC {a['overall']:.4f} (FinQA {a['FinQA']:.4f}, TATQA {a['TATQA']:.4f})  "
            f"claim ROC {rag_result['claim_level']['roc_auc']:.4f}  ECE {rag_result['ece']:.4f}  "
            f"strict R {t['strict_recall']:.4f}  P {t['unsupported_precision']:.4f}  "
            f"lenient R {t['lenient_recall']:.4f}  uncertain {t['uncertain_rate']:.4f}"
        )
        print(
            f"Inserted  strict R {o['strict_recall']:.3f}  P {o['precision']:.3f}  "
            f"lenient R {o['lenient_recall']:.3f}  uncertain {o['uncertain_rate']:.3f}  "
            f"AUROC {ins_result['scores_overall'].get('auroc', float('nan')):.3f}  "
            f"ECE {ins_result['scores_overall'].get('ece', float('nan')):.3f}"
        )
        print(
            f"Storyboard B  strict R {sb.get('strict_recall', float('nan')):.3f}  "
            f"P {sb.get('precision', float('nan')):.3f}"
        )

    # Spread across seeds (Field 5).
    def values(getter):
        return np.array([getter(report["by_seed"][str(s)]) for s in SEEDS], dtype=float)

    spread_metrics = {
        "ragbench_answer_auroc": lambda r: r["ragbench"]["answer_auroc"]["overall"],
        "ragbench_claim_roc_auc": lambda r: r["ragbench"]["claim_level"]["roc_auc"],
        "ragbench_ece": lambda r: r["ragbench"]["ece"],
        "ragbench_strict_recall": lambda r: r["ragbench"]["three_label"]["strict_recall"],
        "inserted_strict_recall": lambda r: r["inserted"]["overall"]["strict_recall"],
        "inserted_auroc": lambda r: r["inserted"]["scores_overall"].get("auroc", np.nan),
        "inserted_ece": lambda r: r["inserted"]["scores_overall"].get("ece", np.nan),
        "storyboard_b_strict_recall": lambda r: (r["inserted"]["storyboard_b"] or {}).get("strict_recall", np.nan),
    }

    spread = {}
    for name, getter in spread_metrics.items():
        v = values(getter)
        spread[name] = {
            "values": v.tolist(),
            "mean": float(np.nanmean(v)),
            "std": float(np.nanstd(v, ddof=1)) if np.isfinite(v).sum() > 1 else None,
        }
    report["spread_across_seeds"] = spread

    main = report["by_seed"][str(MAIN_SEED)]
    report["field_10_main_model"] = field_10(main["ragbench"], main["inserted"])

    rag_main, ins_main = tables[MAIN_SEED]
    rag_main.to_csv(OUTPUT_DIR / "ragbench_test_predictions.csv", index=False)
    ins_main.to_csv(OUTPUT_DIR / "inserted_test_predictions.csv", index=False)

    # Reliability diagrams for the main model (Field 8).
    diagrams = {}
    diagrams["ragbench_test"] = reliability_diagram(
        main["ragbench"]["calibration_table"],
        "RAGBench test (FinQA + TATQA)",
        OUTPUT_DIR / "reliability_ragbench_test.png",
    )

    if "calibration_table" in main["inserted"]["scores_overall"]:
        diagrams["inserted_test"] = reliability_diagram(
            main["inserted"]["scores_overall"]["calibration_table"],
            "Inserted Error Set v2 test",
            OUTPUT_DIR / "reliability_inserted_test.png",
        )

    sb_scores = main["inserted"]["scores_storyboard_b"] or {}
    if "calibration_table" in sb_scores:
        diagrams["storyboard_b_test"] = reliability_diagram(
            sb_scores["calibration_table"],
            "Storyboard B slice (test)",
            OUTPUT_DIR / "reliability_storyboard_b_test.png",
        )
    report["reliability_diagrams"] = diagrams

    report["error_analysis_samples"] = export_error_samples(rag_main, ins_main)

    row = comparison_row(main, spread, metadata[MAIN_SEED])
    (OUTPUT_DIR / "comparison_table_row.md").write_text(row, encoding="utf-8")

    out = OUTPUT_DIR / "final_test_results.json"
    out.write_text(json.dumps(_clean(report), indent=2, allow_nan=False), encoding="utf-8")

    lock["status"] = "finished"
    lock["finished_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    lock["results"] = out.name
    write_lock(lock)

    print("\n========== SPREAD ACROSS SEEDS ==========")
    for name, s in spread.items():
        std = "N/A" if s["std"] is None else f"{s['std']:.4f}"
        print(f"{name:<28} mean {s['mean']:.4f}  std {std}")

    print("\n========== FIELD 10 (MAIN MODEL, SEED 42) ==========")
    for name, ok in report["field_10_main_model"].items():
        print(f"{name:<36} {ok}")

    print("\n" + row)
    print(f"Wrote {OUTPUT_DIR}")
    print(f"The lock file {LOCK_PATH.name} now stops any second test run.")

    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the one-time final test (Experiment Card, Field 5)."
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Required. The test split can be used only once.",
    )
    args = parser.parse_args()

    if not args.confirm:
        raise SystemExit(
            "This uses the test split, which can be used only once. "
            "Run again with --confirm when everything on validation is final."
        )

    run()


if __name__ == "__main__":
    main()

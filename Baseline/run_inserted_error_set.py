"""Run the rule-based baseline on the Inserted Error Set.

    python Baseline/run_inserted_error_set.py                 # train + validation
    python Baseline/run_inserted_error_set.py --final-test    # ALSO test (run once, Field 5)

Writes predictions, metrics and versions to Baseline/outputs/.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from claimcheck_baseline import evidence, metrics, rules  # noqa: E402

OUT = Path(__file__).resolve().parent / "outputs"


def run_split(df: pd.DataFrame, cfg: rules.Config) -> tuple[pd.DataFrame, float]:
    preds, t0 = [], time.perf_counter()
    for r in df.itertuples():
        v = rules.verify(r.claim, r.evidence, cfg)  # model sees claim + passage only
        preds.append({"claim_id": r.claim_id, "pred": v.label, "rule": v.rule,
                      "explanation": v.explanation, "evidence_used": " || ".join(v.evidence)})
    ms = (time.perf_counter() - t0) * 1000 / max(len(df), 1)
    return df.merge(pd.DataFrame(preds), on="claim_id"), ms


def summarise(res: pd.DataFrame) -> dict:
    sb = res[res.subset == "storyboard_b"]
    overall = metrics.binary_metrics(res.label, res.pred)
    sb_m = metrics.binary_metrics(sb.label, sb.pred) if len(sb) else None
    flag_all = metrics.binary_metrics(res.label, pd.Series("Unsupported", index=res.index))
    return {
        "overall": overall,
        "general_claims": metrics.binary_metrics(res[res.subset == "general"].label, res[res.subset == "general"].pred),
        "storyboard_b": sb_m,
        "reference_flag_everything": flag_all,
        "targets_met": metrics.meets_targets(overall, sb_m),
        "per_error_type": metrics.per_error_type(res).round(3).reset_index().to_dict(orient="records"),
        "rules_fired": res.groupby(["error_type", "rule"]).size().rename("n").reset_index().to_dict(orient="records"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--final-test", action="store_true", help="also evaluate the test split (do this once)")
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    cfg = rules.Config()
    data = evidence.load_inserted_error_set()
    splits = ["train", "validation"] + (["test"] if args.final_test else [])

    report = {
        "method": "Rule-based baseline (number normalization + keyword matching)",
        "run_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "versions": {**rules.versions(), "evidence": evidence.EVIDENCE_VERSION},
        "config": cfg.__dict__,
        "input_sha256": evidence.input_hashes(),
        "python": platform.python_version(),
        "splits": {},
    }
    for split in splits:
        res, ms = run_split(data[data.split == split], cfg)
        res.to_csv(OUT / f"inserted_predictions_{split}.csv", index=False)
        s = summarise(res)
        s["latency_ms_per_claim"] = round(ms, 3)
        report["splits"][split] = s

        o, sb = s["overall"], s["storyboard_b"]
        print(f"\n=== {split} (n={o['n']}, {o['n_unsupported_gold']} wrong claims) ===")
        print(f"overall      strict R {o['strict_recall']:.3f}  lenient R {o['lenient_recall']:.3f}  "
              f"P {o['precision']:.3f}  F1 {o['f1']:.3f}  false alarms {o['false_alarm_rate']:.3f}  uncertain {o['uncertain_rate']:.3f}")
        if sb:
            print(f"storyboard B strict R {sb['strict_recall']:.3f}  lenient R {sb['lenient_recall']:.3f}  "
                  f"P {sb['precision']:.3f}  false alarms {sb['false_alarm_rate']:.3f}")
        fa = s["reference_flag_everything"]
        print(f"flag-everything reference: R 1.000  P {fa['precision']:.3f}")
        print(metrics.per_error_type(res).round(3).to_string())
        print("targets:", s["targets_met"], f"| {ms:.2f} ms/claim")

    (OUT / "inserted_metrics.json").write_text(json.dumps(report, indent=2, default=float))
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()

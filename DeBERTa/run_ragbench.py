"""Evaluate fine-tuned DeBERTa on RAGBench FinQA and TAT-QA (Experiment Card Fields 6-10).

    python DeBERTa/run_ragbench.py                # validation: temperature, thresholds, answer roll-up per seed
    python DeBERTa/run_ragbench.py --final-test   # test, once, with the saved setting

Claims, passages and gold labels are read exactly as in Baseline/run_ragbench.py.
One setting per seed is chosen on the two validation splits together and used
for both subsets (like the baseline's single RAGBench config).
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from claimcheck_deberta import data, model, report  # noqa: E402
from provenance_block import provenance  # noqa: E402

from claimcheck_baseline import metrics  # noqa: E402  (path added by claimcheck_deberta.data)

OUT = HERE / "outputs"


def evaluate(claims: pd.DataFrame, score: np.ndarray, setting: dict) -> dict:
    ans = report.answer_scores(claims, score, setting["aggregation"])
    return {"claim_level": report.label_metrics(claims.label, score, setting),
            "answer_level": report.label_metrics(ans.label, ans.score.to_numpy(), setting)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--final-test", action="store_true", help="evaluate the test splits (do this once)")
    args = ap.parse_args()
    chosen = json.loads((OUT / "chosen_config.json").read_text())
    setting_path = OUT / "ragbench_setting.json"
    split = "test" if args.final_test else "validation"
    out_path = OUT / f"ragbench_metrics_{split}.json"
    if args.final_test:
        if out_path.exists():
            sys.exit(f"{out_path.name} exists: the RAGBench test splits were already used once (Field 5).")
        settings = json.loads(setting_path.read_text())["per_seed"]

    sets = {s: data.load_ragbench(s, split) for s in data.SUBSETS}
    rep = {"method": "DeBERTa-v3-large (NLI checkpoint), fine-tuned on RAGBench FinQA/TAT-QA",
           "split": split, "run_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "versions": model.versions(), "train_config": {k: chosen[k] for k in ("lr", "epochs", "seeds")},
           "input_sha256": data.input_hashes(), "provenance": provenance(include_inserted=False), "per_seed": {}}
    new_settings = {}
    for seed, ckpt in chosen["checkpoints"].items():
        v = model.Verifier(str(HERE / ckpt))
        scores, ms, used = {}, {}, {}
        for s, (claims, passages) in sets.items():
            scored, ms[s] = v.score(claims, passages)
            scores[s], used[s] = scored.score.to_numpy(), scored.evidence_used.to_numpy()
        if args.final_test:
            setting = settings[seed]
        else:
            gold = pd.concat([sets[s][0].label for s in data.SUBSETS], ignore_index=True)
            setting = report.choose_setting(gold, np.concatenate([scores[s] for s in data.SUBSETS]))
            aurocs = {how: np.mean([metrics.auroc(a.label, pd.Series(1 - a.score.to_numpy()))
                                    for a in (report.answer_scores(sets[s][0], scores[s], how) for s in data.SUBSETS)])
                      for how in report.AGGREGATIONS}
            setting["aggregation"] = max(report.AGGREGATIONS, key=lambda h: (aurocs[h], h == "lowest"))
            setting["answer_auroc_by_aggregation"] = aurocs
            new_settings[seed] = setting
        rep["per_seed"][seed] = {"setting": setting, "subsets": {}}
        for s, (claims, _) in sets.items():
            ev = evaluate(claims, scores[s], setting)
            ev["latency_ms_per_claim"] = round(ms[s], 3)
            rep["per_seed"][seed]["subsets"][s] = ev
            claims.drop(columns=["support_keys"]).assign(score=scores[s], evidence_used=used[s]).to_csv(
                OUT / f"ragbench_{s.lower()}_{split}_scores_seed{seed}.csv", index=False)
            c = ev["claim_level"]
            print(f"seed {seed} {s:<6} claims: AUROC {c['score']['auroc']:.3f}  strict R {c['strict_recall']:.3f}  "
                  f"P {c['precision']:.3f}  F1 {c['f1']:.3f}  uncertain {c['uncertain_rate']:.3f}  ECE {c['ece']:.3f}  "
                  f"| {ms[s]:.1f} ms/claim")
        del v
        model.free_memory()

    rep["across_seeds"] = {f"{s}.{k}": report.across_seeds({sd: m["subsets"] for sd, m in rep["per_seed"].items()},
                                                          [s, "claim_level", *k.split(".")])
                           for s in data.SUBSETS
                           for k in ("score.auroc", "strict_recall", "precision", "f1", "uncertain_rate", "ece")}
    if not args.final_test:
        setting_path.write_text(json.dumps({"chosen_on": "RAGBench validation (FinQA + TAT-QA)", "per_seed": new_settings,
                                            "chosen_at_utc": rep["run_at_utc"]}, indent=2, default=float))
    out_path.write_text(json.dumps(rep, indent=2, default=float))
    print(f"\nacross seeds: {json.dumps(rep['across_seeds'], default=float)}\nWrote {OUT}")


if __name__ == "__main__":
    main()

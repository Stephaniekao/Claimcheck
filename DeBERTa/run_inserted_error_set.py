"""Evaluate fine-tuned DeBERTa on the Inserted Error Set v2 (Experiment Card Fields 6-10).

    python DeBERTa/run_inserted_error_set.py                # validation: fit temperature + thresholds per seed
    python DeBERTa/run_inserted_error_set.py --final-test   # test, once, with the saved setting

Needs DeBERTa/outputs/chosen_config.json from train.py. DeBERTa never trains on
this set; its validation split is used only for the temperature and the two
thresholds. Same passages and the same metric code as the baseline.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from claimcheck_deberta import data, model, report  # noqa: E402
from provenance_block import provenance  # noqa: E402

from claimcheck_baseline import metrics  # noqa: E402  (path added by claimcheck_deberta.data)

OUT = HERE / "outputs"


def summarise(res: pd.DataFrame, setting: dict) -> dict:
    sb, gen = res[res.subset == "storyboard_b"], res[res.subset == "general"]
    overall = report.label_metrics(res.label, res.score.to_numpy(), setting)
    sb_m = report.label_metrics(sb.label, sb.score.to_numpy(), setting) if len(sb) else None
    pred = pd.Series(res.pred.to_numpy(), index=res.index)
    return {"overall": overall,
            "general_claims": report.label_metrics(gen.label, gen.score.to_numpy(), setting),
            "storyboard_b": sb_m,
            "reference_flag_everything": metrics.binary_metrics(res.label, pd.Series("Unsupported", index=res.index)),
            "targets_met": metrics.meets_targets(overall, sb_m),
            "per_error_type": metrics.per_error_type(res.assign(pred=pred)).round(3).reset_index().to_dict(orient="records")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--final-test", action="store_true", help="evaluate the test split (do this once)")
    args = ap.parse_args()
    chosen = json.loads((OUT / "chosen_config.json").read_text())
    setting_path = OUT / "inserted_setting.json"
    split = "test" if args.final_test else "validation"
    out_path = OUT / f"inserted_metrics_{split}.json"
    if args.final_test:
        if out_path.exists():
            sys.exit(f"{out_path.name} exists: the Inserted Error Set test split was already used once (Field 5).")
        settings = json.loads(setting_path.read_text())["per_seed"]

    claims, passages = data.load_inserted_error_set()
    part = claims[claims.split == split].reset_index(drop=True)
    rep = {"method": "DeBERTa-v3-large (NLI checkpoint), fine-tuned on RAGBench FinQA/TAT-QA",
           "split": split, "run_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "versions": model.versions(), "train_config": {k: chosen[k] for k in ("lr", "epochs", "seeds")},
           "input_sha256": data.input_hashes(), "provenance": provenance(), "per_seed": {}}
    new_settings = {}
    for seed, ckpt in chosen["checkpoints"].items():
        v = model.Verifier(str(HERE / ckpt))
        scored, ms = v.score(part, passages)
        score = scored.score.to_numpy()
        if not args.final_test:
            settings_seed = report.choose_setting(part.label, score)
            new_settings[seed] = settings_seed
        else:
            settings_seed = settings[seed]
        res = part.assign(score=score, evidence_used=scored.evidence_used.to_numpy(), pred=report.cal.to_labels(
            report.cal.apply_temperature(score, settings_seed["temperature"]),
            settings_seed["lower"], settings_seed["upper"]).to_numpy())
        res.to_csv(OUT / f"inserted_predictions_{split}_seed{seed}.csv", index=False)
        s = summarise(res, settings_seed)
        s["setting"], s["latency_ms_per_claim"] = settings_seed, round(ms, 3)
        rep["per_seed"][seed] = s
        o, sb = s["overall"], s["storyboard_b"]
        print(f"seed {seed}: strict R {o['strict_recall']:.3f}  P {o['precision']:.3f}  F1 {o['f1']:.3f}  "
              f"uncertain {o['uncertain_rate']:.3f}  AUROC {o['score']['auroc']:.3f}  ECE {o['ece']:.3f}  "
              f"| Storyboard B strict R {sb['strict_recall']:.3f}  | {ms:.1f} ms/claim")
        del v
        model.free_memory()

    rep["across_seeds"] = {f"{part_}.{k}": report.across_seeds(rep["per_seed"], [part_, k])
                           for part_ in ("overall", "storyboard_b")
                           for k in ("strict_recall", "precision", "f1", "uncertain_rate", "ece")}
    if not args.final_test:
        setting_path.write_text(json.dumps({"chosen_on": "Inserted Error Set v2 validation", "per_seed": new_settings,
                                            "chosen_at_utc": rep["run_at_utc"]}, indent=2))
    out_path.write_text(json.dumps(rep, indent=2, default=float))
    print(f"\nacross seeds: {json.dumps(rep['across_seeds'], default=float)}\nWrote {OUT}")


if __name__ == "__main__":
    main()

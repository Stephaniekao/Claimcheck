"""Rule-based support score (score 1.0.0), added after the rules 1.2.0 test run.

    python Baseline/run_support_score.py                # validation (weights: reuse saved choice, else 4 trials)
    python Baseline/run_support_score.py --final-test   # test: Inserted Error Set only

Labels are not changed: the script checks every label against the predictions
saved by run_inserted_error_set.py / run_ragbench.py for the same split, and
stops if any differs. Run those first.

Labels are not changed: the script checks every label against the saved
rules 1.2.0 predictions and stops if any differs. Results go to
Baseline/outputs/support_score/, never over the 1.2.0 files.

Weights are chosen on RAGBench validation by mean claim-level AUROC (FinQA and
TAT-QA). 16 trials went to the 1.2.0 config, so 4 remain under Field 5's 20.
The same weights are then used on the Inserted Error Set.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from claimcheck_baseline import evidence, metrics, provenance, rules, score  # noqa: E402
from run_ragbench import OUT as BASE_OUT, SUBSETS, load  # noqa: E402

OUT = BASE_OUT / "support_score"
NAMES = list(score.WEIGHT_TRIALS)


def score_ragbench(df: pd.DataFrame, cfg: rules.Config, ws: list[score.Weights]) -> pd.DataFrame:
    rows = []
    for r in df.itertuples():
        units = rules.analyse_passage([s[1] for doc in r.documents_sentences for s in doc])
        bad = set(r.unsupported_response_sentence_keys)
        for key, sent in r.response_sentences:
            scores, v = score.support_scores(sent, None, cfg, ws, units=units)
            rows.append({"answer": r.Index, "key": key, "label": "Unsupported" if key in bad else "Supported",
                         "answer_label": "Supported" if bool(r.adherence_score) else "Unsupported",
                         "pred": v.label, "rule": v.rule, **dict(zip([f"score_{n}" for n in NAMES[:len(ws)]]
                                                                      if len(ws) > 1 else ["score"], scores))})
    return pd.DataFrame(rows)


def score_inserted(df: pd.DataFrame, cfg: rules.Config, w: score.Weights) -> pd.DataFrame:
    rows = []
    for r in df.itertuples():
        s, v = score.support_score(r.claim, r.evidence, cfg, w)
        rows.append({"claim_id": r.claim_id, "subset": r.subset, "error_type": r.error_type, "label": r.label,
                     "pred": v.label, "rule": v.rule, "score": s})
    return pd.DataFrame(rows)


def check_labels(new: pd.DataFrame, saved_csv: Path, on: list[str]) -> None:
    """Stop if the score run changed any rules 1.2.0 label."""
    old = pd.read_csv(saved_csv, usecols=on + ["pred"])
    m = new[on + ["pred"]].merge(old, on=on, suffixes=("", "_saved"), how="outer", indicator=True)
    bad = m[(m._merge != "both") | (m.pred != m.pred_saved)]
    if len(bad):
        sys.exit(f"Labels differ from {saved_csv.name} in {len(bad)} rows; the score must not change labels.")
    assert (new.pred == new.score.map(score.label_of)).all(), "score band disagrees with label"


def answer_scores(claims: pd.DataFrame) -> pd.DataFrame:
    """Answer score = its weakest claim (the 'lowest' roll-up chosen for 1.2.0)."""
    return claims.groupby("answer").agg(label=("answer_label", "first"), score=("score", "min")).reset_index()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--final-test", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    chosen_cfg = json.loads((BASE_OUT / "ragbench_chosen_config.json").read_text())
    if chosen_cfg["aggregation"] != "lowest":
        sys.exit("answer_scores() assumes the 'lowest' roll-up chosen for rules 1.2.0")
    rag_cfg = rules.Config(rel_tol=chosen_cfg["rel_tol"], min_word_overlap=chosen_cfg["min_word_overlap"],
                           derived_policy=chosen_cfg["derived_policy"])
    chosen_path = OUT / "chosen_weights.json"

    if not args.final_test and chosen_path.exists():
        chosen = json.loads(chosen_path.read_text())  # weights already chosen: no new trials
        split, splits_ies = "validation", ["train", "validation"]
    elif not args.final_test:
        split = "validation"
        ws = [score.WEIGHT_TRIALS[n] for n in NAMES]
        trials = {n: {} for n in NAMES}
        for s in SUBSETS:
            claims = score_ragbench(load(s, split), rag_cfg, ws)
            for n in NAMES:
                c = claims.rename(columns={f"score_{n}": "score"})
                trials[n][f"{s}_claim_auroc"] = metrics.auroc(c.label, 1 - c.score)
                a = answer_scores(c)
                trials[n][f"{s}_answer_auroc"] = metrics.auroc(a.label, 1 - a.score)
        tdf = pd.DataFrame(trials).T
        tdf["mean_claim_auroc"] = tdf[[f"{s}_claim_auroc" for s in SUBSETS]].mean(axis=1)
        tdf["mean_answer_auroc"] = tdf[[f"{s}_answer_auroc" for s in SUBSETS]].mean(axis=1)
        tdf.index.name = "weights"
        tdf.to_csv(OUT / "validation_weight_trials.csv")
        print(tdf.round(4).to_string())
        best = tdf.sort_values(["mean_claim_auroc", "mean_answer_auroc"], ascending=False).index[0]
        chosen = {"weights_name": best, "weights": score.WEIGHT_TRIALS[best].__dict__,
                  "chosen_on": "RAGBench validation: mean claim-level AUROC (FinQA, TAT-QA)",
                  "n_trials": len(NAMES), "trials_used_in_total_with_rules_1.2.0": chosen_cfg["n_trials"] + len(NAMES),
                  "chosen_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        chosen_path.write_text(json.dumps(chosen, indent=2))
        splits_ies = ["train", "validation"]
    else:
        chosen = json.loads(chosen_path.read_text())
        split, splits_ies = "test", ["test"]
    w = score.Weights(**chosen["weights"])

    report = {"split": split, "weights": chosen, "versions": {**rules.versions(), "score": score.SCORE_VERSION},
              "python": platform.python_version(), "provenance": provenance.provenance(),
              "note": "Added after the rules 1.2.0 test run; labels unchanged.",
              "ragbench": {}, "inserted_error_set": {}}
    # RAGBench test was used once (rules 1.2.0, support score 1.0.0); later runs score validation only.
    rag_subsets = SUBSETS if split == "validation" else []
    print(f"\n=== support score {score.SCORE_VERSION} | {split} | weights {chosen['weights_name']} {chosen['weights']} ===")
    for s in rag_subsets:
        claims = score_ragbench(load(s, split), rag_cfg, [w])
        check_labels(claims, BASE_OUT / f"ragbench_{s.lower()}_{split}_claims.csv", ["answer", "key"])
        claims.to_csv(OUT / f"ragbench_{s.lower()}_{split}_scores.csv", index=False)
        ans = answer_scores(claims)
        report["ragbench"][s] = {"claim_level": metrics.score_metrics(claims.label, claims.score),
                                 "answer_level": metrics.score_metrics(ans.label, ans.score)}
    ies = evidence.load_inserted_error_set()
    for sp in splits_ies:
        res = score_inserted(ies[ies.split == sp], rules.Config(), w)
        check_labels(res, BASE_OUT / f"inserted_predictions_{sp}.csv", ["claim_id"])
        res.to_csv(OUT / f"inserted_{sp}_scores.csv", index=False)
        sb = res[res.subset == "storyboard_b"]
        report["inserted_error_set"][sp] = {"overall": metrics.score_metrics(res.label, res.score),
                                            "storyboard_b": metrics.score_metrics(sb.label, sb.score)}

    for name, block in [*((f"RAGBench {s}", report["ragbench"][s]) for s in rag_subsets),
                        *((f"Inserted {sp}", report["inserted_error_set"][sp]) for sp in splits_ies)]:
        for lvl, m in block.items():
            p80 = m["precision_at_recall_0.80"]
            print(f"{name:<20} {lvl:<12} n={m['n']:<6} AUROC {m['auroc']:.3f}  AP {m['average_precision']:.3f} "
                  f"(base rate {m['base_rate']:.3f})  P@R0.80 {p80['precision']:.3f}  distinct scores {m['n_distinct_scores']}")
    out_path = OUT / f"metrics_{split}.json"
    if not rag_subsets and out_path.exists():
        # Keep the one RAGBench test result rather than dropping it from the file.
        old = json.loads(out_path.read_text())
        report["ragbench"] = old.get("ragbench", {})
        report["ragbench_from"] = old.get("ragbench_from", {"versions": old.get("versions"), "weights": old.get("weights")})
    out_path.write_text(json.dumps(report, indent=2, default=float))
    print(f"Labels identical to the saved rules {rules.RULES_VERSION} predictions on every split checked.")


if __name__ == "__main__":
    main()

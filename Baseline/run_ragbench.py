"""Run the rule-based baseline on RAGBench FinQA and TAT-QA.

    python Baseline/run_ragbench.py                  # validation: pick config (<= 20 trials, Field 5)
    python Baseline/run_ragbench.py --final-test     # test once with the chosen config

Claims = RAGBench `response_sentences` (the same split for every method).
Passage = the retrieved `documents_sentences` for that answer.
Gold:  sentence Unsupported  <=> its key is in `unsupported_response_sentence_keys`
       answer Unsupported    <=> `adherence_score` is False (RAGBench's answer label)
Not used as input: question, any annotation/explanation column, adherence_score.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from claimcheck_baseline import metrics, rules  # noqa: E402

PROJECT = Path(__file__).resolve().parents[1]
RAG = PROJECT / "Data" / "RAGBench"
OUT = Path(__file__).resolve().parent / "outputs"
SUBSETS = ["FinQA", "TATQA"]
ORDER = {"Unsupported": 0, "Uncertain": 1, "Supported": 2}

# Answer roll-up rules (Field 6: "a rule chosen on the validation split").
#   lowest          : the answer gets its lowest claim label
#   lowest_checkable: same, but sentences with no number (rule "no_amount") are
#                     skipped; an answer with nothing checkable is Uncertain
AGGREGATIONS = ["lowest", "lowest_checkable"]
# 2 x 2 x 2 x 2 = 16 trials (Field 5 allows 20).
GRID = {"rel_tol": [0.0, 0.01], "min_word_overlap": [0.3, 0.5], "derived_policy": ["uncertain", "unsupported"],
        "aggregation": AGGREGATIONS}


def load(subset: str, split: str) -> pd.DataFrame:
    return pd.read_parquet(RAG / subset / f"{split}.parquet",
                           columns=["id", "documents_sentences", "response_sentences",
                                    "unsupported_response_sentence_keys", "adherence_score"])


def predict(df: pd.DataFrame, cfg: rules.Config) -> tuple[pd.DataFrame, float]:
    rows, t0, n_claims = [], time.perf_counter(), 0
    for r in df.itertuples():
        units = rules.analyse_passage([s[1] for doc in r.documents_sentences for s in doc])
        bad = set(r.unsupported_response_sentence_keys)
        for key, sent in r.response_sentences:
            v = rules.verify(sent, None, cfg, units=units)
            n_claims += 1
            rows.append({"answer": r.Index, "id": r.id, "key": key, "claim": sent, "label": "Unsupported" if key in bad else "Supported",
                         "pred": v.label, "rule": v.rule, "explanation": v.explanation,
                         "answer_label": "Supported" if bool(r.adherence_score) else "Unsupported"})
    ms = (time.perf_counter() - t0) * 1000 / max(n_claims, 1)
    return pd.DataFrame(rows), ms


def aggregate(claims: pd.DataFrame, how: str) -> pd.DataFrame:
    def roll(g: pd.DataFrame) -> str:
        if how == "lowest_checkable":
            g = g[g.rule != "no_amount"]
            if g.empty:
                return "Uncertain"
        return min(g.pred, key=ORDER.__getitem__)
    ans = claims.groupby("answer").agg(label=("answer_label", "first"))
    ans["pred"] = claims.groupby("answer")[["pred", "rule"]].apply(roll)
    return ans.reset_index()


def evaluate(claims: pd.DataFrame, how: str) -> dict:
    ans = aggregate(claims, how)
    return {"claim_level": metrics.binary_metrics(claims.label, claims.pred),
            "answer_level": metrics.binary_metrics(ans.label, ans.pred),
            "claim_rules_fired": claims.groupby(["label", "rule"]).size().rename("n").reset_index().to_dict(orient="records"),
            "claim_confusion": metrics.confusion(claims).to_dict()}


def line(name: str, m: dict) -> str:
    return (f"{name:<18} n={m['n']:<6} wrong={m['n_unsupported_gold']:<5} strict R {m['strict_recall']:.3f}  "
            f"lenient R {m['lenient_recall']:.3f}  P {m['precision']:.3f}  F1 {m['f1']:.3f}  "
            f"false alarms {m['false_alarm_rate']:.3f}  uncertain {m['uncertain_rate']:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--final-test", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    chosen_path = OUT / "ragbench_chosen_config.json"

    if not args.final_test:
        val = {s: load(s, "validation") for s in SUBSETS}
        trials, cache = [], {}
        for rel_tol, overlap, policy, how in itertools.product(*GRID.values()):
            key = (rel_tol, overlap, policy)
            if key not in cache:
                cfg = rules.Config(rel_tol=rel_tol, min_word_overlap=overlap, derived_policy=policy)
                cache[key] = {s: predict(val[s], cfg) for s in SUBSETS}
            res = {s: evaluate(cache[key][s][0], how) for s in SUBSETS}
            f1 = sum(res[s]["claim_level"]["f1"] for s in SUBSETS) / len(SUBSETS)
            trials.append({"rel_tol": rel_tol, "min_word_overlap": overlap, "derived_policy": policy, "aggregation": how,
                           "mean_claim_f1": f1, **{f"{s}_{lvl}_{k}": res[s][lvl][k]
                                                   for s in SUBSETS for lvl in ("claim_level", "answer_level")
                                                   for k in ("strict_recall", "precision", "f1", "uncertain_rate")}})
            unc = sum(res[s]["claim_level"]["uncertain_rate"] for s in SUBSETS) / len(SUBSETS)
            trials[-1]["mean_claim_uncertain"] = unc
            print(f"trial {len(trials):>2}: rel_tol={rel_tol} overlap={overlap} derived={policy:<11} agg={how:<16} "
                  f"mean claim F1 {f1:.4f}  uncertain {unc:.3f}")
        trials_df = pd.DataFrame(trials)
        trials_df.to_csv(OUT / "ragbench_validation_trials.csv", index=False)
        # Field 8 selection: keep Uncertain <= 1/3 (AC2), then highest Unsupported F1.
        # If no trial satisfies AC2, fall back to the highest F1 and say so.
        trials_df["mean_answer_f1"] = trials_df[[f"{s}_answer_level_f1" for s in SUBSETS]].mean(axis=1)
        ok = trials_df[trials_df.mean_claim_uncertain <= 1 / 3]
        pool = ok if len(ok) else trials_df
        best = pool.sort_values(["mean_claim_f1", "mean_answer_f1", "rel_tol"], ascending=[False, False, True]).iloc[0]
        chosen = {"rel_tol": float(best.rel_tol), "min_word_overlap": float(best.min_word_overlap),
                  "derived_policy": best.derived_policy, "aggregation": best.aggregation,
                  "ac2_satisfied": bool(len(ok)),
                  "chosen_on": "RAGBench validation: Uncertain <= 1/3 (AC2), then mean claim-level Unsupported F1",
                  "n_trials": len(trials), "versions": rules.versions(),
                  "chosen_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        chosen_path.write_text(json.dumps(chosen, indent=2))
        split, data = "validation", val
    else:
        chosen = json.loads(chosen_path.read_text())
        split, data = "test", {s: load(s, "test") for s in SUBSETS}

    cfg = rules.Config(rel_tol=chosen["rel_tol"], min_word_overlap=chosen["min_word_overlap"],
                       derived_policy=chosen["derived_policy"])
    report = {"split": split, "config": chosen, "versions": rules.versions(), "subsets": {}}
    print(f"\n=== RAGBench {split} | rel_tol={cfg.rel_tol} overlap={cfg.min_word_overlap} "
          f"derived={cfg.derived_policy} agg={chosen['aggregation']} | AC2 satisfiable: {chosen['ac2_satisfied']} ===")
    for s in SUBSETS:
        claims, ms = predict(data[s], cfg)
        claims.to_csv(OUT / f"ragbench_{s.lower()}_{split}_claims.csv", index=False)
        ev = evaluate(claims, chosen["aggregation"])
        ev["latency_ms_per_claim"] = round(ms, 3)
        report["subsets"][s] = ev
        print(line(f"{s} claims", ev["claim_level"]))
        print(line(f"{s} answers", ev["answer_level"]))
        print(f"{'':<18} rules fired:", claims.groupby(["label", "rule"]).size().to_dict(), f"| {ms:.2f} ms/claim")
    (OUT / f"ragbench_metrics_{split}.json").write_text(json.dumps(report, indent=2, default=float))


if __name__ == "__main__":
    main()

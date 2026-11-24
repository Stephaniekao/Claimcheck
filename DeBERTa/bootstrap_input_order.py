"""Evaluate the amended input-order decision rule from saved validation scores."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPLICATES = 2000
RESAMPLE_SEED = 2026


def auc_counter(score, gold, groups):
    """Prepare weighted rank AUROC with half credit for tied positive/negative pairs."""
    risk = 1 - np.asarray(score, dtype=float)
    assert np.isfinite(risk).all()
    order = np.argsort(risk, kind="stable")
    starts = np.r_[0, np.flatnonzero(np.diff(risk[order])) + 1]
    sorted_gold = gold[order]
    group_order = groups[order]

    def count(group_weights):
        weights = group_weights[group_order]
        positives = np.add.reduceat(weights * sorted_gold, starts)
        negatives = np.add.reduceat(weights * ~sorted_gold, starts)
        denominator = positives.sum() * negatives.sum()
        if not denominator:
            return float("nan")
        return float(positives @ (np.cumsum(negatives) - 0.5 * negatives) / denominator)

    return count


def compare_subset(subset, trial_dir, rng, candidate="evidence_first"):
    keys = ["answer", "id", "key", "label"]
    old = pd.read_csv(HERE / "outputs" / f"ragbench_{subset}_validation_scores_seed0.csv")
    new = pd.read_csv(trial_dir / f"{subset}_validation.csv")
    aligned = old.merge(new[keys + ["score"]], on=keys, suffixes=("_old", "_new"),
                        validate="one_to_one")
    assert len(aligned) == len(old) == len(new)
    gold = aligned.label.eq("Unsupported").to_numpy()
    groups, answers = pd.factorize(aligned.answer)
    n = len(answers)
    assert n and (groups >= 0).all()
    count_old = auc_counter(aligned.score_old, gold, groups)
    count_new = auc_counter(aligned.score_new, gold, groups)
    full = np.ones(n, dtype=np.int64)
    old_auc, new_auc = count_old(full), count_new(full)

    draws = []
    while len(draws) < REPLICATES:
        weight = np.bincount(rng.integers(n, size=n), minlength=n)
        delta = count_new(weight) - count_old(weight)
        if np.isfinite(delta):
            draws.append(delta)

    def top_k(frame):
        ranked = frame.sort_values(["score", "id", "key"])
        return {k: {"hits": hits, "precision": hits / k} for k in (20, 50, 100)
                for hits in [int(ranked.head(k).label.eq("Unsupported").sum())]}

    summary = {"answers": n, "claims": len(aligned), "claim_first_auroc": old_auc,
               f"{candidate}_auroc": new_auc, "delta": new_auc-old_auc,
               "claim_first_top_k": top_k(old), f"{candidate}_top_k": top_k(new)}
    return summary, np.asarray(draws)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trial_dir", type=Path)
    parser.add_argument("--hard-negatives", action="store_true")
    args = parser.parse_args()
    rng = np.random.default_rng(RESAMPLE_SEED)
    candidate = "hard_negatives" if args.hard_negatives else "evidence_first"
    pairs = dict((sub, compare_subset(sub, args.trial_dir, rng, candidate)) for sub in ("finqa", "tatqa"))
    subsets = {sub: summary for sub, (summary, _) in pairs.items()}
    delta = float(np.mean([s["delta"] for s in subsets.values()]))
    draws = np.mean([samples for _, samples in pairs.values()], axis=0)
    interval = np.quantile(draws, [0.025, 0.975]).tolist()
    summary = dict(split="validation", seed=0, bootstrap_replicates=REPLICATES, resample_seed=RESAMPLE_SEED,
                   subsets=subsets, mean_auroc_delta=delta, paired_bootstrap_ci95=interval,
                   qualifies_for_seed_confirmation=bool(delta > 0.016 and interval[0] > 0))
    if args.hard_negatives:
        original_hits = sum(s["claim_first_top_k"][50]["hits"] for s in subsets.values())
        assert original_hits == 29
        candidate_hits = sum(s["hard_negatives_top_k"][50]["hits"] for s in subsets.values())
        summary.update(original_top_50_hits=original_hits, candidate_top_50_hits=candidate_hits,
                       qualifies_for_seed_confirmation=bool(summary["qualifies_for_seed_confirmation"]
                                                            and candidate_hits >= original_hits - 2))
    serialized = json.dumps(summary, indent=2)
    (args.trial_dir / "paired_bootstrap_decision.json").write_text(serialized, encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()

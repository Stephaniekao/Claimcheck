"""Run the predeclared single-seed input-order comparison on validation."""
import argparse
import json
from pathlib import Path

import pandas as pd
import torch

import train
from claimcheck_deberta import data, model


class EvidenceFirstTokenizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __getattr__(self, name):
        return getattr(self.tokenizer, name)

    def __call__(self, text, text_pair=None, **kwargs):
        if text_pair is None:
            return self.tokenizer(text, **kwargs)
        assert kwargs.get("truncation") == "only_second"
        return self.tokenizer(text_pair, text, **{**kwargs, "truncation": "only_first"})


def summarize(frame):
    risk = 1 - frame.score
    ranked = frame.sort_values(["score", "id", "key"])
    return {
        "auroc": train.metrics.auroc(frame.label, risk),
        "top_hits": {k: int(ranked.head(k).label.eq("Unsupported").sum()) for k in (20, 50, 100)},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--score-saved-checkpoint", action="store_true")
    args = parser.parse_args()
    assert torch.cuda.is_available(), "CUDA is required for the matched trial"
    checkpoint = args.output / "checkpoint"
    if not args.score_saved_checkpoint:
        args.output.mkdir(parents=True, exist_ok=False)
    verifier = model.Verifier(str(checkpoint.resolve())) if args.score_saved_checkpoint else model.Verifier()
    verifier.tok = EvidenceFirstTokenizer(verifier.tok)
    hours = None
    if not args.score_saved_checkpoint:
        pairs = train.build_pairs(verifier, {s: data.load_ragbench(s, "train") for s in data.SUBSETS}, seed=0)
        assert len(pairs) == 14665
        hours = verifier.fit(pairs, lr=1e-5, schedule_epochs=2, seed=0)
        verifier.save(checkpoint)
    result = {}
    for subset in data.SUBSETS:
        claims, passages = data.load_ragbench(subset, "validation")
        print(f"Scoring full {subset} validation", flush=True)
        scored, _ = verifier.score(claims, passages)
        candidate = claims.assign(score=scored.score, evidence_used=scored.evidence_used)
        base = pd.read_csv(train.OUT / f"ragbench_{subset.lower()}_validation_scores_seed0.csv")
        keys = ["answer", "id", "key", "label"]
        assert len(base.merge(candidate[keys], on=keys, validate="one_to_one")) == len(base) == len(candidate)
        result[subset] = {"claim_first": summarize(base), "evidence_first": summarize(candidate)}
        candidate.drop(columns="support_keys").to_csv(args.output / f"{subset.lower()}_validation.csv", index=False)
        print(subset, result[subset], flush=True)
    delta = sum(r["evidence_first"]["auroc"] - r["claim_first"]["auroc"] for r in result.values()) / len(result)
    hits_ok = all(r["evidence_first"]["top_hits"][k] >= r["claim_first"]["top_hits"][k]
                  for r in result.values() for k in (20, 50))
    summary = dict(trial=5, seed=0, lr=1e-5, epochs=2, training_pairs=14665, training_hours=hours,
                   validation=result, mean_auroc_delta=delta, both_top_k_non_decreasing=hits_ok,
                   qualifies_for_confirmation=delta > 0.016 and hits_ok)
    (args.output / "result.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

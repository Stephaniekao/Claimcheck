"""Mine hard negatives and train a matched claim-first model on RAGBench."""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import train
from claimcheck_deberta import data, model

KEYS = ["answer", "id", "key"]


def mine(train_sets, checkpoint, output):
    verifier = model.Verifier(str(checkpoint))
    results = []
    for subset, (claims, passages) in train_sets.items():
        negatives = claims[claims.label.eq("Unsupported")].reset_index(drop=True)
        print(f"Mining {len(negatives)} {subset} training negatives", flush=True)
        for start in range(0, len(negatives), 128):
            chunk = negatives.iloc[start:start + 128].reset_index(drop=True)
            scored, _ = verifier.score(chunk, passages)
            assert np.isfinite(scored.score).all()
            results.append(chunk[KEYS + ["claim"]].assign(
                subset=subset, mined_window=scored.evidence_used, miner_support_score=scored.score))
            print(f"  mined {min(start + len(chunk), len(negatives))}/{len(negatives)}", flush=True)
    mined = pd.concat(results, ignore_index=True)
    assert not mined.duplicated(["subset"] + KEYS).any()
    mined.to_csv(output / "mined_windows.csv", index=False)
    audit = mined.sample(n=30, random_state=2026).copy()
    audit["full_evidence"] = ["\n".join(train_sets[r.subset][1][r.answer]["sentences"]) for r in audit.itertuples()]
    audit.insert(0, "audit_id", range(1, len(audit) + 1))
    audit.to_csv(output / "label_audit_sample.csv", index=False, encoding="utf-8-sig")
    return mined


def replace_unsupported(verifier, train_sets, mined, seed):
    rng = random.Random(seed)
    parts = []
    replaced = 0
    for subset, (claims, passages) in train_sets.items():
        sampled = train.sample_train(claims, seed).reset_index(drop=True)
        pairs = verifier.training_pairs(sampled, passages, rng)
        assert pairs.claim.eq(sampled.claim).all()
        bad = sampled.label.eq("Unsupported")
        selected = sampled.loc[bad, KEYS + ["claim"]].merge(
            mined.loc[mined.subset.eq(subset), KEYS + ["claim", "mined_window"]],
            on=KEYS + ["claim"], how="left", validate="one_to_one", sort=False)
        assert selected.mined_window.notna().all() and len(selected) == int(bad.sum())
        replaced += int((pairs.loc[bad, "window"].to_numpy() != selected.mined_window.to_numpy()).sum())
        pairs.loc[bad, "window"] = selected.mined_window.to_numpy()
        parts.append(pairs)
    print(f"Unsupported window text changed for {replaced} pairs", flush=True)
    return pd.concat(parts, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--mined-from", type=Path)
    args = parser.parse_args()
    assert torch.cuda.is_available()
    args.output.mkdir(parents=True, exist_ok=False)
    train_sets = {s: data.load_ragbench(s, "train") for s in data.SUBSETS}
    if args.mined_from:
        mined = pd.read_csv(args.mined_from, keep_default_na=False)
    else:
        config = json.loads((train.OUT / "chosen_config.json").read_text())
        checkpoint = (train.HERE / config["checkpoints"]["0"]).resolve()
        mined = mine(train_sets, checkpoint, args.output)
        model.free_memory()
    verifier = model.Verifier()
    pairs = replace_unsupported(verifier, train_sets, mined, args.seed)
    assert len(pairs) == 14665
    hours = verifier.fit(pairs, lr=1e-5, schedule_epochs=2, seed=args.seed)
    verifier.save(args.output / "checkpoint")
    summary = dict(trial=6, seed=args.seed, training_hours=hours, validation={})
    for subset in data.SUBSETS:
        print(f"Scoring full {subset} validation", flush=True)
        claims, passages = data.load_ragbench(subset, "validation")
        scored, _ = verifier.score(claims, passages)
        claims = claims.assign(score=scored.score, evidence_used=scored.evidence_used)
        summary["validation"][subset] = train.metrics.auroc(claims.label, 1 - claims.score)
        claims.drop(columns="support_keys").to_csv(args.output / f"{subset.lower()}_validation.csv", index=False)
    (args.output / "result.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(summary, flush=True)


if __name__ == "__main__":
    main()

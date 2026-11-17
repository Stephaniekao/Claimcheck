"""Score the pinned MiniCheck release in the project's validation pipeline."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import train
from bootstrap_input_order import compare_subset
from claimcheck_deberta import data, model, report

CHECKPOINT = "lytang/MiniCheck-DeBERTa-v3-Large"
REVISION = "2f2d01a54fa022a7ffadb76260e1ea8bc88c82bb"


class MiniCheckVerifier(model.Verifier):
    def __init__(self):
        self.tok = AutoTokenizer.from_pretrained(CHECKPOINT, revision=REVISION, trust_remote_code=False)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            CHECKPOINT, revision=REVISION, trust_remote_code=False, weights_only=True,
            dtype=torch.float32).to(model.device())
        assert self.model.config.num_labels == 2
        assert self.model.config.id2label[1] == "1"
        a, b = "The revenue was 10.", "Revenue was 10."
        assert self.tok(a, b).input_ids == self.tok(a + self.tok.eos_token + b).input_ids
        self._len_cache = {}

    @torch.inference_mode()
    def _entail_prob(self, pairs):
        self.model.eval()
        order = sorted(range(len(pairs)), key=lambda i: len(pairs[i][1]))
        scores = np.empty(len(pairs), dtype=np.float32)
        for start in range(0, len(order), 32):
            ids = order[start:start + 32]
            inputs = self.tok([pairs[i][1] for i in ids], [pairs[i][0] for i in ids],
                              truncation="only_first", max_length=model.MAX_LEN, padding=True,
                              return_token_type_ids=False, return_tensors="pt").to(model.device())
            with torch.autocast(device_type=model.device().type, dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                logits = self.model(**inputs).logits.float()
            scores[ids] = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
        return scores


def evaluate(verifier, data_sets, output, sample_only):
    stage = "sample" if sample_only else "full"
    summary, frames = {}, []
    for subset, (claims, passages) in data_sets.items():
        if sample_only:
            baseline = pd.read_csv(train.OUT / f"ragbench_{subset.lower()}_validation_scores_seed0.csv")
            keys = ["answer", "id", "key", "label"]
            claims = claims.merge(baseline[keys], on=keys, validate="one_to_one")
            bad = claims.label.eq("Unsupported")
            claims = pd.concat([claims[bad], claims[~bad].sample(n=3*int(bad.sum()), random_state=2026)],
                               ignore_index=True)
        print(f"Scoring {stage} {subset}: {len(claims)} claims", flush=True)
        scored, _ = verifier.score(claims, passages)
        frame = claims.assign(score=scored.score, evidence_used=scored.evidence_used)
        frames.append(frame)
        summary[subset] = train.metrics.auroc(frame.label, 1 - frame.score)
        filename = f"{subset.lower()}_{'sample' if sample_only else 'validation'}.csv"
        frame.drop(columns="support_keys").to_csv(output / filename, index=False)
        print(subset, summary[subset], flush=True)
    combined = pd.concat(frames, ignore_index=True)
    result = {"auroc": summary, "mean_auroc": float(np.mean(list(summary.values()))),
              "shared_setting": report.choose_setting(combined.label, combined.score.to_numpy())}
    print(stage, result, flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    assert torch.cuda.is_available()
    args.output.mkdir(parents=True, exist_ok=False)
    verifier = MiniCheckVerifier()
    sets = {s: data.load_ragbench(s, "validation") for s in data.SUBSETS}
    result = {"checkpoint": CHECKPOINT, "revision": REVISION,
              "sample": evaluate(verifier, sets, args.output, sample_only=True)}
    result_path = args.output / "minicheck_screen.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if result["sample"]["mean_auroc"] < 0.70:
        return
    result["full"] = evaluate(verifier, sets, args.output, sample_only=False)
    rng = np.random.default_rng(2026)
    checks = [compare_subset(s.lower(), args.output, rng, "minicheck") for s in data.SUBSETS]
    delta = float(np.mean([m["delta"] for m, _ in checks]))
    lo, hi = np.quantile(np.mean([ds for _, ds in checks], axis=0), [0.025, 0.975]).tolist()
    hits = sum(m["minicheck_top_k"][50]["hits"] for m, _ in checks)
    result["comparison"] = {"by_subset": dict(zip(data.SUBSETS, [m for m, _ in checks])),
                            "delta": delta, "ci95": [lo, hi], "top_50_hits_combined": hits,
                            "passes_direct_adoption_gate": delta > 0.016 and lo > 0 and hits >= 27,
                            "passes_training_screen": result["full"]["mean_auroc"] >= 0.78}
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()

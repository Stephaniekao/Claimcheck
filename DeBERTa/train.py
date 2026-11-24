"""Fine-tune DeBERTa-v3-large on RAGBench FinQA + TAT-QA train (Experiment Card Fields 3 and 5).

    python DeBERTa/train.py --smoke     # ~5 min: checks the code runs end to end, saves nothing
    python DeBERTa/train.py --pilot     # ~30 min: does fine-tuning help, and how long will the full run take? saves nothing
    python DeBERTa/train.py             # 4 tuning trials (seed 0), then seeds 1 and 2 for the chosen one

Training data: every Unsupported train claim, plus SUPPORTED_PER_UNSUPPORTED
Supported claims per Unsupported one (RAGBench finance is about 1.5% Unsupported;
training on all of it would teach the model to answer Supported). Sampling uses
the run's seed. The model never trains on the Inserted Error Set.

Trials: 2 learning rates x checkpoints after epoch 1 and 2 of a 2-epoch schedule
= 4 trials (Field 5 allows 20), all with seed 0. Chosen by mean claim-level
AUROC on RAGBench validation (FinQA, TAT-QA), the card's RAGBench target.
The chosen setting is then trained with seeds 1 and 2 (Field 5: 3 seeds).
No test split is read here.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from claimcheck_deberta import data, model  # noqa: E402
from provenance_block import provenance  # noqa: E402

sys.path.insert(0, str(HERE.parent / "Baseline"))
from claimcheck_baseline import metrics  # noqa: E402

OUT = HERE / "outputs"
CKPT = HERE / "checkpoints"   # git-ignored: about 1.7 GB per checkpoint
LRS = [1e-5, 2e-5]
SCHEDULE_EPOCHS = 2
SEEDS = [0, 1, 2]
SUPPORTED_PER_UNSUPPORTED = 4


def sample_train(claims: pd.DataFrame, seed: int) -> pd.DataFrame:
    bad = claims[claims.label == "Unsupported"]
    good = claims[claims.label == "Supported"]
    good = good.sample(n=min(len(good), SUPPORTED_PER_UNSUPPORTED * len(bad)), random_state=seed)
    return pd.concat([bad, good]).sort_index()


def build_pairs(v: model.Verifier, train: dict, seed: int, limit: int | None = None) -> pd.DataFrame:
    rng = random.Random(seed)
    parts = []
    for s, (claims, passages) in train.items():
        c = sample_train(claims, seed)
        if limit:
            c = c.sample(n=min(limit, len(c)), random_state=seed)
        parts.append(v.training_pairs(c, passages, rng))
    return pd.concat(parts, ignore_index=True)


def val_sample(val: dict, n: int, seed: int = 0) -> dict:
    """A validation sample that keeps every Unsupported claim (up to n/10), so AUROC is defined."""
    out = {}
    for s, (claims, passages) in val.items():
        bad = claims[claims.label == "Unsupported"]
        bad = bad.sample(n=min(len(bad), n // 10), random_state=seed)
        good = claims[claims.label == "Supported"].sample(n=n - len(bad), random_state=seed)
        out[s] = (pd.concat([bad, good]).sort_index().reset_index(drop=True), passages)
    return out


def val_auroc(v: model.Verifier, val: dict) -> dict:
    out = {}
    for s, (claims, passages) in val.items():
        sc, _ = v.score(claims, passages)
        out[s] = metrics.auroc(claims.label, pd.Series(1 - sc.score.to_numpy()))
    out["mean"] = sum(out[s] for s in val) / len(val)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="tiny run to check the code; writes nothing")
    ap.add_argument("--pilot", action="store_true", help="small real run: zero-shot vs fine-tuned AUROC, timing")
    args = ap.parse_args()

    train = {s: data.load_ragbench(s, "train") for s in data.SUBSETS}
    val = {s: data.load_ragbench(s, "validation") for s in data.SUBSETS}
    if args.smoke or args.pilot:
        n_pairs, n_val = (32, 200) if args.smoke else (1000, 2000)
        name = "smoke" if args.smoke else "pilot"
        val = val_sample(val, n_val)
        v = model.Verifier()
        print(f"{name}: zero-shot val AUROC (before fine-tuning) {val_auroc(v, val)}")
        pairs = build_pairs(v, train, seed=0, limit=n_pairs)
        hours = v.fit(pairs, lr=LRS[1], schedule_epochs=1, seed=0)
        print(f"{name}: fine-tuned val AUROC {val_auroc(v, val)}")
        full = sum(len(sample_train(c, 0)) for c, _ in train.values())
        print(f"{name} OK: {len(pairs)} pairs in {hours * 60:.1f} min on {model.gpu_name()}. "
              f"Full run trains on {full} pairs per epoch: about {hours / len(pairs) * full:.1f} h per epoch, "
              f"plus scoring the full validation set after each epoch.")
        return

    OUT.mkdir(exist_ok=True)
    CKPT.mkdir(exist_ok=True)
    trials, gpu_hours = [], {}

    # 1) 4 trials with seed 0
    for lr in LRS:
        v = model.Verifier()
        pairs = build_pairs(v, train, seed=0)

        def end(epoch: int, lr=lr, v=v) -> None:
            path = CKPT / f"lr{lr:g}_ep{epoch}_seed0"
            v.save(path)
            print(f"Scoring validation: lr={lr:g} epoch={epoch} seed=0", flush=True)
            a = val_auroc(v, val)
            trials.append({"lr": lr, "epochs": epoch, "seed": 0, "checkpoint": str(path.relative_to(HERE)),
                           **{f"val_auroc_{k}": a[k] for k in a}})
            pd.DataFrame(trials).to_csv(OUT / "train_trials.csv", index=False)
            print(f"trial {len(trials)}: lr={lr:g} epoch={epoch}  val AUROC {a}")

        gpu_hours[f"lr{lr:g}_seed0"] = v.fit(pairs, lr=lr, schedule_epochs=SCHEDULE_EPOCHS, seed=0, on_epoch_end=end)
        del v
        model.free_memory()
    trials_df = pd.DataFrame(trials)
    best = trials_df.sort_values(["val_auroc_mean", "epochs", "lr"], ascending=[False, True, True]).iloc[0]
    lr, epochs = float(best.lr), int(best.epochs)

    # 2) the chosen setting with seeds 1 and 2 (same 2-epoch schedule, stop at the chosen epoch)
    checkpoints = {0: best.checkpoint}
    for seed in SEEDS[1:]:
        v = model.Verifier()
        pairs = build_pairs(v, train, seed=seed)
        path = CKPT / f"lr{lr:g}_ep{epochs}_seed{seed}"
        gpu_hours[f"lr{lr:g}_seed{seed}"] = v.fit(pairs, lr=lr, schedule_epochs=SCHEDULE_EPOCHS, seed=seed,
                                                  stop_after=epochs)
        v.save(path)
        checkpoints[seed] = str(path.relative_to(HERE))
        del v
        model.free_memory()

    chosen = {"lr": lr, "epochs": epochs, "schedule_epochs": SCHEDULE_EPOCHS, "seeds": SEEDS,
              "checkpoints": checkpoints, "n_trials": len(trials),
              "chosen_on": "RAGBench validation: mean claim-level AUROC (FinQA, TAT-QA)",
              "supported_per_unsupported": SUPPORTED_PER_UNSUPPORTED, "gpu_hours": gpu_hours,
              "gpu_hours_total": sum(gpu_hours.values()), "versions": model.versions(),
              "input_sha256": data.input_hashes(), "provenance": provenance(),
              "chosen_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    (OUT / "chosen_config.json").write_text(json.dumps(chosen, indent=2, default=float))
    print(f"\nchosen: lr={lr:g}, epochs={epochs}; checkpoints {checkpoints}\nWrote {OUT}")


if __name__ == "__main__":
    main()

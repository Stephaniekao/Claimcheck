# Release Plan (week11-release)

All numbers here and in the Results Presentation come from the commit tagged `week11-release`.

## 1. Release candidate

| | |
|---|---|
| Model | DeBERTa-v3-large verifier, fine-tuned on RAGBench FinQA + TAT-QA train (lr 1e-5, 2 epochs, claim-first input) |
| Commit | The commit tagged `week11-release`. The test results were produced at local commit `53ecda0`; the DeBERTa code is byte-identical to `1e2b540` on `main` (see `DeBERTa/README.md`) |
| Saved model | The folder `DeBERTa/checkpoints/lr1e-05_ep2_seed1/` (seed 1, about 1.74 GB), loaded as a whole: `model.safetensors`, `config.json`, `tokenizer.json`, `tokenizer_config.json`. It is used with seed 1's own saved temperature and thresholds in `DeBERTa/results/inserted_setting.json` and `DeBERTa/results/ragbench_setting.json` |
| Where the checkpoint is stored | On Chengyuan's machine only; not in GitHub (git-ignored, too large). Seeds 0 and 2 are stored alongside it |

**Why seed 1.** The three seeds are three separate models, and only one can be shipped. Seed 1 has the highest Inserted Error Set **validation** F1 (0.90 vs 0.77 for seeds 0 and 2), and it meets every target on its own. Seed 2 does not: its Inserted v2 ECE is 0.101, above 0.10. The seed was picked after the test run, using validation F1 only. The three-seed mean ± std in section 2 describes the experiment; the numbers for the shipped model are seed 1's own.

## 2. Why this candidate (decision rule, Experiment Card Field 10)

Rule: meet all targets, then highest F1, then Storyboard B recall, then lower cost. Only complete rows are compared. The Uncertain cap (≤ ⅓ of claims) is read on every test set.

| Model (test split) | Status | Inserted v2 R / P / F1 | Storyboard B R (n = 25) | RAGBench claim AUROC (FinQA · TAT-QA) | Uncertain on RAGBench | Seeds | Meets all targets |
|---|---|---|---|---|---|---|---|
| Baseline | Complete | 1.00 / 1.00 / 1.00 | 1.00 | 0.57 · 0.46 (support score, added after the test run) | 73% | Deterministic, spread 0 | No: AUROC and Uncertain cap on RAGBench |
| TF-IDF + LR | Complete | 0.00 / N/A / N/A | 0.00 | 0.64 · 0.59 | | 3, std 0.000 | No |
| DeBERTa-v3-large | Complete | 0.99 / 0.80 / 0.88 ± 0.05 | 1.00 | 0.81 · 0.79 | 0% | 3 (0, 1, 2), mean ± std | Yes, on the three-seed mean at the frozen thresholds |
| LLM judge | Not run (dropped) | | | | | | |

DeBERTa is the only complete row that meets all targets, so the F1 tie-break is not needed.

DeBERTa seed 1 alone: Inserted v2 R / P / F1 0.98 / 0.91 / 0.94, ECE 0.056; RAGBench claim AUROC 0.82 · 0.79, answer AUROC 0.78 · 0.76.

## 3. Known limits

- **Few RAGBench errors caught:** at the frozen thresholds, DeBERTa's error recall is 0.19 (FinQA) and 0.26 (TAT-QA), and its precision is 0.17 and 0.09. Alarms need human review.
- **No Uncertain band:** the chosen lower and upper thresholds are equal, so DeBERTa never says Uncertain.
- **Inserted v2 ECE is close to the target:** the three-seed mean is 0.079, but the company-bootstrap 95% interval is [0.026, 0.151] and crosses 0.10.
- **Inserted Error Set is easy:** passages list each fact cleanly, and the rule-based baseline scores 1.00 on it.
- **Small error-type counts in test:** 1 to 3 claims per general error type; Storyboard B has 25.
- **Cost:** a GPU is needed. Latency is 32.4 ms (FinQA) and 19.3 ms (TAT-QA) per claim; training took 2.74 GPU hours.
- **FinanceBench** has not been run yet.

## 4. Deployment

- **Shadow first:** the rule-based baseline stays live and every Unsupported or Uncertain claim goes to a reviewer. DeBERTa runs in shadow on the same claims; its labels are logged, not shown, and compared with reviewer decisions.
- **Canary next:** if DeBERTa agrees with reviewers better than the baseline, show its labels on a small share of traffic, then grow.
- **Rollback:** the baseline is deterministic and has no weights, so it is the fallback at any time.

## 5. Before the Final Showcase

- **Run:** FinanceBench evaluation (147 test examples) for DeBERTa and the baseline.
- **Run:** human review of the sampled error cases (reviewer columns in the error-analysis CSVs are still empty).
- **Drop:** the LLM judge. We have three complete rows, no API budget, and DeBERTa already meets the targets.

## 6. Reproduce

```bash
git clone https://github.com/Stephaniekao/Claimcheck.git
cd Claimcheck
git checkout week11-release
pip install -r requirements.txt
./run_all.sh --quick        # baseline unit tests + Inserted Error Set smoke test, seconds
python -m unittest discover DeBERTa/tests   # no GPU needed
```

Test splits are run once and are not rerun. Logged test results are in `Baseline/outputs/`, `Classical/outputs/final_test/` and `DeBERTa/results/`. Rerunning DeBERTa needs the checkpoint from section 1.

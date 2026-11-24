# DeBERTa verifier

Deep row of the Experiment Card (owner: Chengyuan). DeBERTa-v3-large, starting from an NLI checkpoint ([MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli](https://huggingface.co/MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli)) and fine-tuned on RAGBench FinQA + TAT-QA train. For each claim it gives a **support score** (0–1): the probability that the cited passage supports the claim. Two thresholds chosen on validation turn the score into Supported / Uncertain / Unsupported.

It reads the same claims and passages as the baseline (`Baseline/claimcheck_baseline/evidence.py` for the Inserted Error Set, `response_sentences` / `documents_sentences` for RAGBench) and is scored by the same code (`Baseline/claimcheck_baseline/metrics.py`).

**Status:** code ready; not trained yet.

## Run

Needs a CUDA GPU (developed for one 24 GB GPU) and RAGBench in `Data/RAGBench/` (`./run_all.sh --fetch`).

```bash
python -m pip install -r requirements.txt        # numpy, pandas, pyarrow (root)
# then torch, transformers, sentencepiece, protobuf: see DeBERTa/requirements.in

python -m unittest discover DeBERTa/tests        # no GPU needed
python DeBERTa/train.py --smoke                  # ~5 min: checks the code runs, saves nothing
python DeBERTa/train.py --pilot                  # ~30 min: zero-shot vs fine-tuned AUROC, full-run time estimate
python DeBERTa/train.py                          # 4 tuning trials + 2 more seeds -> outputs/chosen_config.json
python DeBERTa/run_inserted_error_set.py         # Inserted Error Set v2 validation: temperature + thresholds
python DeBERTa/run_ragbench.py                   # RAGBench validation: temperature, thresholds, answer roll-up

python DeBERTa/run_inserted_error_set.py --final-test   # once
python DeBERTa/run_ragbench.py --final-test             # once
```

`--final-test` refuses to run if the test result file already exists (Field 5). Checkpoints go to `DeBERTa/checkpoints/` (about 1.7 GB each, git-ignored).

## Design

| Choice | What we do | Why |
|---|---|---|
| Long passages | Pack whole sentences into windows that fit 512 tokens; score the claim against each window and keep the highest score | Cutting at 512 tokens can drop the line that supports the claim |
| Training data | Every Unsupported train claim, plus 4 Supported ones per Unsupported one (sampled with the run's seed) | RAGBench finance is about 1.5% Unsupported |
| Training window | Supported: the window with most of its annotated supporting sentences (train split only). Unsupported: a random window | The annotation picks the window; it is never model input, and never read for validation or test |
| Loss | Binary log loss on the checkpoint's P(entailment) | Keeps the NLI head, so the model starts as a zero-shot NLI checker |
| Tuning | 2 learning rates (1e-5, 2e-5) x epoch 1 or 2 of a 2-epoch schedule = 4 trials, seed 0, chosen by mean claim-level AUROC on RAGBench validation | Field 5: at most 20 trials, validation only |
| Seeds | The chosen setting is also trained with seeds 1 and 2; results are reported per seed and as mean ± std | Field 5: 3 seeds |
| Calibration | Temperature scaling fitted on validation; ECE (10 bins) before and after | Field 8: the interface shows a confidence level |
| Thresholds | On validation: Uncertain ≤ 1/3 and precision ≥ 0.60, then highest strict recall. If no pair reaches precision 0.60, the highest F1 with Uncertain ≤ 1/3, and the output says so | Fields 8 and 10, AC2 |
| Where thresholds come from | Inserted Error Set: its own validation split. RAGBench: FinQA + TAT-QA validation together, one setting for both | Each dataset's thresholds are set on that dataset's validation data |
| Answer roll-up (RAGBench) | Lowest or mean claim score, chosen on validation by answer-level AUROC | Field 6, to compare with RAGBench's answer-level baseline |

DeBERTa never trains on the Inserted Error Set, so it cannot learn patterns in how those errors were generated.

## Evidence for the interface

Every prediction CSV has `evidence_used`: the passage window that gave the claim its score. For highlighting, `Verifier.explain(claim, passage)` returns the single sentence in that window that best supports the claim, copied from the passage so it can be highlighted exactly (AC3). It is only called for the claims a user looks at, never during evaluation.

## Outputs

`outputs/chosen_config.json` (trials, chosen setting, checkpoints, GPU hours), `outputs/*_setting.json` (temperature and thresholds per seed), `outputs/*_metrics_{validation,test}.json` and per-claim score CSVs. Each JSON records the package versions, input SHA256s, git commit and hardware (including the GPU), using the baseline's provenance block.

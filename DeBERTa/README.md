# DeBERTa verifier

Deep row of the Experiment Card (owner: Chengyuan). DeBERTa-v3-large, starting from an NLI checkpoint ([MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli](https://huggingface.co/MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli)) and fine-tuned on RAGBench FinQA + TAT-QA train. For each claim it gives a **support score** (0–1): the probability that the cited passage supports the claim. Two thresholds chosen on validation turn the score into Supported / Uncertain / Unsupported.

It reads the same claims and passages as the baseline (`Baseline/claimcheck_baseline/evidence.py` for the Inserted Error Set, `response_sentences` / `documents_sentences` for RAGBench) and is scored by the same code (`Baseline/claimcheck_baseline/metrics.py`).

**Status:** three-seed training, validation and final testing completed. The frozen configuration uses claim-first inputs, learning rate `1e-5`, two epochs, and seeds 0, 1 and 2. Each final evaluation command was executed once for the frozen three-seed run. FinanceBench evaluation remains pending.

## Saved final test results

Values are mean ± sample standard deviation across the three individual models. They are not an ensemble. All labels and settings were frozen before final testing.

| Dataset | Strict Unsupported recall | Unsupported precision | F1 | Claim AUROC | ECE | Uncertain rate |
|---|---:|---:|---:|---:|---:|---:|
| FinQA | 0.189 ± 0.055 | 0.169 ± 0.030 | 0.172 ± 0.005 | 0.811 ± 0.007 | 0.030 ± 0.007 | 0.000 |
| TATQA | 0.263 ± 0.062 | 0.092 ± 0.014 | 0.134 ± 0.009 | 0.787 ± 0.005 | 0.051 ± 0.008 | 0.000 |
| Inserted v2 | 0.992 ± 0.014 | 0.796 ± 0.096 | 0.881 ± 0.051 | 0.983 | 0.079 ± 0.023 | 0.000 |

RAGBench answer-level AUROC is 0.765 (FinQA) and 0.773 (TAT-QA), averaged across seeds. Each model uses mean claim-score aggregation, selected on validation, and `adherence_score` as answer gold.

Test sizes are 12,836 FinQA claims, 12,248 TAT-QA claims, and 80 Inserted v2 claims. Storyboard B strict recall is 1.000 for all three seeds. Inserted v2 flag-all F1 is 0.667; DeBERTa mean F1 is 0.881. The rule-based baseline achieves F1 1.000 on this dataset.

## Thresholds and practical limitations

- RAGBench uses a single temperature/threshold setting per seed chosen jointly on FinQA and TAT-QA validation. Inserted v2 uses its own validation split for temperature and threshold selection. DeBERTa weights are trained only on RAGBench train.
- Classical uses RAGBench-derived thresholds on Inserted v2. Comparisons of Inserted recall, precision and Uncertain rate therefore reflect both model differences and calibration protocol differences.
- The chosen lower and upper thresholds are equal for every seed, producing zero Uncertain labels. Strict and lenient recall are consequently identical. The validation search allowed an Uncertain band, but its objective selected a zero-width band.
- RAGBench claim AUROC exceeds 0.75 on both subsets. At the frozen thresholds, mean error recall is only 18.9% on FinQA and 26.3% on TAT-QA, and error precision is 16.9% and 9.2%. These alarms require human review.
- Inserted v2 mean ECE is 0.079. Its descriptive company-bootstrap 95% interval is [0.026, 0.151], based on eight test companies and 2,000 resamples. The interval crosses the 0.10 target.
- The frozen three-seed mean point estimates meet the recorded DeBERTa targets on these test splits. FinanceBench and end-user workflow effectiveness have not been established by these runs.
- Input-order and hard-negative trials failed the recorded validation adoption criteria; MiniCheck failed its screening criterion. The original claim-first models were retained. See [validation_diagnostics.md](validation_diagnostics.md) for protocols and results.

## Logged artifacts and hardware

The small original result reports are committed under `results/` so readers can inspect them without executing evaluation.

| Artifact | Contents |
|---|---|
| [ragbench_metrics_test.json](results/ragbench_metrics_test.json) | Per-seed claim/answer metrics, frozen settings, provenance and latency |
| [inserted_metrics_test.json](results/inserted_metrics_test.json) | Per-seed Inserted v2 and Storyboard B metrics |
| [supplemental_metrics.json](results/supplemental_metrics.json) | Saved test ranking hits and company-bootstrap ECE interval |
| [chosen_config.json](results/chosen_config.json) | Original training search, selected configuration and GPU hours |
| [ragbench_setting.json](results/ragbench_setting.json), [inserted_setting.json](results/inserted_setting.json) | Temperatures and thresholds fitted on validation |

Final-test results were produced at local commit `53ecda0ee18f6d7102cc0c9441e2cb361b6a6c05`. All 18 DeBERTa files tracked in that commit are byte-identical to their counterparts in published commit [`1e2b540`](https://github.com/Stephaniekao/Claimcheck/commit/1e2b540e2b4d391bbf41f99a4a06035f25c9d841), which was merged into `main` through PR #6. Publication added only `DeBERTa/.gitignore`. The DeBERTa Python source, pinned requirements, and shared `Baseline/claimcheck_baseline` modules on `main` at documentation commit `cc5c6e8` match the local evaluation commit. Later changes updated the README and archived the original reports under `results/`. The result JSONs retain their original evaluation provenance.

The recorded original training/search consumed 2.74 GPU hours, excluding later exploratory runs. Evaluation used an NVIDIA GeForce RTX 3090 on Windows, Python 3.14.0. Mean GPU inference latency was 32.4 ms/claim on FinQA and 19.3 ms/claim on TAT-QA. Checkpoints and per-claim prediction files are retained locally; trained weights are required to run frozen-model inference.

The commands below describe reproduction. The final test splits have already been evaluated. Inspect the saved reports above to read the existing results. A fresh clone lacks the local checkpoints and `outputs/` files, so an absent result file in a new checkout does not make these test splits unseen.

## Run

The recorded run used a CUDA GPU with 24 GB VRAM and RAGBench in `Data/RAGBench/`. Exact installed package versions are recorded in `requirements.txt` in this folder. The existing scripts also have a CPU fallback; CPU training time was not measured.

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

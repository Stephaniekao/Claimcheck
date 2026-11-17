## Classical Machine Learning Model

## Overview

This folder contains the classical machine learning model for ClaimCheck. We use TF-IDF, financial features, and Logistic Regression to check whether a financial claim is supported by its cited evidence.

The model is trained with two labels, Supported and Unsupported. We use thresholds selected on validation data to produce three final predictions: Supported, Unsupported, and Uncertain.

The model uses some shared parts from the `Baseline/` folder: the number normalizer, the passages for Inserted Error Set v2, and the code for precision at recall 0.80. The Experiment Card says all methods should use the same inputs (Field 12). If one of these shared parts changes, our results also change, so we save their versions with every run.

We ran the final test once, with the model, thresholds, and answer rule chosen on validation. The test results are in the Final Test Results section. The other results in this README are validation results.

## Summary of Results

We ran the final test once. The model, thresholds, and answer rule were all chosen on validation.

| Result | RAGBench test | Inserted Error Set v2 test |
|---|---:|---:|
| Answer AUROC (RAGBench) or AUROC (Inserted) | 0.582 | 0.866 |
| Strict recall | 0.000 | 0.000 |
| Lenient recall | 0.504 | 1.000 |
| Uncertain rate | 0.333 | 1.000 |
| ECE | 0.003 | 0.459 |
| Storyboard B strict recall | N/A | 0.000 |

The model does not meet the Field 10 targets. On RAGBench, it ranks claims only a little better than random, and it labels no claim as Unsupported. On Inserted Error Set v2, it ranks claims well, but the thresholds from RAGBench put every claim in the Uncertain range. The three seeds give the same model, so the spread across seeds is 0. The details are in the Final Test Results section.

## Quick Start

### See the results without running anything

All results are already saved in this folder.

| What | Where |
|---|---|
| Final test results and the Field 10 check | `outputs/final_test/final_test_results.json` |
| Our row for the comparison table | `outputs/final_test/comparison_table_row.md` |
| Reliability diagrams | `outputs/final_test/reliability_ragbench_test.png`, `reliability_inserted_test.png`, and `reliability_storyboard_b_test.png` |
| Error samples for Field 9 | `outputs/final_test/error_analysis/` |
| Validation results of the best model | `outputs/evaluation_full_best.json` |
| Results of the three seeds | `outputs/seed_summary_validation.csv` |
| Inserted Error Set v2 validation results | `outputs/inserted_error_set/inserted_validation.json` |

### Run the model again

Before the first run:

1. Make sure the `Baseline/` folder is in the repository. The model uses the shared normalizer and passages from it.
2. Create the RAGBench files with `python fetch_ragbench.py` from the main folder of the repository.
3. Install the packages from the `Classical/` folder:

```bash
pip install -r requirements.txt
```

Then run these two commands from the `Classical/` folder:

```bash
bash run_all.sh
bash run_seeds.sh
```

`run_all.sh` runs the 62 tests, the tuning with seed 42, and the Inserted Error Set v2 validation. `run_seeds.sh` runs the tuning again with seeds 43 and 44. Each tuning run takes several minutes on a laptop.

If everything is the same as our run, `run_all.sh` prints these numbers for the best trial:

| Result | Value |
|---|---:|
| Claim ROC AUC | 0.6277 |
| Answer AUROC | 0.6031 |
| Strict recall | 0.0072 |
| Uncertain rate | 0.3333 |

Small differences can appear if the library versions are different.

### About the final test

The final test was run once, and `outputs/final_test/final_test_lock.json` stops it from running again. This follows Field 5 of the Experiment Card: the test set is used only once. Please do not delete the lock file. The test results are already saved, so there is no need to run it again.

### Use the model on new claims

See step 8 in How to Run. It shows how to label new claim and evidence pairs with the saved model.

## Data

We use RAGBench FinQA and TATQA for training, validation, and testing. The data is stored in `Data/RAGBench`, and we keep the original train, validation, and test splits to avoid data leakage. The RAGBench CSV and parquet files are not stored in git. Create them with `python fetch_ragbench.py` from the main folder of the repository, or by running the first two code cells of `Data/data_download_v2.ipynb`.

The original RAGBench data is at the answer level. Our data loader converts each answer into individual claims and assigns labels based on the original annotations. The annotation fields are used for labels only, not as model inputs.

For the answer level, we use `adherence_score` as the gold label. The baseline uses the same label. This column is not in the CSV files, so the data loader reads it from the parquet files. It also checks that the rows in the two files are in the same order.

We also use Inserted Error Set v2, which contains 400 claims, for evaluation. The passages come from the shared `evidence.py`, so every method reads the same passage for each claim. The model is not trained on this set. As the Experiment Card says (Field 4), we only used its training split to develop the rules of the scope feature.

### Training Data Distribution

| Dataset | Supported | Unsupported | Total |
|---|---:|---:|---:|
| FinQA | 66,451 | 1,153 | 67,604 |
| TATQA | 94,494 | 1,780 | 96,274 |
| Total | 160,945 | 2,933 | 163,878 |

Unsupported claims make up less than 2% of the training data. This is an important problem for our model because we want to identify incorrect financial claims.

### Validation Data Distribution

| Dataset | Validation Claims |
|---|---:|
| FinQA | 9,379 |
| TATQA | 12,164 |
| Total | 21,543 |

The combined validation set contains 21,267 Supported claims and 276 Unsupported claims. Because there are only 276 Unsupported claims, small differences between results can be noise. In our experiments, a difference of about 0.02 AUROC or less should not be treated as a real difference.

## Project Structure

All paths below are relative to `Classical/`. The `Baseline/` folder must also be in the repository, because `shared.py` reads the shared parts from there.

| File | Description |
|---|---|
| `claimcheck_classical/config.py` | Model settings, random seed, tuning parameters, and output paths |
| `claimcheck_classical/shared.py` | Reads the shared normalizer, passages, and precision at recall code from `Baseline/` |
| `claimcheck_classical/data_loader.py` | Loads RAGBench (with `adherence_score`) and Inserted Error Set v2 with the shared passages |
| `claimcheck_classical/preprocessing.py` | Normalizes claim and evidence text |
| `claimcheck_classical/model_io.py` | Saves and loads the trained model and TF-IDF vectorizer |
| `claimcheck_classical/provenance.py` | Saves file checksums, library versions, hardware, and the git commit with each trained model |
| `claimcheck_classical/features/text_features.py` | Creates TF-IDF vectors and calculates cosine similarity |
| `claimcheck_classical/features/numerical_features.py` | Compares the numbers in the claim and the passage with the shared normalizer |
| `claimcheck_classical/features/period_features.py` | Checks reporting years and quarters |
| `claimcheck_classical/features/scope_features.py` | Checks financial scope for each line of the passage |
| `claimcheck_classical/features/feature_builder.py` | Combines seven features into one feature matrix |
| `claimcheck_classical/features/evidence_features.py` | Extra evidence-aware features, answer-level scores, and trial sampling |
| `claimcheck_classical/training/train.py` | Trains Logistic Regression |
| `claimcheck_classical/training/tune.py` | Runs the 20 trials and evaluates the best model |
| `claimcheck_classical/evaluation/metrics.py` | Calculates binary and three-label metrics |
| `claimcheck_classical/evaluation/calibration.py` | Calculates Brier score and ECE |
| `claimcheck_classical/evaluation/thresholds.py` | Selects the lower and upper thresholds |
| `claimcheck_classical/evaluation/inserted_validation.py` | Tests the saved model on Inserted Error Set v2 train and validation |
| `claimcheck_classical/evaluation/seed_summary.py` | Shows the mean and standard deviation of the validation results for seeds 42, 43, and 44 |
| `claimcheck_classical/evaluation/final_test.py` | Runs the final test only once and then locks it |
| `claimcheck_classical/evaluation/error_analysis.py` | Builds prediction tables and error cases |
| `claimcheck_classical/evaluation/feature_analysis.py` | Analyzes feature distributions and individual cases |
| `claimcheck_classical/evaluation/feature_diagnostics.py` | Calculates single-feature ROC-AUC and shows example claims |
| `claimcheck_classical/evaluation/feature_ablation.py` | Compares different feature sets on validation data |
| `claimcheck_classical/inference/predict.py` | Predicts three labels for new claim and evidence pairs using the saved thresholds |
| `run.py` | Older training and validation pipeline on a sample. It does not use the current threshold rule. |
| `EXPERIMENT_LOG.md` | Results of the earlier experiments, before the current version of the model |
| `requirements.txt` | Python packages needed by the Classical model |
| `run_all.sh` | Runs all tests, the full tuning run, and the Inserted Error Set v2 test |
| `run_seeds.sh` | Runs the tuning again with seeds 43 and 44, and shows the summary of the three seeds |
| `.gitignore` | Excludes trained models, caches, and old results from git |
| `tests/` | 62 unit and integration tests for the features, model pipeline, thresholds, prediction, provenance, shared parts, Inserted Error Set v2 code, and final test |

## Model Pipeline

The current pipeline follows these steps:

1. **Data Loading:** Load FinQA and TATQA claims from the original training splits.
2. **Preprocessing:** Normalize claim and evidence text.
3. **Feature Engineering:** Fit TF-IDF using training data only and calculate seven features.
4. **Training:** Train Logistic Regression using Supported = 0 and Unsupported = 1.
5. **Tuning:** Run 18 random trials over n-gram range, vocabulary size, and regularization strength C. We also compare 2 answer rules, so there are 20 trials in total. Trials are scored on the validation splits only.
6. **Validation:** Evaluate the best model on the original validation splits without fitting TF-IDF again. Calculate ECE and select two thresholds on validation data.
7. **Inserted Error Set v2:** Use the saved model and thresholds on the train and validation splits.
8. **Final Evaluation:** Test the selected model once on the RAGBench test splits and the Inserted Error Set v2 test split with `final_test.py`. This is done, and the lock file stops a second test run.

### Feature Matrix

We use four types of features: text similarity, numerical consistency, period consistency, and financial scope consistency.

Each claim and evidence pair produces seven features:

| Feature | Description |
|---|---|
| `cosine_similarity` | TF-IDF similarity between claim and evidence |
| `numerical_consistency` | Checks if every number in the claim is in the passage, using the shared normalizer |
| `period_consistency` | Checks reporting years and quarters |
| `scope_consistency` | Checks if each number has the same scope (total or segment) as its line in the passage |
| `numerical_missing` | Indicates insufficient numerical information |
| `period_missing` | Indicates insufficient period information |
| `scope_missing` | Indicates insufficient scope information |

When a consistency score cannot be determined, the feature builder replaces NaN with 0 and adds a missing indicator. These indicators are different from the final Uncertain prediction.

The numerical and scope features read numbers with the shared normalizer (version 1.3.0). It treats `$4.1 billion`, `4,100 million`, and `4.1B` as the same value, and it allows rounding. It does not read dates such as `2025-12-31` as numbers. It also checks the minus sign in the same way as the baseline, and numbers in brackets such as `(5,860)` are not counted as a conflict.

The scope feature reads the passage one line at a time, so each number gets the scope of its own line. We added the word "entire" to the total scope words. We found this word in the Inserted Error Set v2 training split, which the Experiment Card allows (Field 4).

The period feature still uses our own parser, because it can read quarters and the shared normalizer cannot.

### Model Settings

The model standardizes the features and then uses Logistic Regression with L2 regularization, the `liblinear` solver, 2,000 maximum iterations, and random seed 42.

The tuning search uses n-gram ranges (1,1), (1,2), and (1,3), vocabulary sizes 5,000, 10,000, and 20,000, and C values 0.01, 0.1, 1, and 10. The 18 trials are chosen randomly from these combinations. The best trial has the highest Average Precision on validation data.

We also choose the answer rule on validation data. The answer rule turns the claim scores of one answer into one answer score. We compare two rules: the lowest claim score (`min`) and the average claim score (`mean`). The Experiment Card says to choose this rule on the validation split (Field 6). The baseline counts this kind of choice in the 20 trials, so we do the same. This is why the search uses 18 trials.

### Thresholds

We select two thresholds on the RAGBench validation P(Supported). The rule follows Field 8 and our team decisions:

1. The lower threshold gives the highest strict recall, while Unsupported precision is at least 0.60.
2. The upper threshold is as high as possible, while the Uncertain rate stays at one third or lower. The upper threshold does not change strict recall or precision, so we can use the Uncertain label as much as AC2 allows.

Tuning and prediction use the same rule. If P(Supported) is below the lower threshold, the claim is Unsupported. If it is above the upper threshold, the claim is Supported. Otherwise, it is Uncertain. If no pair of thresholds meets the conditions, the model labels every claim as Uncertain.

Some claims are always labeled Uncertain, as Field 12 says. These are claims without a cited passage and claims that the shared normalizer cannot read. They are counted in the one third Uncertain limit, so the other claims get less space in the Uncertain label.

A claim "cannot be read" when it has a digit, but the normalizer finds no number and no year in it. Examples are "1.2x" and "Form 10-K". A claim with no digits is not counted, because the model can still score it from its text. The baseline uses a wider rule (`no_amount`), which also labels claims without numbers as Uncertain. With the wider rule, about 27% of FinQA and 36% of TATQA validation claims would be Uncertain before the model scores them. TATQA would then go over the one third limit.

The selected thresholds are saved in `models/classical_model_full_best.json`, and `predict_with_frozen_model()` reads them from there.

### Optional Extensions

The default model uses only the seven features above, which follow the Experiment Card. `config.py` has two switches for extensions. Both are off by default.

| Switch | What it adds |
|---|---|
| `USE_EXTRA_FEATURES` | Six more numeric features: sentence-level cosine similarity (maximum and top 3), token coverage, tolerant number match, claim number count, and claim length |
| `USE_CLAIM_TEXT` | TF-IDF of the claim wording as extra model input |

These extensions go beyond the Experiment Card, so we use them only for comparison unless the team agrees to change the card.

## How to Run

Run the following commands from the `Classical/` directory. The `Baseline/` folder must be in the same repository.

Before the first run, install the packages and create the RAGBench CSV and parquet files (see the Data section):

```bash
pip install -r requirements.txt
```

We need `pyarrow` to read `adherence_score` from the parquet files, and `matplotlib` to draw the reliability diagrams.

The simplest way to reproduce the main result is:

```bash
bash run_all.sh
```

This runs all tests, the full tuning run, and the Inserted Error Set v2 test. The steps below run each part separately.

### 1. Run Tests

```bash
python -m unittest discover -s tests -v
```

The test suite contains 62 tests: 11 feature tests, 5 period and extra feature tests, 4 model pipeline tests, 5 unit parsing tests, 7 integration tests, 4 threshold tests, 4 provenance tests, 6 prediction tests, 9 shared part tests, 3 Inserted Error Set v2 tests, and 4 final test tests.

### 2. Analyze Features

```bash
python -m claimcheck_classical.evaluation.feature_analysis --subset FinQA --sample-size 100 --examples 5
python -m claimcheck_classical.evaluation.feature_diagnostics
python -m claimcheck_classical.evaluation.feature_ablation
```

`feature_diagnostics` and `feature_ablation` use all Unsupported training claims and a sample of Supported claims. They are scored on the complete validation splits.

### 3. Run Initial Training

```bash
python -m claimcheck_classical.training.train --sample-size 1000
```

This command randomly selects up to 1,000 training claims from each dataset using random seed 42 and prints a coefficient table.

### 4. Run Tuning

```bash
python -m claimcheck_classical.training.tune --full-data
```

This command trains on all training claims, runs the 18 trials, chooses the answer rule, and evaluates the best model on the validation splits. It saves the trial table, the best model, predictions, coefficients, and calibration results to `outputs/` and `models/`. It also records the training time.

The `--seed` option changes the seed for the trial sampling and the solver. The main run uses seed 42.

Running `tune` without `--full-data` uses a small stratified sample. It is only a smoke test, and its scores are not meaningful because the training data is not imbalanced like the real data.

### 5. Evaluate on Inserted Error Set v2

```bash
python -m claimcheck_classical.evaluation.inserted_validation
```

This command uses the saved model and thresholds on Inserted Error Set v2 train and validation. It does not select new thresholds. It will not run the test split, because we use the test split only once at the end.

### 6. Run Three Seeds

```bash
bash run_seeds.sh
```

The Experiment Card asks for at least three seeds (Field 5). Logistic Regression always gives the same result for the same data, so the seed changes which trials are chosen and the seed of the solver. This command runs the tuning with seeds 43 and 44 and then shows the mean and the standard deviation of seeds 42, 43, and 44 on validation. Seed 42 is still the main model. The other seeds are only used to measure the spread.

### 7. Final Test (Only Once)

```bash
python -m claimcheck_classical.evaluation.final_test --confirm
```

This command uses the test splits, so we run it only once (Field 5). It needs the models of all three seeds. It tests the RAGBench test splits and the Inserted Error Set v2 test split, including the Storyboard B slice. Nothing is chosen on test. The models, thresholds, and answer rule all come from validation.

Before it reads any test data, it writes `outputs/final_test/final_test_lock.json`. After that, the command refuses to run again. Without `--confirm`, it does not run.

It saves the results, the reliability diagrams (Field 8), the Field 10 check, a row for the comparison table, and the error samples for Field 9.

### 8. Predict New Claims

```python
import pandas as pd
from claimcheck_classical.inference.predict import predict_with_frozen_model

claims = pd.DataFrame({
    "claim": ["Total revenue was $4.1 billion in 2019."],
    "evidence": ["Total revenue was 4,100 million in 2019.\nNet income was 300 million in 2019."],
})

result = predict_with_frozen_model(claims, "models/classical_model_full_best.json")
print(result[["predicted_label", "reason", "p_supported"]])
```

The evidence should store one sentence per line. The trained model file is not stored in git, so run `bash run_all.sh` first to create it.

## Saved Files

`run_all.sh` writes the following files. The model file is not committed because it can be rebuilt, but its checksum is saved in the run record.

| File | Description |
|---|---|
| `models/classical_model_full_best.joblib` | Trained model and TF-IDF vectorizer (not committed) |
| `models/classical_model_full_best.json` | Thresholds, settings, SHA256 checksums of the model, the RAGBench files, and the shared parts, normalizer version, git commit, library versions, and hardware |
| `outputs/tuning_trials_full.csv` | Results of all 18 trials |
| `outputs/evaluation_full_best.json` | Validation metrics of the best trial |
| `outputs/predictions_full_best.csv` | Validation predictions |
| `outputs/calibration_full_best.csv` | Calibration table |
| `outputs/coefficients_full_best.csv` | All model coefficients |
| `outputs/inserted_error_set/inserted_validation.json` | Inserted Error Set v2 train and validation metrics |
| `outputs/inserted_error_set/predictions_train.csv` and `predictions_validation.csv` | Inserted Error Set v2 predictions with feature values |
| `models/classical_model_full_seed43_best.*` and `..._seed44_best.*` | Models of the extra seeds (from `run_seeds.sh`) |
| `outputs/seed_runs_validation.csv` and `seed_summary_validation.csv` | Validation results for each seed, and their mean and standard deviation |
| `outputs/final_test/final_test_results.json` | All test results, the spread across seeds, and the Field 10 check |
| `outputs/final_test/final_test_lock.json` | Stops a second test run |
| `outputs/final_test/reliability_*.png` | Reliability diagrams for RAGBench test, Inserted Error Set v2 test, and the Storyboard B slice |
| `outputs/final_test/comparison_table_row.md` | Our row for the comparison table |
| `outputs/final_test/error_analysis/` | Error samples for Field 9, with empty columns for the two reviewers |

The random seed is 42. Small differences can appear when the library versions change, so the library versions are saved in the run record.

## Current Validation Results

These results use the shared normalizer, the scope feature for each line, `adherence_score` as the answer gold label, and the current threshold rule. They replace the earlier results in [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md).

### RAGBench Validation

All 18 trials gave similar results. ROC-AUC was between 0.6207 and 0.6284, Average Precision was between 0.0223 and 0.0290, and answer AUROC with the `min` rule was between 0.5568 and 0.5730. These differences are within noise. Five of the 18 trials found thresholds that meet the precision condition.

The best trial uses n-gram range (1,3), vocabulary size 10,000, and C = 10.

| Result | Value |
|---|---:|
| ROC-AUC (claims) | 0.6277 |
| Average Precision (claims) | 0.0290 |
| Answer AUROC (`mean` rule) | 0.6031 |
| Answer AUROC (FinQA) | 0.6078 |
| Answer AUROC (TATQA) | 0.5745 |
| Precision at recall 0.80 | 0.0152 |
| ECE | 0.0052 |
| Strict Recall | 0.0072 |
| Unsupported Precision | 0.6667 |
| Lenient Recall | 0.5181 |
| Uncertain Rate | 0.3333 |
| Claims the normalizer cannot read | 968 (4.5%) |

The `mean` rule was chosen on validation. Its answer AUROC is 0.6031, and the `min` rule gives 0.5713. The baseline uses the lowest claim label, so the two methods use different answer rules. The answer AUROC in `tuning_trials_full.csv` uses the `min` rule, so it cannot be compared directly with 0.6031.

For 97 validation answers, `adherence_score` and "any claim is Unsupported" give different labels. The difference is only in one direction. Every answer with an unsupported sentence also has `adherence_score` = False. But 97 answers have `adherence_score` = False and no unsupported sentence. So `adherence_score` marks 250 answers as Unsupported, and "any claim" marks 153. With the `mean` rule, the answer AUROC is 0.6031 with `adherence_score` and 0.5734 with "any claim". We use `adherence_score` because the baseline uses it, so the two methods can be compared.

The model labels only 3 validation claims as Unsupported, and 2 of them are correct. The precision condition is met, but the strict recall is close to 0. This means the model does not really find incorrect claims. Because the upper threshold now fills the Uncertain label up to one third, the lenient recall went up from 0.0072 to 0.5181. About half of the incorrect claims are sent to the analyst to check. The precision at recall 0.80 is close to the base rate of 0.013, so the model ranks claims only a little better than random.

The ECE is low because the predicted probabilities are close to the base rate of Unsupported claims, not because the model separates the two labels well.

Coefficients of the best model (standardized features, predicting Unsupported):

| Feature | Coefficient |
|---|---:|
| `cosine_similarity` | 0.2491 |
| `numerical_consistency` | -0.3037 |
| `period_consistency` | -0.5478 |
| `scope_consistency` | -0.0546 |
| `numerical_missing` | -0.4758 |
| `period_missing` | -0.6598 |
| `scope_missing` | -0.0977 |

The coefficient of `numerical_consistency` is larger than before (it was -0.2016 with our own parser, see [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md)), so the shared normalizer helps a little. The coefficient of `scope_consistency` is still close to 0. RAGBench has no scope signal, so the model does not learn to use the scope feature. The coefficient of `cosine_similarity` is still positive, which matches the single-feature result in [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md).

This model does not reach the target AUROC of 0.75 on RAGBench.

### Three Seeds

We ran the tuning with seeds 42, 43, and 44 (Field 5). All three seeds chose the same best configuration: n-gram range (1,3), vocabulary size 10,000, and C = 10. So every validation metric is the same for the three seeds, and the standard deviation is 0. Logistic Regression always gives the same result for the same data and settings. The 18 trials of each seed also show that the model is not sensitive to the settings: the ROC-AUC of all trials is between 0.6207 and 0.6288.

The training time of the best trial is 455 seconds on average (standard deviation 10 seconds), or about 0.13 CPU hours. It does not need a GPU.

### Inserted Error Set v2

We used the saved model and the RAGBench thresholds on Inserted Error Set v2. We did not select new thresholds.

| Result | Train (240) | Validation (80) | Storyboard B, validation (20) |
|---|---:|---:|---:|
| Strict recall | 0.000 | 0.000 | 0.000 |
| Lenient recall | 1.000 | 1.000 | 1.000 |
| Uncertain rate | 1.000 | 1.000 | 1.000 |
| AUROC | 0.788 | 0.814 | 0.920 |
| Average Precision (base rate 0.500) | 0.857 | 0.872 | N/A |
| Precision at recall 0.80 | 0.615 | 0.627 | N/A |
| ECE | 0.463 | 0.461 | 0.457 |

The Storyboard B AUROC on train is 0.791. We used the train split to develop the scope words, so the train results are not an independent check.

Every claim is labeled Uncertain. On both splits, P(Supported) is between 0.931 and 0.976. With the RAGBench thresholds, the Uncertain range is from 0.894 to 0.980. So every claim is inside the Uncertain range. No claim is low enough to be Unsupported, and no claim is high enough to be Supported.

The ranking is better than on RAGBench. On both splits, the median P(Supported) is about 0.971 for Supported claims and about 0.953 for Unsupported claims. This gives an AUROC of about 0.81. The precision at recall 0.80 is 0.627, so there is a threshold that meets both targets on this set. But we measured it on the same validation data, so we cannot use it as a result. The Storyboard B AUROC of 0.920 comes from only 20 claims. The scope coefficient is close to 0, so this result does not come from the scope feature. We will check which features cause it in the error analysis.

The calibration is not good. All scores are in a range of only 0.045. The model gives every claim about a 95% chance of being correct, but half of the claims are wrong. This is why the ECE is about 0.46.

Field 8 and Field 12 of the Experiment Card talk about this problem. When the tool that writes the answers changes, the calibration and the thresholds may not work anymore.

With the current rule, the model does not meet any Field 10 target on Inserted Error Set v2.

## Final Test Results

We ran `final_test.py` once. Nothing was chosen on test. The model, the thresholds, and the answer rule (`mean`) all came from validation. The three seeds give the same results, so the spread across seeds is 0.

### RAGBench Test

| Result | Test | Validation |
|---|---:|---:|
| Answer AUROC | 0.5816 | 0.6031 |
| Answer AUROC (FinQA) | 0.5678 | 0.6078 |
| Answer AUROC (TATQA) | 0.5724 | 0.5745 |
| ROC-AUC (claims) | 0.6171 | 0.6277 |
| Strict Recall | 0.0000 | 0.0072 |
| Lenient Recall | 0.5041 | 0.5181 |
| Uncertain Rate | 0.3327 | 0.3333 |
| ECE | 0.0032 | 0.0052 |

The test results are a little lower than the validation results. The difference is about 0.02, which is close to noise.

The strict recall is 0 on test. On validation, the lower threshold was set by only 3 claims, so it is very low. On test, no claim has a score below it. This matches the validation result: the model does not really label claims as Unsupported.

The Uncertain rate and the lenient recall are almost the same as on validation. So the rule that fills the Uncertain label up to one third works in the same way on test. About half of the incorrect claims are sent to the analyst to check.

The answer AUROC for both subsets together (0.5816) is higher than the answer AUROC of each subset (0.5678 and 0.5724). This is not an error. FinQA and TATQA have different rates of Unsupported answers and different score ranges, so when we put them together, part of the ranking comes from the difference between the two subsets.

### Inserted Error Set v2 Test

| Result | Test (80) |
|---|---:|
| Strict recall | 0.000 |
| Lenient recall | 1.000 |
| Uncertain rate | 1.000 |
| AUROC | 0.866 |
| ECE | 0.459 |
| Storyboard B strict recall | 0.000 |

Every claim is labeled Uncertain, the same as on validation. The scores are inside the Uncertain range of the RAGBench thresholds. The AUROC is 0.866, which is even higher than on validation (0.814), so the model still ranks these claims well. But the calibration is not good (ECE 0.459), because the model gives every claim about a 95% chance of being correct.

### Field 10

| Target | Result |
|---|---|
| Strict recall at least 0.80 on Inserted Error Set v2 | Not met (0.000) |
| Precision at least 0.60 on Inserted Error Set v2 | Not met (no claim labeled Unsupported) |
| Strict recall at least 0.80 on the Storyboard B slice | Not met (0.000) |
| Answer AUROC at least 0.75 on RAGBench | Not met (0.5816) |
| ECE at most 0.10 on RAGBench | Met (0.0032) |
| ECE at most 0.10 on Inserted Error Set v2 | Not met (0.459) |

The Classical model does not meet the Field 10 targets. As Field 10 and Field 11 say, we report this in the comparison table. The fallback in Field 11 is used only if no method meets the targets.

### Comparison Table Row

| Model | Overall (strict recall / precision / F1 / AUROC) | Storyboard B slice (strict recall) | Spread across 3 seeds (± std) | Calibration (ECE) | Training cost (GPU hours) | Inference latency (ms per claim) | Inspectability |
|---|---|---|---|---|---|---|---|
| TF-IDF + Logistic Regression | 0.00 / N/A / N/A / 0.58 | 0.00 | AUROC ± 0.000 | 0.003 (RAGBench) / 0.459 (Inserted) | 0 GPU (0.12 CPU hours) | 2.7 | 7 standardized coefficients |

Strict recall, precision, and F1 come from the Inserted Error Set v2 test split. AUROC is the answer AUROC on the RAGBench test splits. Precision and F1 are N/A because the model labels no claim as Unsupported. The inference latency includes building the features, on a laptop CPU.

The test results, the lock file, the reliability diagrams, and the error samples are in `outputs/final_test/`.

## Earlier Experiments

The results before the change to the shared normalizer, the scope feature for each line, `adherence_score`, and the current threshold rule are in [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md). We keep them to show how the model changed over time.

## Current Status and Next Steps

We have completed data loading, preprocessing, feature engineering with the shared normalizer, training, tuning with three seeds, validation metrics, calibration (ECE), threshold selection, the Inserted Error Set v2 test on train and validation, the final test, prediction for new claims with the saved thresholds, and a saved run record with checksums and versions.

The model reaches an answer AUROC of 0.5816 on the RAGBench test splits, which is below the target of 0.75. On the Inserted Error Set v2 test split, the model ranks claims well (AUROC 0.866), but the RAGBench thresholds label every claim as Uncertain. The model does not meet the Field 10 targets, and we report this as Field 10 and Field 11 say.

Our targets from the Experiment Card are at least 0.80 strict recall and at least 0.60 precision for Unsupported claims on Inserted Error Set v2, and at least 0.75 AUROC on the RAGBench test splits. The model must also reach at least 0.80 strict recall on the Storyboard B slice.

### Team Decisions

1. We select the thresholds on RAGBench validation and only report Inserted Error Set v2 validation. Precision of at least 0.60 is not possible on RAGBench because only 1.3% of the claims are Unsupported. We report it as missed, and we also report precision at recall 0.80.
2. We choose the lower threshold first. Then the upper threshold fills the Uncertain label up to one third of the claims.
3. We use `adherence_score` as the answer gold label, like the baseline.
4. We use the Inserted Error Set v2 training split only to develop the rules of the scope feature. We do not use it to train the model (Experiment Card, Fields 3 and 4).
5. We use the shared normalizer and passages from `Baseline/` (Field 12).

### Next Steps

1. Do the error analysis with the test results, because Field 9 says it comes after the final test run. The samples are in `outputs/final_test/error_analysis/`. Each file has empty columns for the two reviewers.
2. Add our row to the team comparison table.
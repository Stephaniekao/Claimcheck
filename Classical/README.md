## Classical Machine Learning Model

## Overview

This folder contains the classical machine learning model for ClaimCheck. We use TF-IDF, financial features, and Logistic Regression to check whether a financial claim is supported by its cited evidence.

The model is trained with two labels, Supported and Unsupported. We use thresholds selected on validation data to produce three final predictions: Supported, Unsupported, and Uncertain.

The model uses some shared parts from the `Baseline/` folder: the number normalizer, the passages for Inserted Error Set v2, and the code for precision at recall 0.80. The Experiment Card says all methods should use the same inputs (Field 12). If one of these shared parts changes, our results also change, so we save their versions with every run.

All results in this README are validation results. The test splits have not been used yet.

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
| `claimcheck_classical/evaluation/error_analysis.py` | Builds prediction tables and error cases |
| `claimcheck_classical/evaluation/feature_analysis.py` | Analyzes feature distributions and individual cases |
| `claimcheck_classical/evaluation/feature_diagnostics.py` | Calculates single-feature ROC-AUC and shows example claims |
| `claimcheck_classical/evaluation/feature_ablation.py` | Compares different feature sets on validation data |
| `claimcheck_classical/inference/predict.py` | Predicts three labels for new claim and evidence pairs using the saved thresholds |
| `run.py` | Older training and validation pipeline on a sample. It does not use the current threshold rule. |
| `requirements.txt` | Python packages needed by the Classical model |
| `run_all.sh` | Runs all tests, the full tuning run, and the Inserted Error Set v2 test |
| `.gitignore` | Excludes trained models, caches, and old results from git |
| `tests/test_features.py` | Unit tests for feature engineering |
| `tests/test_period_fix.py` | Tests for the period parsing fix and the extra features |
| `tests/test_model_pipeline.py` | Unit tests for the model pipeline |
| `tests/test_number_units.py` | Tests for financial unit parsing such as `4.1B` and `$300K` |
| `tests/test_thresholds.py` | Tests the threshold selection against an exhaustive search |
| `tests/test_predict.py` | Tests prediction, missing passages, and the saved thresholds |
| `tests/test_provenance.py` | Tests checksums and the saved run record |
| `tests/test_ragbench_integration.py` | Integration tests using RAGBench data |
| `tests/test_shared_components.py` | Tests the shared normalizer features, the threshold rules, and the answer rules |
| `tests/test_inserted_validation.py` | Tests the Inserted Error Set v2 code and checks that it does not run the test split |

## Model Pipeline

The current pipeline follows these steps:

1. **Data Loading:** Load FinQA and TATQA claims from the original training splits.
2. **Preprocessing:** Normalize claim and evidence text.
3. **Feature Engineering:** Fit TF-IDF using training data only and calculate seven features.
4. **Training:** Train Logistic Regression using Supported = 0 and Unsupported = 1.
5. **Tuning:** Run 18 random trials over n-gram range, vocabulary size, and regularization strength C. We also compare 2 answer rules, so there are 20 trials in total. Trials are scored on the validation splits only.
6. **Validation:** Evaluate the best model on the original validation splits without fitting TF-IDF again. Calculate ECE and select two thresholds on validation data.
7. **Inserted Error Set v2:** Use the saved model and thresholds on the train and validation splits.
8. **Final Evaluation:** Test the selected model once on the RAGBench test splits and the Inserted Error Set v2 test split. This is not done yet.

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

We need `pyarrow` to read `adherence_score` from the parquet files.

The simplest way to reproduce the main result is:

```bash
bash run_all.sh
```

This runs all tests, the full tuning run, and the Inserted Error Set v2 test. The steps below run each part separately.

### 1. Run Tests

```bash
python -m unittest discover -s tests -v
```

The test suite contains 58 tests: 11 feature tests, 5 period and extra feature tests, 4 model pipeline tests, 5 unit parsing tests, 7 integration tests, 4 threshold tests, 4 provenance tests, 6 prediction tests, 9 shared part tests, and 3 Inserted Error Set v2 tests.

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

This command trains on all training claims, runs the 18 trials, chooses the answer rule, and evaluates the best model on the validation splits. It saves the trial table, the best model, predictions, coefficients, and calibration results to `outputs/` and `models/`.

Running `tune` without `--full-data` uses a small stratified sample. It is only a smoke test, and its scores are not meaningful because the training data is not imbalanced like the real data.

### 5. Evaluate on Inserted Error Set v2

```bash
python -m claimcheck_classical.evaluation.inserted_validation
```

This command uses the saved model and thresholds on Inserted Error Set v2 train and validation. It does not select new thresholds. It will not run the test split, because we use the test split only once at the end.

### 6. Predict New Claims

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

The random seed is 42. Small differences can appear when the library versions change, so the library versions are saved in the run record.

## Current Validation Results

These results use the shared normalizer, the scope feature for each line, `adherence_score` as the answer gold label, and the current threshold rule. They replace the earlier results in the next section.

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

For 97 validation answers, `adherence_score` and "any claim is Unsupported" give different labels. This means the two labels are not the same. We use `adherence_score`, like the baseline.

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

The coefficient of `numerical_consistency` is larger than before (it was -0.2016 with our own parser), so the shared normalizer helps a little. The coefficient of `scope_consistency` is still close to 0. RAGBench has no scope signal, so the model does not learn to use the scope feature. The coefficient of `cosine_similarity` is still positive, which matches the single-feature result below.

This model does not reach the target AUROC of 0.75 on RAGBench.

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

## Earlier Experiments

These results came before we changed to the shared normalizer, the scope feature for each line, `adherence_score`, and the current threshold rule. We keep them to show how the model changed over time.

### Feature Analysis

We first tested the feature matrix using 100 FinQA training claims. The matrix had 100 rows and 7 columns.

| Result | Value |
|---|---:|
| Average cosine similarity | 0.1354 |
| Numerical missing rate | 16% |
| Period missing rate | 45% |
| Scope missing rate | 71% |

Some Supported claims received low numerical consistency scores because the rules at that time did not fully handle comparisons or calculations.

### Logistic Regression Training

We trained the first Logistic Regression model using 2,000 training claims and seven features.

| Label | Count | Percentage |
|---|---:|---:|
| Supported | 1,962 | 98.1% |
| Unsupported | 38 | 1.9% |

The training data is highly imbalanced.

### Initial Validation Results

We evaluated the model on the complete FinQA and TATQA validation splits.

| Metric | Result |
|---|---:|
| Accuracy | 0.9872 |
| Unsupported Precision | 0.0000 |
| Unsupported Recall | 0.0000 |
| Unsupported F1 | 0.0000 |
| ROC-AUC | 0.6042 |
| Average Precision | 0.0202 |

The model predicted all validation claims as Supported at threshold 0.5. The maximum predicted probability for Unsupported was only 0.0946. This explains why the model did not predict any Unsupported claims.

Although accuracy was high, the model did not detect any incorrect claims. We cannot use accuracy alone to evaluate this task.

### Period Parsing Fix

We found a bug in `period_features.py`. A year at the end of a sentence, such as "in 2019.", was not recognized. This made the period feature missing, and it made the numerical feature wrongly inconsistent because the year was treated as a financial number. We fixed the bug and added tests. The fix changed the validation metrics very little.

### Number Unit Parsing Fix

We found that unit abbreviations were read as the wrong number. For example, `$4.1B` became 4.0 and `$2.5M` became 2.0, while `$4.1 billion` was read correctly. We changed B, bn, M, mm, mn, and K to the correct values and added tests. Compared with the run before the fix, the best ROC-AUC changed from 0.6178 to 0.6181 and the answer AUROC from 0.5832 to 0.5831.

Now the features use the shared normalizer. We keep our own parser only for the older tests and tools. Our parser also read dates such as `2025-12-31` as the numbers 12 and 31, but the shared normalizer does not.

### First Full Training Data Run

This run used the original seven features on the complete training splits (163,878 claims). It was done before the two fixes above and before the trial search.

| N-gram | ROC-AUC | Average Precision | Dual Thresholds Found |
|---|---:|---:|---|
| (1,1) | 0.6178 | 0.0211 | No |
| (1,2) | 0.6181 | 0.0221 | No |
| (1,3) | 0.6185 | 0.0222 | No |

The result is almost the same for all three n-gram ranges, because the n-gram range only changes one feature, the cosine similarity. No thresholds met the precision requirement, so all claims became Uncertain.

### Full Training Data with Tuning (Earlier Version)

We ran the 20 trials on the complete training splits with our own number parser. The answer gold label was "any claim is Unsupported", and the answer score used the `min` rule.

| Result | Value |
|---|---:|
| Best n-gram range | (1,2) |
| Best vocabulary size | 10,000 |
| Best C | 10 |
| ROC-AUC | 0.6181 |
| Average Precision | 0.0291 |
| Answer AUROC | 0.5831 |
| Answer AUROC (FinQA) | 0.5324 |
| Answer AUROC (TATQA) | 0.5924 |
| ECE | 0.0052 |
| Strict Recall | 0.0072 |
| Unsupported Precision | 0.6667 |
| Uncertain Rate | 0.0000 |

The Uncertain rate was 0 because the upper threshold was set just above the lower threshold. The current rule fills the Uncertain label up to one third.

### Single-Feature Analysis

We calculated the ROC-AUC of each feature alone for detecting Unsupported claims. A value of 0.5 means the feature cannot separate the two labels. These values were calculated with our own number parser.

| Feature | ROC-AUC |
|---|---:|
| `cosine_similarity` | 0.584 |
| `numerical_consistency` | 0.493 |
| `period_consistency` | 0.540 |
| `scope_consistency` | 0.498 |
| `numerical_missing` | 0.443 |
| `period_missing` | 0.451 |
| `scope_missing` | 0.475 |

The numerical and scope features are close to 0.5. In RAGBench, many Unsupported claims have numbers that do appear in the evidence. The cosine similarity is higher for Unsupported claims, which is the opposite of what we expected. The Unsupported claims are also longer (claim length ROC-AUC 0.627).

We also checked eight Unsupported claims by hand. We found three patterns:

1. Calculated values, such as percentages and averages. These look the same as correct calculations on the surface.
2. Inference or hedging wording, such as "This suggests" or "Based on the information provided".
3. At least one claim that appears almost word for word in the evidence. This may be an annotation error.

### Feature Set Comparison

We compared different feature sets using the same Logistic Regression settings (C = 1, no tuning). We trained on all 2,933 Unsupported claims and 20,000 sampled Supported claims from each dataset. We evaluated on the complete validation splits. The answer score is the lowest claim support probability in each answer, and the answer gold label was "any claim is Unsupported".

| Feature Set | ROC-AUC | Average Precision | Answer AUROC | FinQA | TATQA |
|---|---:|---:|---:|---:|---:|
| Claim length only | 0.6382 | 0.0185 | 0.5308 | 0.5208 | 0.5058 |
| Original 7 features | 0.6214 | 0.0274 | 0.5846 | 0.5451 | 0.5891 |
| 13 numeric features | 0.6653 | 0.0252 | 0.5665 | 0.5536 | 0.5459 |
| Claim wording only | 0.7086 | 0.0369 | 0.6463 | 0.5902 | 0.7055 |
| 13 features + claim wording | 0.7296 | 0.0438 | 0.6610 | 0.6099 | 0.7040 |

The base rate of Unsupported claims is 0.013, so all Average Precision values are low.

The extra numeric features did not help much. The claim wording helped the most, but it mostly captures the style of the answer. Claims such as "calculate the" or "divide" are usually Supported, and claims such as "exceeded", "there are", or "not provided" are more often Unsupported. This is why we keep it off by default.

We think class imbalance was not the main problem. Class weights and sampling do not change how the model ranks claims, so they cannot fix a low ROC-AUC. The features themselves have limited signal.

### Inserted Error Set v2 (Constructed Claims)

Before we had the shared passages, we tested the numeric features on claims that we made with the v2 claim templates. The Temporal and Metric Confusion errors were not separated. The scope feature also did not work, because `normalize_text` removed the line breaks, so every number got the scope `mixed`. Now the scope feature reads the passage line by line. The results with the real v2 passages are in the section above.

## Current Status and Next Steps

We have completed data loading, preprocessing, feature engineering with the shared normalizer, training, tuning, validation metrics, calibration (ECE), threshold selection, the Inserted Error Set v2 test on train and validation, error analysis export, prediction for new claims with the saved thresholds, and a saved run record with checksums and versions.

The model reaches an answer-level AUROC of 0.6031 on RAGBench validation. This is below the target of 0.75. On Inserted Error Set v2 validation, the model ranks claims much better (AUROC 0.814), but the RAGBench thresholds label every claim as Uncertain. These are validation results only. If the final results do not reach the targets, we will report this as described in Field 10 and Field 11 of the Experiment Card.

Our targets from the Experiment Card are at least 0.80 strict recall and at least 0.60 precision for Unsupported claims on Inserted Error Set v2, and at least 0.75 AUROC on the RAGBench test splits. The model must also reach at least 0.80 strict recall on the Storyboard B slice.

### Team Decisions

1. We select the thresholds on RAGBench validation and only report Inserted Error Set v2 validation. Precision of at least 0.60 is not possible on RAGBench because only 1.3% of the claims are Unsupported. We report it as missed, and we also report precision at recall 0.80.
2. We choose the lower threshold first. Then the upper threshold fills the Uncertain label up to one third of the claims.
3. We use `adherence_score` as the answer gold label, like the baseline.
4. We use the Inserted Error Set v2 training split only to develop the rules of the scope feature. We do not use it to train the model (Experiment Card, Fields 3 and 4).
5. We use the shared normalizer and passages from `Baseline/` (Field 12).

### Next Steps

1. Run three seeds and report the mean and the spread (Field 5). Logistic Regression always gives the same result for the same data, so the seed changes which trials are chosen.
2. Group the errors into the categories from Field 9. We will use the Inserted Error Set v2 predictions, which include the feature values.
3. Record the training time and the prediction time per claim. On a laptop CPU, the Inserted Error Set v2 test takes about 3 ms per claim.
4. Write `final_test.py`. It will test the RAGBench test splits and the Inserted Error Set v2 test split only once, with ECE and a reliability diagram on the test split and the Storyboard B slice.

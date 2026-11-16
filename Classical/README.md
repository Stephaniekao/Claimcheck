## Classical Machine Learning Model

## Summary

- **Status:** training, tuning, thresholds, and prediction work. The test splits are not used yet.
- **Result (validation):** the default model has an answer-level AUROC of 0.583, below the 0.75 target.
- **Run:** `bash run_all.sh` (see How to Run for the data setup).
- **Open questions for the team:** see the end of this file.
  
## Overview

This folder contains the classical machine learning model for ClaimCheck. We use TF-IDF, financial features, and Logistic Regression to check whether a financial claim is supported by its cited evidence.

The model is trained with two labels, Supported and Unsupported. We use thresholds selected on validation data to produce three final predictions: Supported, Unsupported, and Uncertain.

All results in this README are preliminary validation results. The test splits have not been used yet.

## Data

We use RAGBench FinQA and TATQA for training, validation, and testing. The data is stored in `Data/RAGBench`, and we keep the original train, validation, and test splits to avoid data leakage. The RAGBench CSV files are not stored in git. Create them by running the first two code cells of `Data/data_download_v2.ipynb`, which download the official splits and save them as CSV files in `Data/RAGBench/`.

The original RAGBench data is at the answer level. Our data loader converts each answer into individual claims and assigns labels based on the original annotations. The annotation fields are used for labels only, not as model inputs.

We will also use Inserted Error Set v2, which contains 400 claims, for independent evaluation. The model is not trained on it at this time. The Experiment Card says its training split can be used to develop scope features, and we still need to decide how to use it.

### Training Data Distribution

We checked the full training splits before changing the sampling strategy.

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

All paths below are relative to `Classical/`.

| File | Description |
|---|---|
| `claimcheck_classical/config.py` | Model settings, random seed, tuning parameters, and output paths |
| `claimcheck_classical/data_loader.py` | Loads RAGBench and Inserted Error Set data |
| `claimcheck_classical/preprocessing.py` | Normalizes claim and evidence text |
| `claimcheck_classical/model_io.py` | Saves and loads the trained model and TF-IDF vectorizer |
| `claimcheck_classical/provenance.py` | Saves file checksums, library versions, and the git commit with each trained model |
| `claimcheck_classical/features/text_features.py` | Creates TF-IDF vectors and calculates cosine similarity |
| `claimcheck_classical/features/numerical_features.py` | Extracts and normalizes financial numbers |
| `claimcheck_classical/features/period_features.py` | Checks reporting years and quarters |
| `claimcheck_classical/features/scope_features.py` | Checks financial scope |
| `claimcheck_classical/features/feature_builder.py` | Combines seven features into one feature matrix |
| `claimcheck_classical/features/evidence_features.py` | Extra evidence-aware features, answer-level scores, and trial sampling |
| `claimcheck_classical/training/train.py` | Trains Logistic Regression |
| `claimcheck_classical/training/tune.py` | Runs the 20-trial search and evaluates the best model |
| `claimcheck_classical/evaluation/metrics.py` | Calculates binary and three-label metrics |
| `claimcheck_classical/evaluation/calibration.py` | Calculates Brier score and ECE |
| `claimcheck_classical/evaluation/thresholds.py` | Selects the lower and upper thresholds |
| `claimcheck_classical/evaluation/error_analysis.py` | Builds prediction tables and error cases |
| `claimcheck_classical/evaluation/feature_analysis.py` | Analyzes feature distributions and individual cases |
| `claimcheck_classical/evaluation/feature_diagnostics.py` | Calculates single-feature ROC-AUC and shows example claims |
| `claimcheck_classical/evaluation/feature_ablation.py` | Compares different feature sets on validation data |
| `claimcheck_classical/inference/predict.py` | Predicts three labels for new claim and evidence pairs using the saved thresholds |
| `run.py` | Runs the training and validation pipeline |
| `requirements.txt` | Python packages needed by the Classical model |
| `run_all.sh` | Runs all tests and the full tuning run |
| `.gitignore` | Excludes trained models, caches, and old results from git |
| `tests/test_features.py` | Unit tests for feature engineering |
| `tests/test_period_fix.py` | Tests for the period parsing fix and the extra features |
| `tests/test_model_pipeline.py` | Unit tests for the model pipeline |
| `tests/test_number_units.py` | Tests for financial unit parsing such as `4.1B` and `$300K` |
| `tests/test_thresholds.py` | Tests the threshold selection against an exhaustive search |
| `tests/test_predict.py` | Tests prediction, missing passages, and the saved thresholds |
| `tests/test_provenance.py` | Tests checksums and the saved run record |
| `tests/test_ragbench_integration.py` | Integration tests using RAGBench data |

## Model Pipeline

The current pipeline follows these steps:

1. **Data Loading:** Load FinQA and TATQA claims from the original training splits.
2. **Preprocessing:** Normalize claim and evidence text.
3. **Feature Engineering:** Fit TF-IDF using training data only and calculate seven features.
4. **Training:** Train Logistic Regression using Supported = 0 and Unsupported = 1.
5. **Tuning:** Run at most 20 random trials over n-gram range, vocabulary size, and regularization strength C. Trials are scored on the validation splits only.
6. **Validation:** Evaluate the best model on the original validation splits without fitting TF-IDF again. Calculate ECE and select two thresholds on validation data.
7. **Final Evaluation:** Test the selected model once on held-out data and on Inserted Error Set v2.

### Feature Matrix

We use four types of features: text similarity, numerical consistency, period consistency, and financial scope consistency.

Each claim and evidence pair produces seven features:

| Feature | Description |
|---|---|
| `cosine_similarity` | TF-IDF similarity between claim and evidence |
| `numerical_consistency` | Checks numerical values and financial units |
| `period_consistency` | Checks reporting years and quarters |
| `scope_consistency` | Checks financial scope |
| `numerical_missing` | Indicates insufficient numerical information |
| `period_missing` | Indicates insufficient period information |
| `scope_missing` | Indicates insufficient scope information |

When a consistency score cannot be determined, the feature builder replaces NaN with 0 and adds a missing indicator. These indicators are different from the final Uncertain prediction.

The numerical extractor supports financial units with or without spaces, such as `199million` and `199 million`.

### Model Settings

The model standardizes the features and then uses Logistic Regression with L2 regularization, the `liblinear` solver, 2,000 maximum iterations, and random seed 42.

The tuning search uses n-gram ranges (1,1), (1,2), and (1,3), vocabulary sizes 5,000, 10,000, and 20,000, and C values 0.01, 0.1, 1, and 10. The 20 trials are chosen randomly from these combinations. The best trial has the highest Average Precision on validation data.

### Thresholds

We select two thresholds on the validation P(Supported). The rule is to keep the Uncertain rate at or below one third and keep Unsupported precision at or above 0.60, while giving the highest strict recall. If no pair of thresholds meets these conditions, the model labels every claim as Uncertain.

The selected thresholds are saved in `models/classical_model_full_best.json`, and `predict_with_frozen_model()` reads them from there. A claim without a cited passage is labeled Uncertain instead of being guessed.

### Optional Extensions

The default model uses only the seven features above, which follow the Experiment Card. `config.py` has two switches for extensions. Both are off by default.

| Switch | What it adds |
|---|---|
| `USE_EXTRA_FEATURES` | Six more numeric features: sentence-level cosine similarity (maximum and top 3), token coverage, tolerant number match, claim number count, and claim length |
| `USE_CLAIM_TEXT` | TF-IDF of the claim wording as extra model input |

These extensions go beyond the Experiment Card, so we use them only for comparison unless the team agrees to change the card.

## How to Run

Run the following commands from the `Classical/` directory.

Before the first run, install the packages and create the RAGBench CSV files (see the Data section):

```bash
pip install -r requirements.txt
```

The simplest way to reproduce the main result is:

```bash
bash run_all.sh
```

This runs all tests and then the full tuning run. The steps below run each part separately.

### 1. Run Tests

```bash
python -m unittest discover -s tests -v
```

The test suite contains 46 tests: 11 feature tests, 5 period and extra feature tests, 4 model pipeline tests, 5 unit parsing tests, 7 integration tests, 4 threshold tests, 4 provenance tests, and 6 prediction tests.

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

This command trains on all training claims, runs the 20 trials, and evaluates the best model on the validation splits. It saves the trial table, the best model, predictions, coefficients, and calibration results to `outputs/` and `models/`.

Running `tune` without `--full-data` uses a small stratified sample. It is only a smoke test, and its scores are not meaningful because the training data is not imbalanced like the real data.

### 5. Run Training and Validation

```bash
python run.py --sample-size 1000
```

This command trains Logistic Regression, evaluates it on the complete FinQA and TATQA validation splits, selects thresholds, and exports error analysis.

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
| `models/classical_model_full_best.json` | Thresholds, settings, SHA256 checksums of the model and the RAGBench files, git commit, and library versions |
| `outputs/tuning_trials_full.csv` | Results of all 20 trials |
| `outputs/evaluation_full_best.json` | Validation metrics of the best trial |
| `outputs/predictions_full_best.csv` | Validation predictions |
| `outputs/calibration_full_best.csv` | Calibration table |
| `outputs/coefficients_full_best.csv` | All model coefficients |

The random seed is 42. Small differences can appear when the library versions change, so the library versions are saved in the run record.

## Initial Experiments

These results are from the first version of the model, before the changes described in the next section.

### Feature Analysis

We first tested the feature matrix using 100 FinQA training claims. The matrix had 100 rows and 7 columns.

| Result | Value |
|---|---:|
| Average cosine similarity | 0.1354 |
| Numerical missing rate | 16% |
| Period missing rate | 45% |
| Scope missing rate | 71% |

Some Supported claims received low numerical consistency scores because the current rules do not fully handle comparisons or calculations.

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

## Later Experiments

### Period Parsing Fix

We found a bug in `period_features.py`. A year at the end of a sentence, such as "in 2019.", was not recognized. This made the period feature missing, and it made the numerical feature wrongly inconsistent because the year was treated as a financial number. We fixed the bug and added tests. The fix changed the validation metrics very little.

### Number Unit Parsing Fix

We found that unit abbreviations were read as the wrong number. For example, `$4.1B` became 4.0 and `$2.5M` became 2.0, while `$4.1 billion` was read correctly. The Experiment Card says that `$4.1 billion`, `4,100 million`, and `4.1B` should become the same value. We now convert B, bn, M, mm, mn, and K to the correct values. An abbreviation is accepted only when the number has a dollar sign, a decimal point, or a comma, so a company name such as 3M is not read as a number. We added tests for this fix. It was included in the final full-data tuning run below. Compared with the run before the fix, the best ROC-AUC changed from 0.6178 to 0.6181 and the answer AUROC from 0.5832 to 0.5831, so the fix changed the result very little.

### First Full Training Data Run

This run used the original seven features on the complete training splits (163,878 claims). It was done before the two fixes above and before the 20-trial search.

| N-gram | ROC-AUC | Average Precision | Dual Thresholds Found |
|---|---:|---:|---|
| (1,1) | 0.6178 | 0.0211 | No |
| (1,2) | 0.6181 | 0.0221 | No |
| (1,3) | 0.6185 | 0.0222 | No |

The result is almost the same for all three n-gram ranges, because the n-gram range only changes one feature, the cosine similarity. No thresholds met the precision requirement, so all claims became Uncertain.

### Full Training Data with Tuning

We ran the 20-trial search on the complete training splits (163,878 claims) using the seven features. The best trial was selected by Average Precision on the validation splits.

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

All 20 trials gave almost the same result. ROC-AUC was between 0.6138 and 0.6195, Average Precision was between 0.0214 and 0.0291, and answer AUROC was between 0.5701 and 0.5885. The validation set has only 276 Unsupported claims, so these differences are within noise. The ROC-AUC is also almost the same as the first full-data run (0.6181), so the period fix and the tuning did not change the result much.

Only 4 of the 20 trials found thresholds that meet the conditions. The best trial labels only 3 validation claims as Unsupported, and 2 of them are correct. The precision requirement is met, but the strict recall is only 0.0072. This means the model does not detect incorrect claims in practice. The ECE is low because the predicted probabilities are close to the base rate of Unsupported claims, not because the model separates the two labels well. The Uncertain rate is 0 because the upper threshold is set just above the lower threshold.

Coefficients of the best model (standardized features, predicting Unsupported):

| Feature | Coefficient |
|---|---:|
| `cosine_similarity` | 0.2226 |
| `numerical_consistency` | -0.2016 |
| `period_consistency` | -0.5614 |
| `scope_consistency` | -0.0282 |
| `numerical_missing` | -0.3703 |
| `period_missing` | -0.6473 |
| `scope_missing` | -0.0928 |

The coefficient of `scope_consistency` is close to 0, which matches the single-feature result below: RAGBench has no scope signal. The coefficient of `cosine_similarity` is positive, which means that claims more similar to the evidence are slightly more likely to be Unsupported. This is the opposite of what we expected, and it matches the single-feature result.

This model does not reach the target AUROC of 0.75 or the target strict recall of 0.80.

### Single-Feature Analysis

We calculated the ROC-AUC of each feature alone for detecting Unsupported claims. A value of 0.5 means the feature cannot separate the two labels.

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

- Calculated values, such as percentages and averages. These look the same as correct calculations on the surface.
- Inference or hedging wording, such as "This suggests" or "Based on the information provided".
- At least one claim that appears almost word for word in the evidence. This may be an annotation error.

### Feature Set Comparison

We compared different feature sets using the same Logistic Regression settings (C = 1, no tuning). We trained on all 2,933 Unsupported claims and 20,000 sampled Supported claims from each dataset. We evaluated on the complete validation splits. The answer score is the lowest claim support probability in each answer.

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

### Inserted Error Set v2

We tested the numeric features on constructed claims that follow the v2 claim templates, using a passage with one fact per line. We have not tested the real v2 passages yet.

| Error Type | Separated from Supported | Reason |
|---|---|---|
| Small Numerical, Transposed Digits, Flipped Sign, Unit | Yes | The number does not match the passage |
| Temporal, Metric Confusion | No | The number and the year both appear in the passage, but we do not check that they are on the same line |
| Storyboard B | No | The scope feature does not work on line-by-line passages |

The scope feature fails because `normalize_text` removes line breaks, so every number in the passage gets the scope `mixed`. The scope keywords also do not include "entire", which the Storyboard B Unsupported claims use. We plan to make the number, period, and scope features work line by line.

## Current Status and Next Steps

We have completed data loading, preprocessing, feature engineering, training, tuning, validation metrics, calibration (ECE), threshold selection, error analysis export, prediction for new claims with the saved thresholds, and a saved run record with checksums and versions.

The default model with the original seven features reaches an answer-level AUROC of 0.5831 on the validation splits (FinQA 0.5324, TATQA 0.5924). This is below the target of 0.75. The best answer-level AUROC so far is 0.6610, and it uses the claim wording extension. The default model also labels almost no claim as Unsupported or Uncertain, so its strict recall is close to 0. These are validation results only, so we still need to validate them further. If the final results do not reach the targets, we will report this as described in Field 10 and Field 11 of the Experiment Card.

Our targets from the Experiment Card are at least 0.80 strict recall and at least 0.60 precision for Unsupported claims on Inserted Error Set v2, and at least 0.75 AUROC on the RAGBench test splits. The model must also reach at least 0.80 strict recall on the Storyboard B slice.

Next, we will:

1. Agree with the team on the threshold rules (see the questions below), then evaluate on the test splits once.
2. Make the number, period, and scope features work line by line, and evaluate on Inserted Error Set v2 and the Storyboard B slice.
3. Add calibration results on the test split and the Storyboard B slice, including a reliability diagram.
4. Group the errors into the categories from Field 9.
5. Record the training time and the inference time per claim.

We also need to discuss these questions with the team:

1. Which validation set should we use to select thresholds, RAGBench or Inserted Error Set v2? Precision of at least 0.60 is hard to reach on RAGBench because only 1.3% of the claims are Unsupported.
2. How should we choose the upper threshold so that the Uncertain label is used? Our current rule gives almost no Uncertain predictions.
3. The scope feature has no signal in RAGBench (ROC-AUC 0.498), so a model trained only on RAGBench cannot learn how to use it. Can we use the Inserted Error Set v2 training split for this?
4. Should the answer-level label come from `adherence_score` or from "any claim is Unsupported"?
5. Can we use the shared Inserted Error Set v2 passages and the shared number normalizer?
6. Can we report the extensions (extra features and claim wording) as a comparison?

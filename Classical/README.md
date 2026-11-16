## Classical Machine Learning Model

## Overview

This folder contains the classical machine learning model for ClaimCheck. We use TF-IDF, financial features, and Logistic Regression to check whether a financial claim is supported by its cited evidence.

The model is trained with two labels, Supported and Unsupported. We use thresholds selected on validation data to produce three final predictions: Supported, Unsupported, and Uncertain.

The model uses the team's shared components from `Baseline/` (Experiment Card, Field 12): the number normalizer, the Inserted Error Set passages, and the precision-at-recall metric. A change to any of them changes this model's results, so their versions and checksums are saved with every run.

All results in this README are validation results. The test splits have not been used yet. Some results depend on team decisions that are still open; these are marked **(pending)**.

## Data

We use RAGBench FinQA and TATQA for training, validation, and testing. The data is stored in `Data/RAGBench`, and we keep the original train, validation, and test splits to avoid data leakage. The RAGBench CSV and parquet files are not stored in git. Create them with `python fetch_ragbench.py` from the repository root, or by running the first two code cells of `Data/data_download_v2.ipynb`.

The original RAGBench data is at the answer level. Our data loader converts each answer into individual claims and assigns labels based on the original annotations. The annotation fields are used for labels only, not as model inputs.

The answer-level gold label is `adherence_score`, the same label the baseline uses. It is not one of the 12 CSV columns, so the loader reads it from the parquet file saved by the same download step and checks that the rows line up with the CSV.

We also use Inserted Error Set v2 (400 claims) for evaluation. The passages are built by the team's shared `evidence.py`, so every method reads the same passage for each claim. The model is not trained on this set. Following the Experiment Card (Field 4), its training split was used only to develop the scope feature rules.

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

All paths below are relative to `Classical/`. The model needs the `Baseline/` folder in the same repository, because `shared.py` imports the shared components from there.

| File | Description |
|---|---|
| `claimcheck_classical/config.py` | Model settings, random seed, tuning parameters, and output paths |
| `claimcheck_classical/shared.py` | Imports the team's shared normalizer, passages, and precision-at-recall metric from `Baseline/` |
| `claimcheck_classical/data_loader.py` | Loads RAGBench (with `adherence_score`) and the Inserted Error Set with shared passages |
| `claimcheck_classical/preprocessing.py` | Normalizes claim and evidence text |
| `claimcheck_classical/model_io.py` | Saves and loads the trained model and TF-IDF vectorizer |
| `claimcheck_classical/provenance.py` | Saves file checksums, library versions, hardware, and the git commit with each trained model |
| `claimcheck_classical/features/text_features.py` | Creates TF-IDF vectors and calculates cosine similarity |
| `claimcheck_classical/features/numerical_features.py` | Compares claim and passage amounts with the shared normalizer |
| `claimcheck_classical/features/period_features.py` | Checks reporting years and quarters |
| `claimcheck_classical/features/scope_features.py` | Checks financial scope, one passage line at a time |
| `claimcheck_classical/features/feature_builder.py` | Combines seven features into one feature matrix |
| `claimcheck_classical/features/evidence_features.py` | Extra evidence-aware features, answer-level scores, and trial sampling |
| `claimcheck_classical/training/train.py` | Trains Logistic Regression |
| `claimcheck_classical/training/tune.py` | Runs the 20-trial search and evaluates the best model |
| `claimcheck_classical/evaluation/metrics.py` | Calculates binary and three-label metrics |
| `claimcheck_classical/evaluation/calibration.py` | Calculates Brier score and ECE |
| `claimcheck_classical/evaluation/thresholds.py` | Selects the lower and upper thresholds |
| `claimcheck_classical/evaluation/inserted_validation.py` | Evaluates the saved model on Inserted Error Set v2 train and validation |
| `claimcheck_classical/evaluation/error_analysis.py` | Builds prediction tables and error cases |
| `claimcheck_classical/evaluation/feature_analysis.py` | Analyzes feature distributions and individual cases |
| `claimcheck_classical/evaluation/feature_diagnostics.py` | Calculates single-feature ROC-AUC and shows example claims |
| `claimcheck_classical/evaluation/feature_ablation.py` | Compares different feature sets on validation data |
| `claimcheck_classical/inference/predict.py` | Predicts three labels for new claim and evidence pairs using the saved thresholds |
| `run.py` | Older training and validation pipeline on a sample (does not use the current threshold rule) |
| `requirements.txt` | Python packages needed by the Classical model |
| `run_all.sh` | Runs all tests, the full tuning run, and the Inserted Error Set evaluation |
| `.gitignore` | Excludes trained models, caches, and old results from git |
| `tests/test_features.py` | Unit tests for feature engineering |
| `tests/test_period_fix.py` | Tests for the period parsing fix and the extra features |
| `tests/test_model_pipeline.py` | Unit tests for the model pipeline |
| `tests/test_number_units.py` | Tests for financial unit parsing such as `4.1B` and `$300K` |
| `tests/test_thresholds.py` | Tests the threshold selection against an exhaustive search |
| `tests/test_predict.py` | Tests prediction, missing passages, and the saved thresholds |
| `tests/test_provenance.py` | Tests checksums and the saved run record |
| `tests/test_ragbench_integration.py` | Integration tests using RAGBench data |
| `tests/test_shared_components.py` | Tests the shared normalizer features, the team threshold rules, and answer aggregation |
| `tests/test_inserted_validation.py` | Tests the Inserted Error Set evaluation and that it refuses the test split |

## Model Pipeline

The current pipeline follows these steps:

1. **Data Loading:** Load FinQA and TATQA claims from the original training splits.
2. **Preprocessing:** Normalize claim and evidence text.
3. **Feature Engineering:** Fit TF-IDF using training data only and calculate seven features.
4. **Training:** Train Logistic Regression using Supported = 0 and Unsupported = 1.
5. **Tuning:** Run 18 random trials over n-gram range, vocabulary size, and regularization strength C, plus 2 answer aggregation rules, for 20 trials in total. Trials are scored on the validation splits only.
6. **Validation:** Evaluate the best model on the original validation splits without fitting TF-IDF again. Calculate ECE and select two thresholds on validation data.
7. **Inserted Error Set:** Apply the saved model and thresholds to Inserted Error Set v2 train and validation.
8. **Final Evaluation:** Test the selected model once on the RAGBench test splits and the Inserted Error Set v2 test split (not done yet).

### Feature Matrix

We use four types of features: text similarity, numerical consistency, period consistency, and financial scope consistency.

Each claim and evidence pair produces seven features:

| Feature | Description |
|---|---|
| `cosine_similarity` | TF-IDF similarity between claim and evidence |
| `numerical_consistency` | Checks that every claim amount appears in the passage, using the shared normalizer |
| `period_consistency` | Checks reporting years and quarters |
| `scope_consistency` | Checks that each amount has the same scope (total or segment) as its passage line |
| `numerical_missing` | Indicates insufficient numerical information |
| `period_missing` | Indicates insufficient period information |
| `scope_missing` | Indicates insufficient scope information |

When a consistency score cannot be determined, the feature builder replaces NaN with 0 and adds a missing indicator. These indicators are different from the final Uncertain prediction.

**Numbers.** The numerical and scope features read numbers with the team's shared normalizer (normalizer 1.3.0). So `$4.1 billion`, `4,100 million`, and `4.1B` are the same value, rounding is allowed, dates such as `2025-12-31` are not read as amounts, and a minus sign must agree as in the baseline's sign check. Accounting brackets such as `(5,860)` never count as a conflict.

**Scope.** The scope feature reads the original passage one line at a time, so each amount takes the scope of its own line. The keyword "entire" was added to the total scope words. It was found while developing the feature on the Inserted Error Set v2 training split, as the Experiment Card allows (Field 4).

**Period.** The period feature still uses our own parser, because it also reads quarters, which the shared normalizer does not.

### Model Settings

The model standardizes the features and then uses Logistic Regression with L2 regularization, the `liblinear` solver, 2,000 maximum iterations, and random seed 42.

The tuning search uses n-gram ranges (1,1), (1,2), and (1,3), vocabulary sizes 5,000, 10,000, and 20,000, and C values 0.01, 0.1, 1, and 10. The 18 trials are chosen randomly from these combinations. The best trial has the highest Average Precision on validation data.

The rule that combines claim scores into an answer score is also chosen on validation, from two candidates: the lowest claim score (`min`) and the average claim score (`mean`). The Experiment Card (Field 6) asks for this rule to be chosen on validation. Like the baseline, we count these two candidates toward the 20-trial limit (Field 5), so the search uses 18 trials. **(pending: team to confirm)**

### Thresholds

We select two thresholds on the RAGBench validation P(Supported), following Field 8 and the team's decisions:

1. **Lower threshold:** the highest strict recall with Unsupported precision of at least 0.60.
2. **Upper threshold:** set as high as possible while Uncertain stays at most one third of claims. The upper threshold does not change strict recall or precision, so this uses the Uncertain label as much as AC2 allows.

The decision rule is the same in tuning and inference: P(Supported) below the lower threshold is Unsupported, above the upper threshold is Supported, and Uncertain otherwise. If no pair of thresholds meets the conditions, the model labels every claim as Uncertain.

**Claims that are always Uncertain (Field 12).** A claim without a cited passage, or a claim the shared normalizer cannot read, is labeled Uncertain instead of being guessed. These claims count toward the one-third Uncertain limit, so the scored claims get only the rest of it.

We define "cannot read" as: the claim contains a digit, but the normalizer finds neither an amount nor a period in it (for example "1.2x" or "Form 10-K"). A claim with no digits is not a failure; the model still scores it from its text. The baseline's `no_amount` rule is wider and also makes claims without numbers Uncertain. With the wider rule, about 27% of FinQA and 36% of TATQA validation claims would be Uncertain before the model scores anything, so TATQA alone would break the one-third limit. **(pending: team to confirm the definition)**

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

`pyarrow` is needed to read `adherence_score` from the parquet files.

The simplest way to reproduce the main result is:

```bash
bash run_all.sh
```

This runs all tests, the full tuning run, and the Inserted Error Set evaluation. The steps below run each part separately.

### 1. Run Tests

```bash
python -m unittest discover -s tests -v
```

The test suite contains 58 tests: 11 feature tests, 5 period and extra feature tests, 4 model pipeline tests, 5 unit parsing tests, 7 integration tests, 4 threshold tests, 4 provenance tests, 6 prediction tests, 9 shared component tests, and 3 Inserted Error Set evaluation tests.

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

This command trains on all training claims, runs the 18 trials, chooses the answer aggregation rule, and evaluates the best model on the validation splits. It saves the trial table, the best model, predictions, coefficients, and calibration results to `outputs/` and `models/`.

Running `tune` without `--full-data` uses a small stratified sample. It is only a smoke test, and its scores are not meaningful because the training data is not imbalanced like the real data.

### 5. Evaluate on Inserted Error Set v2

```bash
python -m claimcheck_classical.evaluation.inserted_validation
```

This command applies the saved model and thresholds to Inserted Error Set v2 train and validation. The thresholds are not refitted. The test split is refused; it is evaluated once at the end.

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
| `models/classical_model_full_best.json` | Thresholds, settings, SHA256 checksums of the model, the RAGBench files and the shared components, normalizer version, git commit, library versions, and hardware |
| `outputs/tuning_trials_full.csv` | Results of all 18 trials |
| `outputs/evaluation_full_best.json` | Validation metrics of the best trial |
| `outputs/predictions_full_best.csv` | Validation predictions |
| `outputs/calibration_full_best.csv` | Calibration table |
| `outputs/coefficients_full_best.csv` | All model coefficients |
| `outputs/inserted_error_set/inserted_validation.json` | Inserted Error Set v2 train and validation metrics |
| `outputs/inserted_error_set/predictions_{train,validation}.csv` | Inserted Error Set predictions with feature values |

The random seed is 42. Small differences can appear when the library versions change, so the library versions are saved in the run record.

## Current Validation Results

These results use the shared normalizer, line-by-line scope, `adherence_score` as the answer gold, and the current threshold rule. They replace the earlier results below.

### RAGBench Validation

All 18 trials gave similar results. ROC-AUC was between 0.6207 and 0.6284, Average Precision was between 0.0223 and 0.0290, and answer AUROC with the `min` rule was between 0.5568 and 0.5730. These differences are within noise. Five of the 18 trials found thresholds that meet the precision condition.

The best trial uses n-gram range (1,3), vocabulary size 10,000, and C = 10.

| Result | Value |
|---|---:|
| ROC-AUC (claims) | 0.6277 |
| Average Precision (claims) | 0.0290 |
| Answer AUROC (`mean` rule) | 0.6031 |
| Answer AUROC, FinQA | 0.6078 |
| Answer AUROC, TATQA | 0.5745 |
| Precision at recall 0.80 | 0.0152 |
| ECE | 0.0052 |
| Strict Recall | 0.0072 |
| Unsupported Precision | 0.6667 |
| Lenient Recall | 0.5181 |
| Uncertain Rate | 0.3333 |
| Claims the normalizer cannot read | 968 (4.5%) |

**Answer rule.** The `mean` rule was chosen on validation (answer AUROC 0.6031, against 0.5713 for `min`). The baseline uses the lowest claim label, so the two rows of the comparison table use different answer rules. The answer AUROC values in `tuning_trials_full.csv` use `min`, so they cannot be compared directly with 0.6031.

**Answer gold.** `adherence_score` and "any claim is Unsupported" give different labels for 97 validation answers. So the two labels are not identical, and we use `adherence_score`, as the baseline does.

**Thresholds.** The model labels only 3 validation claims as Unsupported, and 2 of them are correct. The precision condition is met, but strict recall is close to 0, so the model does not detect incorrect claims in practice. Because the upper threshold now fills the Uncertain label up to one third, lenient recall rose from 0.0072 to 0.5181: about half of the incorrect claims are sent to the analyst to check. Precision at recall 0.80 is close to the base rate of 0.013, so the model ranks claims only a little better than chance.

**Calibration.** The ECE is low because the predicted probabilities are close to the base rate of Unsupported claims, not because the model separates the two labels well.

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

The coefficient of `numerical_consistency` is stronger than before (-0.2016 with our own parser), so the shared normalizer helps a little. The coefficient of `scope_consistency` is still close to 0: RAGBench has no scope signal, so the model does not learn to use the scope feature. The coefficient of `cosine_similarity` is still positive, which matches the single-feature result below.

The model does not reach the target AUROC of 0.75 on RAGBench.

### Inserted Error Set v2

We applied the saved model and the RAGBench thresholds to Inserted Error Set v2. The thresholds were not refitted.

| | Train (240) | Validation (80) | Storyboard B, validation (20) |
|---|---:|---:|---:|
| Strict recall | 0.000 | 0.000 | 0.000 |
| Lenient recall | 1.000 | 1.000 | 1.000 |
| Uncertain rate | 1.000 | 1.000 | 1.000 |
| AUROC | 0.788 | 0.814 | 0.920 |
| Average Precision (base rate 0.500) | 0.857 | 0.872 | — |
| Precision at recall 0.80 | 0.615 | 0.627 | — |
| ECE | 0.463 | 0.461 | 0.457 |

Storyboard B AUROC on train is 0.791. The train split was used to develop the scope keywords, so its results are development results.

**Every claim is labeled Uncertain.** On Inserted Error Set v2, P(Supported) lies between 0.931 and 0.976 on both splits. The RAGBench thresholds put Uncertain between 0.894 and 0.980, so every claim falls inside the Uncertain band. No claim is low enough to be Unsupported or high enough to be Supported.

**The ranking is better than on RAGBench.** The median P(Supported) is about 0.971 for Supported claims and about 0.953 for Unsupported claims, on both splits, which gives the AUROC of about 0.81. Precision at recall 0.80 is 0.627, so a threshold that meets both targets exists on this set. But it was measured on the same validation data, so it is not a result. The Storyboard B AUROC of 0.920 comes from only 20 claims, and the scope coefficient is close to 0, so it does not come from the scope feature. We will check which features produce it in the error analysis.

**Calibration fails.** All scores lie in a band 0.045 wide. The model gives every claim about a 95% chance of being correct, while half of them are wrong, so the ECE is about 0.46.

This is the case Field 8 and Field 12 warn about: when the tool that writes the answers changes, the calibration and the thresholds stop working. Whether to refit the thresholds on Inserted Error Set v2 validation is an open question for the team (see below).

With the current rule, the model meets none of the Field 10 targets on Inserted Error Set v2.

## Earlier Experiments

The results in this section came before the change to the shared normalizer, line-by-line scope, `adherence_score`, and the current threshold rule. We keep them as a record of how the model developed.

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

We found that unit abbreviations were read as the wrong number. For example, `$4.1B` became 4.0 and `$2.5M` became 2.0, while `$4.1 billion` was read correctly. We converted B, bn, M, mm, mn, and K to the correct values and added tests. Compared with the run before the fix, the best ROC-AUC changed from 0.6178 to 0.6181 and the answer AUROC from 0.5832 to 0.5831.

The features now use the shared normalizer instead. Our own parser is kept only for the older tests and tools. It also read dates such as `2025-12-31` as the numbers 12 and 31, which the shared normalizer does not.

### First Full Training Data Run

This run used the original seven features on the complete training splits (163,878 claims). It was done before the two fixes above and before the trial search.

| N-gram | ROC-AUC | Average Precision | Dual Thresholds Found |
|---|---:|---:|---|
| (1,1) | 0.6178 | 0.0211 | No |
| (1,2) | 0.6181 | 0.0221 | No |
| (1,3) | 0.6185 | 0.0222 | No |

The result is almost the same for all three n-gram ranges, because the n-gram range only changes one feature, the cosine similarity. No thresholds met the precision requirement, so all claims became Uncertain.

### Full Training Data with Tuning (earlier version)

We ran the 20-trial search on the complete training splits using our own number parser. The answer gold was "any claim is Unsupported", and the answer score used the `min` rule.

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

The Uncertain rate was 0 because the upper threshold was set just above the lower threshold. The current rule fills the Uncertain label up to one third instead.

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

- Calculated values, such as percentages and averages. These look the same as correct calculations on the surface.
- Inference or hedging wording, such as "This suggests" or "Based on the information provided".
- At least one claim that appears almost word for word in the evidence. This may be an annotation error.

### Feature Set Comparison

We compared different feature sets using the same Logistic Regression settings (C = 1, no tuning). We trained on all 2,933 Unsupported claims and 20,000 sampled Supported claims from each dataset. We evaluated on the complete validation splits. The answer score is the lowest claim support probability in each answer, and the answer gold was "any claim is Unsupported".

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

### Inserted Error Set v2 (constructed claims)

Before the shared passages were available, we tested the numeric features on constructed claims that follow the v2 claim templates. Temporal and Metric Confusion errors were not separated, and the scope feature did not work, because `normalize_text` removed line breaks and every number got the scope `mixed`. The scope feature now reads the passage line by line, and the results on the real v2 passages are in the section above.

## Current Status and Next Steps

We have completed data loading, preprocessing, feature engineering with the shared normalizer, training, tuning, validation metrics, calibration (ECE), threshold selection, the Inserted Error Set v2 evaluation on train and validation, error analysis export, prediction for new claims with the saved thresholds, and a saved run record with checksums and versions.

The model reaches an answer-level AUROC of 0.6031 on RAGBench validation, below the target of 0.75. On Inserted Error Set v2 validation, it ranks claims much better (AUROC 0.814), but the RAGBench thresholds label every claim Uncertain. These are validation results only. If the final results do not reach the targets, we will report this as described in Field 10 and Field 11 of the Experiment Card.

Our targets from the Experiment Card are at least 0.80 strict recall and at least 0.60 precision for Unsupported claims on Inserted Error Set v2, and at least 0.75 AUROC on the RAGBench test splits. The model must also reach at least 0.80 strict recall on the Storyboard B slice.

### Decisions made with the team

1. Thresholds are selected on RAGBench validation. Inserted Error Set v2 validation is only reported. Precision of at least 0.60 cannot be reached on RAGBench at a 1.3% base rate, so it is reported as missed, together with precision at recall 0.80.
2. The lower threshold is chosen first. The upper threshold then fills the Uncertain label up to one third of claims.
3. The answer-level gold is `adherence_score`, as in the baseline.
4. The Inserted Error Set v2 training split is used only to develop the scope feature rules, not to fit model weights (Experiment Card, Fields 3 and 4).
5. The model uses the team's shared normalizer and passages (Field 12).

### Open questions for the team

1. **Normalizer failures:** should "cannot read" use our narrower definition or the baseline's `no_amount` rule? The wider rule would break the one-third Uncertain limit on TATQA.
2. **Trial budget:** should the two answer aggregation rules count toward the 20 trials, as in the baseline?
3. **Answer gold:** 97 validation answers have different labels under `adherence_score` and "any claim is Unsupported". The team should know this, since it affects every method's answer-level results.
4. **Thresholds on Inserted Error Set v2:** keep the RAGBench thresholds and report that every claim is Uncertain, or refit thresholds (with Platt or temperature scaling) on Inserted Error Set v2 validation as Field 8 allows? Refitting would need the same rule for DeBERTa and the LLM, and the thresholds would sit inside a band 0.045 wide, chosen on only 80 claims.
5. **Extensions:** can we report the extra features and the claim wording as a comparison?

### Next steps

1. Agree on the open questions above.
2. Run three seeds and report the mean and spread (Field 5). Logistic Regression is deterministic, so the seed changes the trial sampling.
3. Group the errors into the categories from Field 9, using the Inserted Error Set predictions with feature values.
4. Record the training time and the inference time per claim. The Inserted Error Set evaluation takes about 3 ms per claim on a laptop CPU.
5. Write `final_test.py`, which evaluates the RAGBench test splits and the Inserted Error Set v2 test split once, with ECE and a reliability diagram on the test split and the Storyboard B slice.

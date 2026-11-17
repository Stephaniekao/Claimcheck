# Classical Model: Experiment Log

This file keeps the results from before we changed to the shared normalizer, the scope feature for each line, `adherence_score`, and the current threshold rule. We keep them to show how the model changed over time. The current results are in [README.md](README.md).

## Feature Analysis

We first tested the feature matrix using 100 FinQA training claims. The matrix had 100 rows and 7 columns.

| Result | Value |
|---|---:|
| Average cosine similarity | 0.1354 |
| Numerical missing rate | 16% |
| Period missing rate | 45% |
| Scope missing rate | 71% |

Some Supported claims received low numerical consistency scores because the rules at that time did not fully handle comparisons or calculations.

## Logistic Regression Training

We trained the first Logistic Regression model using 2,000 training claims and seven features.

| Label | Count | Percentage |
|---|---:|---:|
| Supported | 1,962 | 98.1% |
| Unsupported | 38 | 1.9% |

The training data is highly imbalanced.

## Initial Validation Results

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

## Period Parsing Fix

We found a bug in `period_features.py`. A year at the end of a sentence, such as "in 2019.", was not recognized. This made the period feature missing, and it made the numerical feature wrongly inconsistent because the year was treated as a financial number. We fixed the bug and added tests. The fix changed the validation metrics very little.

## Number Unit Parsing Fix

We found that unit abbreviations were read as the wrong number. For example, `$4.1B` became 4.0 and `$2.5M` became 2.0, while `$4.1 billion` was read correctly. We changed B, bn, M, mm, mn, and K to the correct values and added tests. Compared with the run before the fix, the best ROC-AUC changed from 0.6178 to 0.6181 and the answer AUROC from 0.5832 to 0.5831.

Now the features use the shared normalizer. We keep our own parser only for the older tests and tools. Our parser also read dates such as `2025-12-31` as the numbers 12 and 31, but the shared normalizer does not.

## First Full Training Data Run

This run used the original seven features on the complete training splits (163,878 claims). It was done before the two fixes above and before the trial search.

| N-gram | ROC-AUC | Average Precision | Dual Thresholds Found |
|---|---:|---:|---|
| (1,1) | 0.6178 | 0.0211 | No |
| (1,2) | 0.6181 | 0.0221 | No |
| (1,3) | 0.6185 | 0.0222 | No |

The result is almost the same for all three n-gram ranges, because the n-gram range only changes one feature, the cosine similarity. No thresholds met the precision requirement, so all claims became Uncertain.

## Full Training Data with Tuning (Earlier Version)

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

## Single-Feature Analysis

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

## Feature Set Comparison

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

## Inserted Error Set v2 (Constructed Claims)

Before we had the shared passages, we tested the numeric features on claims that we made with the v2 claim templates. The Temporal and Metric Confusion errors were not separated. The scope feature also did not work, because `normalize_text` removed the line breaks, so every number got the scope `mixed`. Now the scope feature reads the passage line by line. The results with the real v2 passages are in [README.md](README.md).

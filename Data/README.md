# Data Preparation and Split Notes

This document explains how the datasets were prepared and split for the ClaimCheck project, with a focus on preventing data leakage and reproducing the results.

## 1. RAGBench

**Source:** https://huggingface.co/datasets/galileo-ai/ragbench

**Subsets:** FinQA and TAT-QA

### Data Preparation

The datasets were downloaded from Hugging Face using the `datasets` Python library.

We retained the official training, validation, and test splits without additional random splitting.

The following 12 columns were preserved:

`id`, `question`, `documents`, `response`, `documents_sentences`, `response_sentences`, `sentence_support_information`, `unsupported_response_sentence_keys`, `overall_supported_explanation`, `relevance_explanation`, `dataset`, `split`.

The `dataset` and `split` columns identify the original subset and split. No changes were made to the original questions, responses, documents, or annotations. Nested fields may be serialized when exported to CSV.

### Notes for Model Development

The following fields contain reference annotations or explanations and must not be used as model inputs:

- `sentence_support_information`
- `unsupported_response_sentence_keys`
- `overall_supported_explanation`
- `relevance_explanation`

Claim decomposition, label conversion, and other model preprocessing steps were not performed during data preparation.

## 2. Inserted Error Set

**Final size:** 400 claims from 40 companies

**Source:** SEC financial filings and XBRL data

### Data Leakage Prevention

To reduce the risk of data leakage, we screened candidate companies against FinanceBench company information and the processed RAGBench FinQA and TAT-QA datasets.

Companies identified as overlapping with these datasets were excluded from the eligible company pool. For RAGBench, company name matches were screened across the processed questions, documents, and responses.

We then selected 40 distinct companies and assigned each company to only one split. This prevents claims from the same company from appearing across training, validation, and testing.

The selected companies and their assignments are recorded in `locked_company_split_40.csv`.

### Dataset Generation and Splits

The dataset contains 300 General Claims from 30 companies and 100 Storyboard B claims from 10 additional companies.

General Claims include Supported, Numerical, Temporal, Metric Confusion, and Unit examples. Storyboard B focuses on errors where segment revenue is incorrectly presented as the company's total revenue.

| Dataset | Training | Validation | Testing | Total |
|---|---:|---:|---:|---:|
| General Claims | 210 | 60 | 30 | 300 |
| Storyboard B | 30 | 20 | 50 | 100 |
| **Total** | **240** | **80** | **80** | **400** |

The overall split is 60% training, 20% validation, and 20% testing. Storyboard B has more testing examples to support evaluation of segment revenue errors.

Both correct claims and claims containing inserted errors are retained.

### Download and Reproduction

The dataset was generated from SEC financial filings. We saved the source snapshots, company assignments, and SHA256 checksums so that team members can reproduce the results without downloading SEC data again.

To reproduce the dataset:

1. Download the complete Claimcheck project folder from GitHub, including the `Data` directory.
2. Make sure `Data/Inserted Error Set/source_snapshots/`, `locked_company_split_40.csv`, and `reproducibility_manifest_400.json` are included.
3. Open the reproduction notebook in VS Code or Jupyter Notebook and run all cells.

The notebook uses the saved SEC snapshots and locked company assignments to regenerate the 400 claims. It verifies the source files and compares the SHA256 hashes of all eight regenerated CSV files against the original outputs.

**Reproduction has been successfully tested.** All 400 claims were regenerated, and all eight CSV hashes matched the original manifest. The original files were not modified.

Please use the reproduction notebook for verification. The original generation notebook is retained for reference and should not be run unless a new dataset version is intentionally required.

### Notes for Model Development

The training split is intended for developing baseline rules and classical model scope features. DeBERTa and the LLM Judge will not be trained on this dataset.

Reference values, labels, and error information should be preserved for evaluation but excluded from model inputs where they would reveal the correct answer.

## 3. FinanceBench

**Source:** https://huggingface.co/datasets/PatronusAI/financebench

### Split Design

| Split | Examples |
|---|---:|
| Testing | 147 |
| Blind Test | 3 |
| **Total** | **150** |

FinanceBench is reserved for testing only and is not used for training or threshold tuning.

The three blind test examples are kept separate for the final showcase and excluded from model development.

## 4. General Notes

- All models should use the same predefined dataset splits.
- The Inserted Error Set uses company level splitting to prevent company overlap across splits.
- Companies were screened against FinanceBench and RAGBench before selection to reduce potential data leakage.
- Original annotations and reference information should be preserved for evaluation but not included as model inputs.
- The saved company assignments and source snapshots should be used for reproducibility.
- Changes to the source data, generation rules, or split assignments should be documented.
- Successful SHA256 verification confirms reproducibility, but does not replace manual review of financial claims.

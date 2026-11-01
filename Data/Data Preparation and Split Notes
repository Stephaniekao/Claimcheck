# Data Preparation and Split Notes

This document records how the datasets were downloaded, organized, and split for the ClaimCheck project. It is intended to help team members understand and reuse the data.

## 1. RAGBench

**Source:** https://huggingface.co/datasets/galileo-ai/ragbench

**Subsets:** FinQA and TAT-QA

### Data Preparation

The datasets are downloaded from Hugging Face using the `datasets` Python library.

We use the official training, validation, and test splits without additional random splitting.

The following 12 columns are retained:

`id`, `question`, `documents`, `response`, `documents_sentences`, `response_sentences`, `sentence_support_information`, `unsupported_response_sentence_keys`, `overall_supported_explanation`, `relevance_explanation`, `dataset`, `split`.

The `dataset` and `split` columns identify the source subset and the original split.

No changes are made to the original questions, responses, documents, or annotations. Nested fields may be serialized when exporting to CSV.

### Notes for Model Development

The following fields contain reference annotations or explanations and should not be used as model inputs:

- `sentence_support_information`
- `unsupported_response_sentence_keys`
- `overall_supported_explanation`
- `relevance_explanation`

Claim decomposition, label conversion, and other model preprocessing steps are not performed during data splitting.

## 2. Inserted Error Set

**Planned size:** 400 examples

### Split Design

The dataset will be split by company to prevent information leakage across financial filings.

| Split | Target |
|---|---:|
| Training | 60% |
| Validation | 20% |
| Testing | 20% |

The exact proportions may vary depending on the number of examples available for each company.

Approximately 80 to 100 examples will involve segment revenue errors. The target is around 20 such examples in validation and at least 40 in testing.

Both unchanged correct claims and claims with inserted errors will be retained.

### Notes

The training split is intended for developing baseline rules and classical model scope features. DeBERTa and the LLM Judge will not be trained on this dataset.

The actual split counts and company overlap checks should be recorded after splitting.

## 3. FinanceBench

**Source:** https://huggingface.co/datasets/PatronusAI/financebench

### Split Design

| Split | Examples |
|---|---:|
| Testing | 147 |
| Blind Test | 3 |
| Total | 150 |

FinanceBench is reserved for testing only and is not used for training or threshold tuning.

The three blind test examples will be kept separate for the final showcase and excluded from model development.

## 4. General Notes

- All models should use the same predefined splits.
- Original annotations should be preserved for evaluation.
- Reference labels and explanations must not be included in model inputs.
- Any changes to split assignments should be documented.
- Final example counts and data validation results should be added after processing.

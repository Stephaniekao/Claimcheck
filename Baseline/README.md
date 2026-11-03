# Rule-based baseline

Baseline row of the Experiment Card (owner: Sharon). It uses number normalization and keyword matching, with no training, no randomness and no API. Each claim gets Supported / Uncertain / Unsupported, plus the rule that fired, an explanation, and the passage lines it used. It gives no score, so AUROC and ECE do not apply.

## Run

```bash
python3 Baseline/run_inserted_error_set.py      # train + validation
python3 Baseline/run_ragbench.py                # RAGBench validation: 16-trial config search
python3 Baseline/run_inserted_error_set.py --final-test   # test, once, after rules are frozen
python3 Baseline/run_ragbench.py --final-test
python3 -m unittest discover Baseline/tests
```

**The test split has not been run yet.** Results go to `outputs/`; each JSON records the code versions and input SHA256s.

## Rules

For each amount in the claim, after normalization (so `$4.1 billion` = `4,100 million` = `4.1B`, with rounding allowed for):

| Condition | Label |
|---|---|
| Same amount on a passage line with the same line item, period and scope | Supported |
| A line with the same item and the same stated period gives a different amount | Unsupported |
| Amount nowhere in the passage | Unsupported |
| Amount appears, but the key words only partly match (e.g. "entire" vs "segment") or name another item | Uncertain |
| Amount not in the passage, but the sentence computes it (ratio, % change) | Uncertain |
| No readable number, or no passage | Uncertain |

A claim is Unsupported if any of its amounts is Unsupported, and Supported only if all of them are Supported.

## Shared inputs (other methods should reuse these)

- **Inserted Error Set passages:** `evidence.load_inserted_error_set()` builds them. Each passage lists every annual fact the cited 10-K reports, from `source_snapshots.zip`; Storyboard B passages list every segment line plus total revenue. The passage depends only on the filing, never on the claim or label.
- **RAGBench claims** are `response_sentences`; gold labels come from `unsupported_response_sentence_keys` (sentences) and `adherence_score` (answers). RAGBench `id` is not unique across generator models, so group answers by row, not by `id`.
- **Number normalizer:** `normalize.py`.

## Validation results

| | Strict R | Lenient R | Precision | F1 | Uncertain |
|---|---|---|---|---|---|
| Inserted Error Set (80) | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 |
| Storyboard B slice (20) | 1.00 | 1.00 | 1.00 | — | 0.00 |
| RAGBench FinQA, claims (9,379) | 0.14 | 0.79 | 0.04 | 0.06 | 0.73 |
| RAGBench TAT-QA, claims (12,164) | 0.10 | 0.76 | 0.03 | 0.05 | 0.75 |

Chosen RAGBench config: `rel_tol=0`, `min_word_overlap=0.3`, `derived_policy=uncertain`, answer roll-up = lowest claim label.

**Notes for the team:**
- **The Inserted Error Set is saturated.** The claims are templated and the evidence is a clean table, so simple rules get everything right.
- **Flagging every claim also passes the precision target**, because 80% of the Inserted Error Set claims are wrong (precision 0.80, recall 1.0).
- **Storyboard B scope errors come out Unsupported, not Uncertain** as Field 3 expected, because the passage includes the total-revenue line.
- **The Uncertain cap (AC2, ≤ ⅓) cannot be met on RAGBench.** About 26% of sentences have no number, and most of the rest compute values that aren't written in the passage.

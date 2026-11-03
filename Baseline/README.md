# Rule-based baseline

Baseline row of the Experiment Card (owner: Sharon). Number normalization plus keyword matching, with no training, no randomness and no API. Each claim gets Supported / Uncertain / Unsupported, plus the rule that fired, an explanation, and the passage lines it used. It gives no score, so AUROC, ECE and the precision–recall curve do not apply.

**Status:** frozen at rules 1.2.0 / normalizer 1.2.0. The one-time test run is done (Field 5), so any further rule change is a new version and must not be re-judged on this test split.

## Run

```bash
python3 Baseline/run_inserted_error_set.py      # train + validation
python3 Baseline/run_ragbench.py                # RAGBench validation: 16-trial config search
python3 Baseline/error_analysis.py              # Field 9 samples from the saved test outputs
python3 -m unittest discover Baseline/tests
# done once already; do not repeat:  --final-test  on both runners
```

Results go to `outputs/`; each JSON records the code versions and input SHA256s.

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

A claim is Unsupported if any of its amounts is Unsupported, and Supported only if all of them are Supported. RAGBench config, chosen on validation: `rel_tol=0`, `min_word_overlap=0.3`, `derived_policy=uncertain`, answer roll-up = lowest claim label.

## Test results

| | n | Strict R | Lenient R | Precision | F1 | False alarms | Uncertain | Flag-all F1 |
|---|---|---|---|---|---|---|---|---|
| Inserted Error Set | 80 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 | 0.89 |
| Storyboard B slice | 50 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 | 0.89 |
| RAGBench FinQA, claims | 12,836 | 0.18 | 0.83 | 0.05 | 0.08 | 0.06 | 0.73 | 0.03 |
| RAGBench FinQA, answers | 2,294 | 0.33 | 0.90 | 0.10 | 0.15 | 0.28 | 0.66 | 0.16 |
| RAGBench TAT-QA, claims | 12,248 | 0.05 | 0.74 | 0.02 | 0.03 | 0.02 | 0.73 | 0.02 |
| RAGBench TAT-QA, answers | 3,338 | 0.12 | 0.85 | 0.06 | 0.08 | 0.07 | 0.72 | 0.07 |

Flag-all F1 is what you get by labelling every claim Unsupported. Latency is about 1 ms per claim on a laptop CPU. Validation results were within a few points of these, with TAT-QA strict recall 0.095 on validation vs 0.048 on test.

**Against Field 10:**
- **Inserted Error Set:** it meets every target that applies (strict recall ≥ 0.80, precision ≥ 0.60, Storyboard B recall ≥ 0.80). But flagging every claim also gets recall 1.0 and precision 0.80, so these targets can't tell methods apart.
- **RAGBench:** the baseline breaks the Uncertain cap (AC2, ≤ ⅓) at 73% Uncertain, and at answer level on FinQA it's no better than flagging everything. Both are noted for the team.

## Error analysis (Field 9)

The baseline made no errors on the Inserted Error Set, so there are none to inspect there. For RAGBench, `outputs/error_analysis/` has, for each subset, 30 missed errors and 20 false alarms (seed 42), each with its passage. Each row has a suggested category from a keyword heuristic, plus empty `reviewer_1_category` and `reviewer_2_category` columns for the team's manual pass. See [outputs/error_analysis/summary.md](outputs/error_analysis/summary.md).

- **Missed errors (38 per subset):** every one was labelled Supported by the strongest rule (number and key words both matched), which is the card's worst case. The numbers really are in the passage; what's wrong is what the sentence asserts with them, or the GPT-4 annotator may be wrong.
- **False alarms:** mostly values the answer calculated (percent of total, sums), plus wording differences between claim and table row.

## Decisions the other methods should share

- **Inserted Error Set passages:** `evidence.load_inserted_error_set()` builds them. Each passage lists every annual fact the cited 10-K reports, from `source_snapshots.zip`; Storyboard B passages list every segment line plus total revenue. The passage depends only on the filing, never on the claim or label. The team needs to agree to these passages, and the other three methods should use them too.
- **Storyboard B result differs from Field 3:** Field 3 expected these claims to come out Uncertain. Because the passage states total revenue, they come out Unsupported instead; without a total line they would be Uncertain, and `tests/` covers both cases. The card's text should be updated.
- **RAGBench claims** are `response_sentences`; gold labels come from `unsupported_response_sentence_keys` (sentences) and `adherence_score` (answers), both produced by RAGBench's GPT-4 annotator, not humans. `id` repeats across generator models, so group answers by row. Once the team's claim splitter replaces the sentences, every method's results change.

## Known limitations (not fixed, because the test run is done)

- **Signs:** amounts are compared by absolute value, because text often carries the sign in words ("net cash used in …"). A pure sign error (+5bn vs −5bn) is not caught. The Inserted Error Set has no sign errors.
- **Calculation cue words are broad** ("about", "over", "margin", "approximately"). A wrong number in such a sentence becomes Uncertain rather than Unsupported.
- **The line-item lexicon is small** (15 items), so RAGBench rows with business-specific labels fall back to word overlap.
- **Scope:** the baseline covers claim verification only. Claim splitting, retrieval and the interface are other parts of the pipeline.
- **FinanceBench is not evaluated:** its 147 answers still need claim splitting and hand labels.

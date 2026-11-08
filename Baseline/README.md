# Rule-based baseline

Baseline row of the Experiment Card (owner: Sharon). Number normalization plus keyword matching, with no training, no randomness and no API. Each claim gets Supported / Uncertain / Unsupported, plus the rule that fired, an explanation, and the passage lines it used. A rule-based **support score** (0–1) was added later, after the test run; see [Support score](#support-score-added-after-the-test-run).

**Status:** rules 1.3.0 / normalizer 1.3.0, which adds a sign check for the Flipped Sign errors in Inserted Error Set v2 and was developed on v2 train and validation only.
- **RAGBench:** the official test result stays rules 1.2.0. That split was used once, so 1.3.0 is checked on RAGBench validation only, where it changes 4 of about 21,500 claims.
- **Inserted Error Set v2:** tested once with rules 1.3.0 at commit `fafeaee` (results below).
- **Support score 1.0.0:** added after the 1.2.0 test run. It never changes a label.

## Run

```bash
./run_all.sh                                    # all of the below on train + validation (see --help)

python3 Baseline/run_inserted_error_set.py      # train + validation
python3 Baseline/run_ragbench.py --no-search    # RAGBench validation with the saved config
python3 Baseline/run_ragbench.py                # (re-runs the 16-trial config search)
python3 Baseline/run_support_score.py           # support score with the saved weights
python3 Baseline/error_analysis.py              # Field 9 samples from the saved test outputs
python3 -m unittest discover Baseline/tests
```

`--final-test` is run once per test split, by hand. The RAGBench test was used with rules 1.2.0, so `run_ragbench.py --final-test` refuses later versions; they are checked on validation instead.

Results go to `outputs/`. Each JSON records the code versions, input SHA256s, git commit, Inserted Error Set version and hardware. Results for Inserted Error Set v1 (rules 1.2.0, including its test run) are in `outputs/archive/inserted-v1_rules-1.2.0/`.

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

| | Rules | n | Strict R | Lenient R | Precision | F1 | False alarms | Uncertain | Flag-all F1 |
|---|---|---|---|---|---|---|---|---|---|
| Inserted Error Set v2 | 1.3.0 | 80 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 | 0.67 |
| Storyboard B slice (v2) | 1.3.0 | 50 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 | 0.67 |
| RAGBench FinQA, claims | 1.2.0 | 12,836 | 0.18 | 0.83 | 0.05 | 0.08 | 0.06 | 0.73 | 0.03 |
| RAGBench FinQA, answers | 1.2.0 | 2,294 | 0.33 | 0.90 | 0.10 | 0.15 | 0.28 | 0.66 | 0.16 |
| RAGBench TAT-QA, claims | 1.2.0 | 12,248 | 0.05 | 0.74 | 0.02 | 0.03 | 0.02 | 0.73 | 0.02 |
| RAGBench TAT-QA, answers | 1.2.0 | 3,338 | 0.12 | 0.85 | 0.06 | 0.08 | 0.07 | 0.72 | 0.07 |

Flag-all F1 is what you get by labelling every claim Unsupported. Latency is about 1 ms per claim on a laptop CPU. RAGBench validation results were within a few points of these, with TAT-QA strict recall 0.095 on validation vs 0.048 on test.

- **v2 test, every error type caught:** Small Numerical 3/3, Transposed Digits 2/2, Flipped Sign 1/1, Temporal 3/3, Metric Confusion 3/3, Unit 3/3, Storyboard B 25/25, and all 40 correct claims Supported.
- **Small error-type counts:** the general test split has only 15 wrong claims, and a single Flipped Sign. Train and validation (17 Flipped Sign claims, all caught by 1.3.0, 4 by 1.2.0) are the better evidence for the sign check.
- **Support score on v2 test:** AUROC 1.00, but with only two distinct scores, so it says nothing about ranking.
- **v1 history** (rules 1.2.0, 80% Unsupported): 1.00 on every metric, flag-all F1 0.89. Files are in `outputs/archive/inserted-v1_rules-1.2.0/`.

**Against Field 10:**
- **Inserted Error Set v2:** it meets every target that applies (strict recall ≥ 0.80, precision ≥ 0.60, Storyboard B recall ≥ 0.80). With v2's 50/50 labels, flagging every claim now gets precision 0.50 and fails the precision target, so the target separates real methods from the trivial one. The set is still easy for number lookup: the passages list each fact as "item for date: value".
- **RAGBench:** the baseline breaks the Uncertain cap (AC2, ≤ ⅓) at 73% Uncertain, and at answer level on FinQA it's no better than flagging everything. Both are noted for the team.

## Support score (added after the test run)

`claimcheck_baseline/score.py`, `run_support_score.py`; results in `outputs/support_score/`.

**What it is.** A number from 0 to 1 for how strongly the passage supports the claim under the baseline's own rules. It is a *support score*, not a confidence that the label is right, so it is not calibrated and ECE still does not apply.

**How it works.**
- **The label sets the band:** Unsupported 0–⅓, Uncertain ⅓–⅔, Supported ⅔–1. Thresholds ⅓ and ⅔ give back exactly the rules' labels, and the runner checks this against the saved predictions on every split and stops if any label differs. So a right number next to the wrong line item, year or segment can never score as Supported.
- **Four features rank claims inside the band:**
  - same number (unit, scale and, from rules 1.3.0, sign are part of this check, so there is no separate unit feature)
  - same line item, or the share of the claim's key words found on the line
  - same year
  - same segment/total scope

  A feature the claim or line doesn't mention is skipped, and the other weights are rescaled. Each amount takes its best passage line; a claim takes its weakest amount.
- **Weights:** chosen on RAGBench validation by mean claim-level AUROC, from 4 settings. That brings the total to 20 trials, the Field 5 limit. The winner was number 0.7 / item 0.1 / year 0.1 / scope 0.1; all four settings were within 0.015 AUROC of each other.

**Test results** (single run; risk = 1 − score; Unsupported is the positive class):

| | AUROC | Average precision (base rate) | Precision at recall 0.80 | AUROC from the label alone |
|---|---|---|---|---|
| Inserted Error Set (and Storyboard B) | 1.00 | 1.00 (0.80) | 1.00 | 1.00 |
| RAGBench FinQA, claims | 0.57 | 0.029 (0.017) | 0.019 | 0.57 |
| RAGBench FinQA, answers | 0.49 | 0.092 (0.085) | 0.085 | — |
| RAGBench TAT-QA, claims | 0.46 | 0.011 (0.012) | 0.012 | 0.50 |
| RAGBench TAT-QA, answers | 0.55 | 0.040 (0.035) | 0.041 | — |

**What it shows.**
- **RAGBench:** the score is close to chance, well short of the 0.75 target. Wrong and correct claims are labelled Supported at the same rate (about 21% each on FinQA validation), so ranking inside the bands can't help much; on TAT-QA it is slightly worse than the label alone. Matching numbers and words does not separate RAGBench's errors, which gives the trained models a concrete floor to beat.
- **Inserted Error Set:** it is again saturated. Only two distinct scores occur, so 1.00 says nothing about ranking.
- **How to report it:** report it next to the official 1.2.0 row, marked as added after the test run. We had already read the 1.2.0 test errors when we designed it.

## Error analysis (Field 9)

The baseline made no errors on the Inserted Error Set, so there are none to inspect there. For RAGBench, `outputs/error_analysis/` has, for each subset, 30 missed errors and 20 false alarms (seed 42), each with its passage. Each row has a suggested category from a keyword heuristic, plus empty `reviewer_1_category` and `reviewer_2_category` columns for the team's manual pass. See [outputs/error_analysis/summary.md](outputs/error_analysis/summary.md).

- **Missed errors (38 per subset):** every one was labelled Supported by the strongest rule (number and key words both matched), which is the card's worst case. The numbers really are in the passage; what's wrong is what the sentence asserts with them, or the GPT-4 annotator may be wrong.
- **False alarms:** mostly values the answer calculated (percent of total, sums), plus wording differences between claim and table row.

## Decisions the other methods should share

- **Inserted Error Set passages:** `evidence.load_inserted_error_set()` builds them. Each passage lists every annual fact the cited 10-K reports, from `source_snapshots.zip`; Storyboard B passages list every segment line plus total revenue. The passage depends only on the filing, never on the claim or label. The team needs to agree to these passages, and the other three methods should use them too.
- **Storyboard B result differs from Field 3:** Field 3 expected these claims to come out Uncertain. Because the passage states total revenue, they come out Unsupported instead; without a total line they would be Uncertain, and `tests/` covers both cases. The card's text should be updated.
- **RAGBench claims** are `response_sentences`; gold labels come from `unsupported_response_sentence_keys` (sentences) and `adherence_score` (answers), both produced by RAGBench's GPT-4 annotator, not humans. `id` repeats across generator models, so group answers by row. Once the team's claim splitter replaces the sentences, every method's results change.

## Sign check (rules 1.3.0)

A claim amount written with a minus does not match a passage amount written as a plain positive, and the reverse. Example: the filing says "Net Income (Loss) …: USD 7,085 million" and the claim says "USD -7,085 million". To avoid false alarms, a sign is ignored when it can't be trusted:
- **Subtraction and dashes:** only a minus attached to the number counts ("-7,085", "USD -7,085", "$-5"). "$469 - $77" and "- Total …" do not.
- **Accounting brackets:** "(5,860)" never conflicts, because tables bracket costs that the answer states as positive.
- **Negative wording:** a plain number next to "used in", "net loss", "decreased" and similar words may mean a negative, so it never conflicts. Bracketed alternatives in a line label ("Net Income (Loss)", "Provided by (Used in)") don't count as negative wording.

`Config(sign_check=False)` reproduces rules 1.2.0. On v2 train and validation the check catches 17 of 17 Flipped Sign claims (1.2.0 caught 4), with no other label changed. A first version that treated brackets and subtraction as signs raised RAGBench false alarms by about a point, so it was narrowed before being kept.

## Known limitations

- **Signs:** a flip is not caught when the plain side uses negative wording or brackets (see above), or when the sign is written only in words ("a loss of $5 million" vs "income of $5 million").
- **Calculation cue words are broad** ("about", "over", "margin", "approximately"). A wrong number in such a sentence becomes Uncertain rather than Unsupported.
- **The line-item lexicon is small** (15 items), so RAGBench rows with business-specific labels fall back to word overlap.
- **Support score is not a confidence:** it ranks claims by rule agreement and is not calibrated; on RAGBench it is near chance.
- **Scope:** the baseline covers claim verification only. Claim splitting, retrieval and the interface are other parts of the pipeline.
- **FinanceBench is not evaluated:** its 147 answers still need claim splitting and hand labels.

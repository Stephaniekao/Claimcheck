# DeBERTa validation diagnostics

The formal configuration is the three-seed claim-first run on commit `59390f5`, using learning rate `1e-5` and two epochs. RAGBench uses a shared threshold policy. Inserted Error Set v2 remains the evaluation version. The results below use validation data.

- The original NLI checkpoint scored 0.521/0.533 AUROC on FinQA/TAT-QA claim-first and 0.602/0.621 evidence-first, on a fixed 1,104-claim sample with enriched Unsupported prevalence. Swapping inputs in the already fine-tuned seed-0 checkpoint instead reduced AUROC from 0.830/0.824 to 0.686/0.725 on this sample.
- On full validation, three-seed mean precision@20/50/100 was 26.7%/22.7%/20.7% for FinQA and 33.3%/28.7%/20.0% for TAT-QA. Random review would average 1.60% and 1.04%, respectively.
- The seed-0 review sampled 20 false positives and 20 false negatives per subset. Six false positives and zero false negatives were within 0.05 of the decision cutoff. Numeric reasoning and potentially inconsistent labels appeared in manual checks. A numeric anchor was lost to truncation in one false negative. Its effect on the prediction remains unverified.
- Averaging the three scores and fitting separate thresholds gave held-out validation F1 of 0.202 and 0.220, compared with 0.169 and 0.171 for the mean single-model performance with shared thresholds. TAT-QA caught 28 versus 38 true errors while false alarms fell from approximately 281 to 100. This is exploratory given the card's shared-setting design and priority on avoiding missed errors.
- On Inserted v2, leave-one-company-out affine calibration had mean ECE 0.101, versus 0.119 for temperature scaling and 0.110 for raw scores. Its log loss and Brier score did not improve over the raw scores.

## Input-order trial 5, recorded before running

Start from the original NLI checkpoint. Use the same seed (0), learning rate (1e-5), two-epoch schedule, training examples, windows and optimizer as the original selected run. Swap the order in training and evaluation and truncate the evidence segment. Evaluate the checkpoint at epoch 2 on full RAGBench validation.

Consider a three-seed confirmation only if the mean of the two claim-level AUROCs rises by strictly more than 0.016 compared with the existing seed-0 checkpoint, and precision@20 and @50 do not fall on either subset. The 0.016 is a practical margin based on observed seed variation, not a significance test. Report precision@100 as an additional result. Seed-0 references are FinQA AUROC 0.827840503 and hits 6/13/22, plus TAT-QA AUROC 0.828583494 and hits 8/16/20 at k=20/50/100.

### Decision rule amendment before validation results

Recorded on 2026-11-15 at 02:05 EST, while the first training epoch was still running and the trial directory had no validation outputs. This rule supersedes the initial rule above. Compare the arithmetic mean of the FinQA and TAT-QA claim-level AUROC differences. Evidence-first meets the validation criterion if the observed difference is strictly above 0.016 and the lower endpoint of a two-sided 95% paired bootstrap interval is strictly above zero.

Draw 2,000 replicates with a fixed resampling seed of 2026. Within each subset, resample complete answers with replacement, keeping every claim in each selected answer and applying the same draw to both models. Calculate the difference in each subset, average the two differences for each replicate, and use its 2.5th and 97.5th percentiles. Report top-20, top-50 and top-100 hits and precision for each subset without a pass/fail gate. Record any loss of hits when discussing the review workflow.

The interval describes validation-answer sampling variation for the two fixed seed-0 checkpoints. It does not estimate training-seed variability. The 0.016 margin is a practical decision threshold, and a positive result supports consideration for independent seed confirmation rather than an automatic model replacement. The running trial's pre-amendment top-k gate is retained in its original output for audit purposes. Use the separate paired-bootstrap report for the amended decision.

### Trial 5 result

The complete validation comparison covered 9,379 FinQA and 12,164 TAT-QA claims. The initial scoring script stopped at a non-unique join key. Scoring recovered from the saved two-epoch checkpoint by including the answer index in the join. Training ran once.

| Subset | Claim-first AUROC | Evidence-first AUROC | Claim-first hits @20/50/100 | Evidence-first hits @20/50/100 |
| --- | ---: | ---: | ---: | ---: |
| FinQA | 0.82784 | 0.81681 | 6 / 13 / 22 | 5 / 11 / 13 |
| TAT-QA | 0.82858 | 0.82378 | 8 / 16 / 20 | 6 / 14 / 20 |

The mean AUROC difference (evidence-first minus claim-first) was −0.00792, with a 95% paired answer-bootstrap interval of [−0.02538, +0.00755]. Trial 5 failed the predeclared rule. The fixed-budget review hit counts also favored the existing configuration. Freeze the original three-seed claim-first model and its saved validation settings for the formal comparison.

### Hard-negative trial 6 protocol, recorded before running

Use the frozen, original seed-0 model to score all windows of each Unsupported RAGBench training claim. Keep its highest-scoring window using the original inference path. Recreate the original seed-0 training pairs and replace only the Unsupported windows, preserving their labels and row order and all Supported pairs. Start from the original NLI weights and retain claim-first inputs, 14,665 pairs, learning rate 1e-5, and two epochs. Use full RAGBench validation for comparison. Draw 30 mined pairs at random with a fixed seed for a label sanity check. Record possible labeling issues without filtering or relabeling any pairs.

Advance only if the mean claim-level AUROC gain exceeds 0.016, the lower limit of the two-sided 95% paired answer-bootstrap interval (2,000 draws, seed 2026) exceeds zero, and the combined FinQA and TAT-QA top-50 hit total is at least 27. The original total is 29 (13 + 16); a decrease of at least three blocks adoption. Also report top-20 and top-100 counts. If the trial passes, confirm with seeds 1 and 2, using the same cached windows mined by the original seed-0 model. Otherwise retain the original configuration. No test split is part of the trial.

### Trial 6 result

The first training process exited at around batch 1,000 without a checkpoint. The completed retry used the saved mined windows. The final comparison used all 21,543 validation claims.

| Subset | Original AUROC | Hard-negative AUROC | Original hits @20/50/100 | Hard-negative hits @20/50/100 |
| --- | ---: | ---: | ---: | ---: |
| FinQA | 0.82784 | 0.81512 | 6 / 13 / 22 | 7 / 10 / 19 |
| TAT-QA | 0.82858 | 0.83072 | 8 / 16 / 20 | 6 / 13 / 20 |

Mean AUROC difference: −0.00529; paired 95% interval: [−0.02318, +0.01198]. Combined top-50 hits declined from 29 to 23. The trial did not qualify for confirmation, so retain the original checkpoints and saved validation settings. The fixed 30-pair training-label sample was exported separately for review; its labels were not used to modify training.

### MiniCheck screen, recorded before scoring

Score the MiniCheck DeBERTa-v3-large release at revision `2f2d01a` without updating weights. Use the original 512-token evidence windows, document before claim and the class-1 support probability, with max aggregation over evidence windows. First score the same fixed 1,104 validation claims as the trial-5 input-order study. Stop if mean claim-level AUROC is below 0.70. At or above 0.70, score full RAGBench validation. A full-validation AUROC of at least 0.78 is the minimum signal for considering matched seed-0 trial 7. A direct replacement would require the trial-6 gain and top-50 criteria (mean AUROC gain >0.016, paired answer-bootstrap 95% lower bound >0, and at least 27 combined top-50 hits, against the original seed 0), plus review of the method change for Experiment Card Field 3. Select shared temperature and thresholds on validation using the original procedure. Sample-fit settings are diagnostic only. No test data are scored.

MiniCheck scored 0.5861 on FinQA (600 claims) and 0.5560 on TAT-QA (504 claims), for mean AUROC 0.5711. It failed the 0.70 screen, so full validation and trial 7 were skipped. Keep the original three-seed configuration and validation thresholds.

### Frozen final evaluation

For the DeBERTa decision, apply ECE ≤ 0.10 separately to the three-seed mean claim-level ECE for each scored dataset (FinQA, TAT-QA and Inserted Error Set v2 overall). The target stays at 0.10. Use the original claim-first checkpoints and the settings already fitted on validation, including the shared RAGBench decision rule. Score each held-out split once. Report top-20, top-50 and top-100 hits and an Inserted v2 company-bootstrap ECE interval as descriptive supplements.

# Archived: Inserted Error Set v1, rules 1.2.0

Results of the rule-based baseline (rules 1.2.0, normalizer 1.2.0, support score 1.0.0)
on Inserted Error Set **v1** (80% Unsupported), including its one test run (2026-11-03).
v1 has since been replaced by v2 (50/50, new error types), so these are history, not
current results.

- `inserted_*`: labels and metrics on v1 train / validation / test
- `support_score/`: support scores on v1, and the score metrics files (which also hold
  the RAGBench support-score results, still current for RAGBench test)
- `error_analysis/`: v1 test errors (there were none) and the summary as written then
- `ragbench_metrics_validation_rules-1.2.0.json`: RAGBench validation with rules 1.2.0,
  kept because the main file now holds the rules 1.3.0 validation check

The RAGBench **test** results for rules 1.2.0 were not moved: they are still the official
RAGBench test result (`../../ragbench_*_test_*`), since that split is used only once.

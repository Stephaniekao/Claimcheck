#!/usr/bin/env bash
# Run the two extra seeds for the spread across seeds (Experiment Card, Field 5).
# Seed 42 is the main run from run_all.sh. The test splits are never loaded.
#
# Usage, from the Classical/ directory, after run_all.sh:
#   bash run_seeds.sh

set -euo pipefail

cd "$(dirname "$0")"

for seed in 43 44; do
    echo "========== TUNING WITH SEED ${seed} =========="
    python -m claimcheck_classical.training.tune --full-data --seed "${seed}"
    echo
done

echo "========== SUMMARY OF SEEDS 42, 43, 44 (VALIDATION) =========="
python -m claimcheck_classical.evaluation.seed_summary

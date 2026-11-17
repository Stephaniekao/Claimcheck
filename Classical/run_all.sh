#!/usr/bin/env bash
# Run the complete validation workflow for the Classical model.
# The test splits are never loaded.
#
# Usage, from the Classical/ directory:
#   bash run_all.sh

set -euo pipefail

cd "$(dirname "$0")"

echo "========== 1. TESTS =========="
python -m unittest discover -s tests -p "test_*.py"

echo
echo "========== 2. TUNING ON THE FULL TRAINING DATA =========="
python -m claimcheck_classical.training.tune --full-data

echo
echo "========== 3. INSERTED ERROR SET v2: TRAIN + VALIDATION =========="
python -m claimcheck_classical.evaluation.inserted_validation

echo
echo "Done. Results are in outputs/ and the model is in models/."
echo "The thresholds and the run record are in models/classical_model_full_best.json."
echo "Inserted Error Set results are in outputs/inserted_error_set/."

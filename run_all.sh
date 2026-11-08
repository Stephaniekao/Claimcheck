#!/usr/bin/env bash
# One command for the whole rule-based baseline on train + validation.
#
#   ./run_all.sh            tests, smoke test, then RAGBench + support score if the data is there
#   ./run_all.sh --quick    tests + smoke test only (seconds, no downloads)
#   ./run_all.sh --fetch    download RAGBench first (checks SHA256 against the team manifest)
#
# It never runs --final-test: test splits are used once, by hand (Experiment Card, Field 5).
# It adds no tuning trials: RAGBench uses the saved config, the support score the saved weights.
# Logged test results are kept; train/validation outputs in Baseline/outputs/ are rewritten.
set -euo pipefail
cd "$(dirname "$0")"

QUICK=0; FETCH=0
for arg in "$@"; do
  case "$arg" in
    --quick) QUICK=1 ;;
    --fetch) FETCH=1 ;;
    -h|--help) sed -n '2,11p' "$0"; exit 0 ;;
    *) echo "unknown option: $arg (try --help)"; exit 2 ;;
  esac
done

PY="${PYTHON:-python3}"
step() { printf '\n==== %s ====\n' "$1"; }

step "Environment"
want="$(cat .python-version 2>/dev/null || echo '?')"
have="$("$PY" -c 'import platform; print(platform.python_version())')"
echo "Python $have (pinned: $want)"
[ "$have" = "$want" ] || echo "WARNING: Python version differs from .python-version; results may not match."
"$PY" -m pip check >/dev/null 2>&1 || echo "WARNING: pip check reports broken requirements."

step "Unit tests"
"$PY" -m unittest discover Baseline/tests

step "Smoke test: Inserted Error Set, train + validation"
"$PY" Baseline/run_inserted_error_set.py

if [ "$QUICK" = 1 ]; then
  step "Done (--quick)"; exit 0
fi

if [ "$FETCH" = 1 ]; then
  step "Download RAGBench"
  "$PY" Baseline/fetch_ragbench.py
fi

if ls Data/RAGBench/FinQA/validation.parquet Data/RAGBench/TATQA/validation.parquet >/dev/null 2>&1; then
  step "RAGBench validation (saved config, no new trials)"
  "$PY" Baseline/run_ragbench.py --no-search
  step "Support score, train + validation (saved weights)"
  "$PY" Baseline/run_support_score.py
else
  step "RAGBench skipped"
  echo "Data/RAGBench/ not found. Run ./run_all.sh --fetch (or the data notebook) to include it."
fi

if [ -f Baseline/outputs/inserted_predictions_test.csv ] && \
   [ -f Baseline/outputs/ragbench_finqa_test_claims.csv ]; then
  step "Error analysis (from the saved test outputs)"
  "$PY" Baseline/error_analysis.py >/dev/null && echo "Wrote Baseline/outputs/error_analysis/"
else
  step "Error analysis skipped (no saved test outputs yet)"
fi

step "Done"
echo "Results: Baseline/outputs/  (each JSON records commit, data version and hardware)"

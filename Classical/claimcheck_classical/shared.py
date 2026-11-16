"""
ClaimCheck Classical Model - Shared Team Components

Experiment Card, Field 12: all four methods share the same inputs,
including the number normalizer and the cited passages. This module
imports them from the Baseline package instead of keeping a copy,
so a change there reaches every method and is recorded with each
result through the versions and checksums below.

Shared components:
1. Number normalizer       Baseline/claimcheck_baseline/normalize.py
2. Inserted Error Set      Baseline/claimcheck_baseline/evidence.py
   passages
3. Precision at recall     Baseline/claimcheck_baseline/metrics.py
   (comparison table column, Field 6)
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .config import PROJECT_ROOT

BASELINE_DIR = PROJECT_ROOT / "Baseline"

if not (BASELINE_DIR / "claimcheck_baseline").is_dir():
    raise ImportError(
        f"Shared components not found in {BASELINE_DIR}. "
        "The Classical model needs the team's Baseline package."
    )

if str(BASELINE_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINE_DIR))

from claimcheck_baseline import evidence as shared_evidence  # noqa: E402
from claimcheck_baseline import metrics as shared_metrics  # noqa: E402
from claimcheck_baseline import normalize  # noqa: E402

NORMALIZER_VERSION = normalize.NORMALIZER_VERSION
EVIDENCE_VERSION = shared_evidence.EVIDENCE_VERSION

# Exact relative tolerance, the RAGBench setting the baseline chose
# on validation (rel_tol=0). Rounding is still allowed through the
# normalizer's half_ulp rule.
REL_TOL = 0.0


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def shared_versions() -> dict:
    """Versions and checksums of the shared components."""

    package = BASELINE_DIR / "claimcheck_baseline"

    return {
        "normalizer": NORMALIZER_VERSION,
        "normalizer_sha256": _sha256(package / "normalize.py"),
        "evidence": EVIDENCE_VERSION,
        "evidence_sha256": _sha256(package / "evidence.py"),
        "metrics_sha256": _sha256(package / "metrics.py"),
    }


# ============================================================
# 1. NUMBER NORMALIZER
# ============================================================

def amounts_match(
    claim_amount,
    claim_text: str,
    passage_amount,
    passage_text: str,
) -> bool:
    """Same value, unit, scale and sign, as the baseline checks it."""

    return (
        normalize.quantities_match(claim_amount, passage_amount, REL_TOL)
        and not normalize.signs_conflict(
            claim_amount, claim_text, passage_amount, passage_text
        )
    )


def normalizer_cannot_read(claim: str) -> bool:
    """True when the claim has digits but the normalizer reads nothing.

    Experiment Card, Field 12: claims the normalizer cannot read are
    labeled Uncertain instead of being guessed, and the failure rate
    is tracked. "Cannot read" means the claim contains a digit, but
    the normalizer finds neither an amount nor a period in it.

    A claim with no digits at all is not a failure: it has no number
    to read, and the model still scores it from its text. This is
    narrower than the baseline's "no_amount" rule, which also makes
    claims without numbers Uncertain; see the README.
    """

    text = str(claim)

    if not any(character.isdigit() for character in text):
        return False

    if normalize.extract_quantities(text):
        return False

    return not normalize.extract_periods(text)


def normalizer_failure_mask(claims) -> np.ndarray:
    """Vectorized normalizer_cannot_read() over a list of claims."""

    return np.array(
        [normalizer_cannot_read(claim) for claim in claims],
        dtype=bool,
    )


# ============================================================
# 2. INSERTED ERROR SET PASSAGES
# ============================================================

def load_inserted_error_set(split: str | None = None) -> pd.DataFrame:
    """Claims with the team's shared passages (evidence.py).

    Model inputs are only claim and evidence. label and error_type
    are for evaluation.
    """

    data = shared_evidence.load_inserted_error_set()

    if split is not None:
        data = data[data["split"] == split].reset_index(drop=True)

    return data


def inserted_input_hashes() -> dict:
    return shared_evidence.input_hashes()


# ============================================================
# 3. PRECISION AT RECALL (FIELD 6)
# ============================================================

def precision_at_recall(y_true, p_unsupported, target: float = 0.80) -> dict:
    """Same definition as the baseline row of the comparison table."""

    gold = pd.Series(np.where(np.asarray(y_true) == 1, "Unsupported", "Supported"))
    risk = pd.Series(np.asarray(p_unsupported, dtype=float))

    return shared_metrics.precision_at_recall(gold, risk, target)

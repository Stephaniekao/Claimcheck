
"""
ClaimCheck Classical Model - Financial Scope Features

Financial scope feature extraction for:
1. Scope keyword detection
2. Total versus segment classification
3. Claim-evidence scope consistency
"""

import re
import math
import numpy as np
from .numerical_features import extract_number_details

# ============================================================
# 1. FINANCIAL SCOPE KEYWORDS
# ============================================================

SCOPE_KEYWORDS = {
    "total": [
        r"\btotal\b",
        r"\bconsolidated\b",
        r"\bcompany[\s-]?wide\b",
    ],
    "segment": [
        r"\bsegment\b",
        r"\bdivision\b",
        r"\bbusiness unit\b",
    ],
}



# ============================================================
# 2. FINANCIAL SCOPE EXTRACTION
# ============================================================

def extract_financial_scope(text: str) -> str:
    """Classify financial scope as total, segment, or unknown."""

    text = text.lower()

    detected_scopes = set()

    for scope, patterns in SCOPE_KEYWORDS.items():
        for pattern in patterns:
            if re.search(pattern, text):
                detected_scopes.add(scope)
                break

    # Return unknown when no explicit scope is detected.
    if not detected_scopes:
        return "unknown"

    # Mixed scopes cannot be represented by one simple label.
    if len(detected_scopes) > 1:
        return "mixed"

    return detected_scopes.pop()


# ============================================================
# 3. FINANCIAL SCOPE CONSISTENCY
# ============================================================

def compute_scope_consistency(
    claim: str,
    evidence: str,
) -> float:
    """Check financial scope consistency using number-scope pairs."""

    # Extract financial scopes.
    claim_scope = extract_financial_scope(claim)
    evidence_scope = extract_financial_scope(evidence)

    # Extract number-scope associations.
    claim_pairs = extract_number_scope_pairs(claim)
    evidence_pairs = extract_number_scope_pairs(evidence)

    # Compare explicit number-scope pairs when available.
    explicit_claim_pairs = [
        item for item in claim_pairs
        if item["scope"] in {"total", "segment"}
    ]

    if explicit_claim_pairs:
        used_evidence = set()

        for claim_item in explicit_claim_pairs:
            matched = False

            for index, evidence_item in enumerate(evidence_pairs):
                if index in used_evidence:
                    continue

                # Both numerical value and scope must match.
                same_value = math.isclose(
                    claim_item["value"],
                    evidence_item["value"],
                    rel_tol=1e-9,
                    abs_tol=1e-9,
                )

                same_scope = (
                    claim_item["scope"] == evidence_item["scope"]
                )

                if same_value and same_scope:
                    used_evidence.add(index)
                    matched = True
                    break

            if not matched:
                return 0.0

        return 1.0

    # Fall back to scope-only comparison when no explicit
    # claim number-scope pairs are available.
    if claim_scope in {"unknown", "mixed"}:
        return np.nan

    if evidence_scope in {"unknown", "mixed"}:
        return np.nan

    return float(claim_scope == evidence_scope)



# ============================================================
# 4. NUMBER-SCOPE ASSOCIATION
# ============================================================

def extract_number_scope_pairs(text: str) -> list[dict]:
    """Associate financial numbers with their clause-level scopes."""

    numbers = extract_number_details(text)
    results = []

    # Split clauses at semicolons or commas that are not
    # between digits. Preserve commas inside numbers.
    clause_pattern = r"[^,;]+(?:,(?=\d{3}(?:\D|$))[^,;]*)*"

    # A simpler and safer approach is to identify separators
    # first, then slice the original text into clauses.
    separators = [
        match.span()
        for match in re.finditer(
            r";|(?<!\d),|,(?!\d{3}(?:\D|$))",
            text,
        )
    ]

    clauses = []
    start = 0

    for separator_start, separator_end in separators:
        clauses.append((start, separator_start))
        start = separator_end

    clauses.append((start, len(text)))

    # Match each number to the scope of its containing clause.
    for number in numbers:
        number_start, number_end = number["span"]
        number_scope = "unknown"

        for clause_start, clause_end in clauses:
            if (
                clause_start <= number_start
                and number_end <= clause_end
            ):
                clause_text = text[clause_start:clause_end]
                number_scope = extract_financial_scope(clause_text)
                break

        results.append({
            "value": number["value"],
            "type": number["type"],
            "scope": number_scope,
        })

    return results

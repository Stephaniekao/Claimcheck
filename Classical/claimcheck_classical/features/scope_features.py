
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
from ..preprocessing import normalize_text

# ============================================================
# 1. FINANCIAL SCOPE KEYWORDS
# ============================================================

SCOPE_KEYWORDS = {
    "total": [
        r"\btotal\b",
        r"\bconsolidated\b",
        r"\bcompany[\s-]?wide\b",
        # Found while developing the scope feature on the Inserted
        # Error Set v2 training split (Experiment Card, Field 4):
        # Storyboard B claims say "the entire company".
        r"\bentire\b",
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
    """Check financial scope consistency using number-scope pairs.

    The evidence should be the original passage with one sentence
    per line. Each line is normalized and scored on its own, so a
    number takes the scope of its own line, not of the whole passage.
    A passage without line breaks is treated as one line, which is
    the old behavior.
    """

    from ..shared import amounts_match

    # Normalize each evidence line separately.
    lines = [
        normalize_text(line)
        for line in str(evidence).split("\n")
    ]
    lines = [line for line in lines if line]

    # Extract financial scopes.
    claim_scope = extract_financial_scope(claim)
    evidence_scope = extract_financial_scope(" ".join(lines))

    # Extract number-scope associations, line by line.
    claim_pairs = extract_number_scope_pairs(claim)
    evidence_pairs = [
        pair
        for line in lines
        for pair in extract_number_scope_pairs(line)
    ]

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

                # Both the amount (shared normalizer) and scope must match.
                same_value = amounts_match(
                    claim_item["quantity"],
                    claim_item["text"],
                    evidence_item["quantity"],
                    evidence_item["text"],
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
    """Associate each amount with the scope of its clause.

    Amounts come from the team's shared normalizer, so years, dates
    and identifiers are not amounts, and "$4.1 billion", "4,100
    million" and "4.1B" are the same value.
    """

    from ..shared import normalize

    # Clauses end at semicolons and at commas that are not inside
    # a number such as 4,100.
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

    results = []

    for amount in normalize.extract_quantities(text):
        scope = "unknown"

        for clause_start, clause_end in clauses:
            if clause_start <= amount.start < clause_end:
                scope = extract_financial_scope(text[clause_start:clause_end])
                break

        results.append({
            "value": amount.value,
            "type": amount.unit,
            "scope": scope,
            "quantity": amount,
            "text": text,
        })

    return results

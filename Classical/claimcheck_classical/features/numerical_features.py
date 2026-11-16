"""
ClaimCheck Classical Model - Numerical Features

Numerical feature extraction for:
1. Financial number extraction
2. Unit normalization
3. Claim-evidence numerical consistency
"""

import re
import math
import numpy as np

from .period_features import extract_reporting_periods


# ============================================================
# 1. FINANCIAL NUMBER EXTRACTION
# ============================================================

UNIT_MULTIPLIERS = {
    "thousand": 1_000,
    "million": 1_000_000,
    "billion": 1_000_000_000,
    "trillion": 1_000_000_000_000,
    "k": 1_000,
    "m": 1_000_000,
    "mm": 1_000_000,
    "mn": 1_000_000,
    "b": 1_000_000_000,
    "bn": 1_000_000_000,
}

# Abbreviations must be attached to the number, such as 4.1B or $300K.
# They are accepted only when the number has a dollar sign, a decimal
# point, or a comma. This keeps names such as "3M" from becoming numbers.
ABBREVIATED_UNITS = {"k", "m", "mm", "mn", "b", "bn"}


def extract_numbers(text: str) -> list[float]:
    """Extract financial numbers and normalize scale units."""

    # Match numbers with optional currency, sign, scale, and parentheses.
    pattern = (
        r"(?<![\w.])"
        r"(\(?\s*(?:[-+]?\$?|\$[-+]?)\d[\d,]*(?:\.\d+)?"
        r"(?:\s*(?:thousand|million|billion|trillion)|(?:bn|mm|mn|k|m|b))?"
        r"\s*\)?)"
        r"(?!\w)"
    )

    matches = re.findall(pattern, text.lower())
    numbers = []

    for match in matches:
        value_text = match.strip()

        # Parentheses indicate a negative financial value.
        is_parenthesized = (
            value_text.startswith("(")
            and value_text.endswith(")")
        )

        # Remove parentheses, currency symbols, and commas.
        cleaned = (
            value_text.replace("(", "")
            .replace(")", "")
            .replace("$", "")
            .replace(",", "")
            .strip()
        )

        # Extract the numerical value and optional scale unit.
        # This supports both "199 million" and "199million".
        number_pattern = (
            r"^([-+]?\d+(?:\.\d+)?)"
            r"\s*(thousand|million|billion|trillion|bn|mm|mn|k|m|b)?$"
        )

        number_match = re.fullmatch(
            number_pattern,
            cleaned,
        )

        if number_match is None:
            continue

        number = float(number_match.group(1))
        unit = number_match.group(2) or ""

        # Skip abbreviations that do not look like financial amounts.
        if unit in ABBREVIATED_UNITS:
            looks_financial = (
                "$" in value_text
                or "." in number_match.group(1)
                or "," in value_text
            )

            if not looks_financial:
                continue

        # Apply the financial scale multiplier.
        number *= UNIT_MULTIPLIERS.get(unit, 1)

        # Convert accounting parentheses to a negative value.
        if is_parenthesized:
            number = -abs(number)

        numbers.append(number)

    return numbers


# ============================================================
# 2. NUMERICAL CONSISTENCY
# ============================================================

def compute_numerical_consistency(
    claim: str,
    evidence: str,
) -> float:
    """Check numerical consistency while excluding reporting years."""

    def financial_numbers(text: str) -> list[dict]:
        """Extract numbers excluding identified reporting years."""

        numbers = extract_number_details(text)
        periods = extract_reporting_periods(text)

        # Collect reporting years identified by the period parser.
        reporting_years = {
            year
            for year, quarter in periods
            if year is not None
        }

        filtered = []

        for number in numbers:
            # Exclude plain numbers that match reporting years.
            if (
                number["type"] == "number"
                and number["value"] in reporting_years
            ):
                continue

            filtered.append(number)

        return filtered

    # Extract financial numbers from claim and evidence.
    claim_numbers = financial_numbers(claim)
    evidence_numbers = financial_numbers(evidence)

    # No financial numbers in the claim means unknown.
    if not claim_numbers:
        return np.nan

    # Track evidence numbers already used for matching.
    used_evidence = set()

    for claim_number in claim_numbers:
        matched = False

        for index, evidence_number in enumerate(evidence_numbers):

            if index in used_evidence:
                continue

            # Percentages must match percentages.
            claim_is_percentage = (
                claim_number["type"] == "percentage"
            )
            evidence_is_percentage = (
                evidence_number["type"] == "percentage"
            )

            if claim_is_percentage != evidence_is_percentage:
                continue

            # Compare normalized numerical values.
            if math.isclose(
                claim_number["value"],
                evidence_number["value"],
                rel_tol=1e-9,
                abs_tol=1e-9,
            ):
                used_evidence.add(index)
                matched = True
                break

        if not matched:
            return 0.0

    return 1.0


# ============================================================
# 3. NUMBER DETAILS WITH UNIT TYPES AND TEXT POSITIONS
# ============================================================

def extract_number_details(text: str) -> list[dict]:
    """Extract normalized numbers, unit types, and text positions."""

    # Match numbers with optional currency, sign, scale, and percentage.
    pattern = (
        r"(?<![\w.])"
        r"(\(?\s*(?:[-+]?\$?|\$[-+]?)\d[\d,]*(?:\.\d+)?"
        r"(?:\s*(?:thousand|million|billion|trillion)|(?:bn|mm|mn|k|m|b))?"
        r"\s*(?:%|percent)?\s*\)?)"
        r"(?!\w)"
    )

    # Use finditer to preserve each match's position in the text.
    matches = re.finditer(pattern, text.lower())
    results = []

    for match in matches:
        raw = match.group(1).strip()

        # Identify the explicit unit type.
        if "%" in raw or "percent" in raw:
            unit_type = "percentage"
        elif "$" in raw:
            unit_type = "currency"
        else:
            unit_type = "number"

        # Remove percentage markers before number normalization.
        cleaned = (
            raw.replace("%", "")
            .replace("percent", "")
            .strip()
        )

        # Reuse financial number normalization.
        values = extract_numbers(cleaned)

        if values:
            results.append({
                "value": values[0],
                "type": unit_type,
                "span": match.span(1),
            })

    return results

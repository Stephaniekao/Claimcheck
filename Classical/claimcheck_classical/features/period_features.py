
"""
ClaimCheck Classical Model - Period Features

Reporting period feature extraction for:
1. Reporting year extraction
2. Quarter extraction
3. Claim-evidence period consistency
"""

import re
import numpy as np

# ============================================================
# 1. REPORTING YEAR EXTRACTION
# ============================================================

def extract_reporting_years(text: str) -> list[int]:
    """Extract reporting years from financial text."""

    # Match standalone years and FY-prefixed years.
    # Exclude years directly preceded by currency symbols.
    pattern = (
        r"(?<![\w$])"
        r"(?:fy\s*)?"
        r"((?:19|20)\d{2})"
        r"(?!\w|\.\d)"
    )

    matches = re.findall(pattern, text.lower())

    return [int(year) for year in matches]



# ============================================================
# 2. REPORTING QUARTER EXTRACTION
# ============================================================

def extract_reporting_quarters(text: str) -> list[int]:
    """Extract reporting quarters from financial text."""

    text = text.lower()

    # Match Q1, Q2, Q3, Q4 and Q 1, Q 2, etc.
    q_pattern = r"\bq\s*([1-4])\b"

    # Match first quarter, second quarter, 3rd quarter, etc.
    word_pattern = (
        r"\b(first|second|third|fourth|"
        r"1st|2nd|3rd|4th)\s+quarter\b"
    )

    quarter_map = {
        "first": 1,
        "second": 2,
        "third": 3,
        "fourth": 4,
        "1st": 1,
        "2nd": 2,
        "3rd": 3,
        "4th": 4,
    }

    # Preserve the order of quarters in the original text.
    matches = []

    for match in re.finditer(q_pattern, text):
        matches.append((match.start(), int(match.group(1))))

    for match in re.finditer(word_pattern, text):
        quarter = quarter_map[match.group(1)]
        matches.append((match.start(), quarter))

    matches.sort(key=lambda item: item[0])

    return [quarter for _, quarter in matches]


# ============================================================
# 3. PERIOD CONSISTENCY
# ============================================================

def compute_period_consistency(
    claim: str,
    evidence: str,
) -> float:
    """Check claim-evidence reporting period consistency."""

    # Extract complete reporting periods.
    claim_periods = extract_reporting_periods(claim)
    evidence_periods = extract_reporting_periods(evidence)

    # No reporting period in the claim means unknown.
    if not claim_periods:
        return np.nan

    # Every claim period must match a compatible evidence period.
    for claim_year, claim_quarter in claim_periods:
        matched = False

        for evidence_year, evidence_quarter in evidence_periods:

            # Match years when the claim specifies a year.
            if claim_year is not None:
                if claim_year != evidence_year:
                    continue

            # Match quarters when the claim specifies a quarter.
            if claim_quarter is not None:
                if claim_quarter != evidence_quarter:
                    continue

            matched = True
            break

        if not matched:
            return 0.0

    return 1.0


# ============================================================
# 4. REPORTING PERIOD PAIRING
# ============================================================

def extract_reporting_periods(
    text: str,
) -> list[tuple[int | None, int | None]]:
    """Extract reporting periods as year-quarter pairs."""

    text = text.lower()

    quarter_words = {
        "first": 1,
        "second": 2,
        "third": 3,
        "fourth": 4,
        "1st": 1,
        "2nd": 2,
        "3rd": 3,
        "4th": 4,
    }

    quarter_token = (
        r"(?:q\s*[1-4]|first|second|third|fourth|"
        r"1st|2nd|3rd|4th)"
    )

    year_token = r"(?:fy\s*)?(?:19|20)\d{2}"

    # Match Q1 2024, first quarter of 2024, and similar formats.
    quarter_first = (
        rf"\b({quarter_token})"
        rf"(?:\s+quarter)?\s+"
        rf"(?:of\s+)?({year_token})\b"
    )

    # Match 2024 Q1, FY2024 Q2, and 2024 first quarter.
    year_first = (
        rf"(?<![\w$])({year_token})\s+"
        rf"({quarter_token})(?:\s+quarter)?\b"
    )

    matches = []
    occupied = []

    def parse_quarter(value: str) -> int:
        """Convert a quarter expression to an integer."""
        value = value.strip()
        if value.startswith("q"):
            return int(value.replace("q", "").strip())
        return quarter_words[value]

    def parse_year(value: str) -> int:
        """Convert a year expression to an integer."""
        return int(re.search(r"(?:19|20)\d{2}", value).group())

    # Extract complete quarter-year pairs.
    for pattern, year_index, quarter_index in [
        (quarter_first, 2, 1),
        (year_first, 1, 2),
    ]:
        for match in re.finditer(pattern, text):
            if any(
                start < match.end() and match.start() < end
                for start, end in occupied
            ):
                continue

            year = parse_year(match.group(year_index))
            quarter = parse_quarter(match.group(quarter_index))

            matches.append((match.start(), (year, quarter)))
            occupied.append(match.span())

    # Extract remaining standalone years.
    year_pattern = (
        r"(?<![\w$])(?:fy\s*)?"
        r"((?:19|20)\d{2})(?!\w|\.\d)"
    )

    for match in re.finditer(year_pattern, text):
        if any(
            start < match.end() and match.start() < end
            for start, end in occupied
        ):
            continue

        matches.append((match.start(), (int(match.group(1)), None)))
        occupied.append(match.span())

    # Extract remaining standalone quarters.
    quarter_pattern = (
        r"\b(q\s*[1-4]|first|second|third|fourth|"
        r"1st|2nd|3rd|4th)"
        r"(?:\s+quarter)?\b"
    )

    for match in re.finditer(quarter_pattern, text):
        if any(
            start < match.end() and match.start() < end
            for start, end in occupied
        ):
            continue

        # Do not treat ordinary words like "first" as quarters
        # unless they are followed by "quarter".
        token = match.group(1)
        if not token.startswith("q") and "quarter" not in match.group():
            continue

        matches.append((match.start(), (None, parse_quarter(token))))
        occupied.append(match.span())

    matches.sort(key=lambda item: item[0])

    return [period for _, period in matches]

"""
ClaimCheck Classical Model - Text Preprocessing

Text preprocessing for:
1. Text normalization

Financial numbers, dates, units, and symbols are preserved.
The original claim and evidence are not modified.
"""

import re
import pandas as pd


# ============================================================
# 1. TEXT NORMALIZATION
# ============================================================

def normalize_text(text: str) -> str:
    """Normalize text casing and whitespace."""

    # Convert text to lowercase.
    text = text.lower()

    # Replace consecutive whitespace with a single space.
    text = re.sub(r"\s+", " ", text)

    # Remove leading and trailing whitespace.
    text = text.strip()

    return text


# ============================================================
# 2. DATAFRAME PREPROCESSING
# ============================================================

def preprocess_claims(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize claim and evidence text without changing originals."""

    # Check required text columns.
    required = {"claim", "evidence"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing preprocessing columns: {sorted(missing)}"
        )

    # Create a copy to preserve the input DataFrame.
    result = df.copy()

    # Normalize claim and evidence into separate columns.
    result["claim_clean"] = result["claim"].apply(normalize_text)
    result["evidence_clean"] = result["evidence"].apply(normalize_text)

    return result

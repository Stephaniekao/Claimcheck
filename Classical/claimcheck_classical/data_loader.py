"""
ClaimCheck Classical Model - Data Loading

Data loading and preparation for:
1. Dataset configuration
2. Input validation and CSV reading
3. RAGBench answer-level loading
4. RAGBench claim-level expansion
5. RAGBench claim-level loading
6. Inserted Error Set loading

All original datasets and predefined splits are preserved.
No data resampling or train/validation/test splitting is performed.
"""

import json
from pathlib import Path

import pandas as pd

from .config import DATA_DIR


# ============================================================
# 1. DATASET CONFIGURATION
# ============================================================

# Supported RAGBench financial subsets
RAGBENCH_SUBSETS = ("FinQA", "TATQA")

# Existing dataset splits
VALID_SPLITS = ("train", "validation", "test")

# RAGBench columns stored as JSON strings in the CSV files
RAGBENCH_JSON_COLUMNS = (
    "response_sentences",
    "documents_sentences",
    "unsupported_response_sentence_keys",
)


# ============================================================
# 2. INPUT VALIDATION AND CSV READING
# ============================================================

def _validate_split(split: str) -> None:
    """Check whether the requested split is valid."""

    if split not in VALID_SPLITS:
        raise ValueError(
            f"Invalid split: {split}. Expected one of {VALID_SPLITS}"
        )


def _read_csv(path: Path) -> pd.DataFrame:
    """Read an existing CSV without modifying the source file."""

    # Stop if the expected dataset file does not exist.
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")

    return pd.read_csv(path)


# ============================================================
# 3. RAGBENCH ANSWER-LEVEL LOADING
# ============================================================

def load_ragbench(subset: str, split: str) -> pd.DataFrame:
    """Load RAGBench answer-level data and parse JSON columns."""

    # Validate the requested RAGBench subset.
    if subset not in RAGBENCH_SUBSETS:
        raise ValueError(
            f"Invalid subset: {subset}. "
            f"Expected one of {RAGBENCH_SUBSETS}"
        )

    # Validate the existing dataset split.
    _validate_split(split)

    # Locate the original RAGBench CSV file.
    path = DATA_DIR / "RAGBench" / subset / f"{split}.csv"
    df = _read_csv(path)

    # Check columns required for claim-level expansion.
    required_columns = {
        "id",
        "response_sentences",
        "documents_sentences",
        "unsupported_response_sentence_keys",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing RAGBench columns: {sorted(missing)}"
        )

    # Convert JSON-encoded strings into Python lists.
    for column in RAGBENCH_JSON_COLUMNS:
        try:
            df[column] = df[column].apply(json.loads)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid JSON in {path}, column {column}"
            ) from exc

    # Preserve the original CSV row position for identification.
    df["answer_row"] = range(len(df))

    # Record the dataset subset and original split.
    df["subset"] = subset
    df["split"] = split

    return df


# ============================================================
# 4. RAGBENCH CLAIM-LEVEL EXPANSION
# ============================================================

def expand_ragbench_claims(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Convert answer-level records into sentence-level claims."""

    rows = []

    # Process each original answer-level record.
    for record in df.itertuples(index=False):

        # Flatten all evidence sentences in their original order.
        evidence_sentences = [
            str(sentence[1])
            for document in record.documents_sentences
            for sentence in document
        ]

        # Combine evidence sentences into one text passage.
        evidence = "\n".join(evidence_sentences)

        # Identify sentence keys labeled as unsupported.
        unsupported_keys = set(
            record.unsupported_response_sentence_keys
        )

        # Create one claim-level row for each response sentence.
        for sentence_key, claim in record.response_sentences:

            # Assign the original RAGBench binary label.
            label = (
                "Unsupported"
                if sentence_key in unsupported_keys
                else "Supported"
            )

            rows.append(
                {
                    "source": "RAGBench",
                    "subset": record.subset,
                    "split": record.split,
                    "answer_row": record.answer_row,
                    "answer_id": record.id,
                    "sentence_key": sentence_key,
                    "claim": claim,
                    "evidence": evidence,
                    "label": label,
                }
            )

    # Return a standardized claim-level DataFrame.
    return pd.DataFrame(
        rows,
        columns=[
            "source",
            "subset",
            "split",
            "answer_row",
            "answer_id",
            "sentence_key",
            "claim",
            "evidence",
            "label",
        ],
    )


# ============================================================
# 5. RAGBENCH CLAIM-LEVEL LOADING
# ============================================================

def load_ragbench_claims(
    subset: str,
    split: str,
) -> pd.DataFrame:
    """Load RAGBench and return sentence-level claim examples."""

    # Load the original answer-level dataset.
    answers = load_ragbench(subset, split)

    # Expand answers into individual claim-level records.
    return expand_ragbench_claims(answers)


# ============================================================
# 6. INSERTED ERROR SET LOADING
# ============================================================

def load_inserted_errors(
    split: str | None = None,
) -> pd.DataFrame:
    """Load existing General and Storyboard B claims without resplitting."""

    # Validate the requested split when provided.
    if split is not None:
        _validate_split(split)

    # Locate the existing Inserted Error Set directory.
    folder = DATA_DIR / "Inserted Error Set"

    # Load one requested split or all three existing splits.
    splits = (split,) if split is not None else VALID_SPLITS

    frames = []

    # Load both Inserted Error Set subsets for each split.
    for current_split in splits:
        for subset, prefix in (
            ("General", "general_claims"),
            ("Storyboard B", "storyboard_b"),
        ):

            # Construct the existing split-specific CSV path.
            path = folder / f"{prefix}_{current_split}.csv"
            df = _read_csv(path)

            # Check required claim-level columns.
            required = {
                "claim_id",
                "claim",
                "label",
                "split",
            }

            missing = required - set(df.columns)

            if missing:
                raise ValueError(
                    f"Missing columns in {path.name}: {sorted(missing)}"
                )

            # Ensure the CSV matches its predefined split.
            if not df["split"].eq(current_split).all():
                raise ValueError(
                    f"Split mismatch in {path.name}"
                )

            # Validate the original binary claim labels.
            if not df["label"].isin(
                ("Supported", "Unsupported")
            ).all():
                raise ValueError(
                    f"Invalid labels in {path.name}"
                )

            # Identify the Inserted Error Set subset.
            df["subset"] = subset

            # Collect the loaded subset without resampling.
            frames.append(df)

    # Combine the existing subsets and preserve all original columns.
    return pd.concat(frames, ignore_index=True)


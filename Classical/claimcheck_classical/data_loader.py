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


def _read_adherence(parquet_path: Path, df: pd.DataFrame) -> pd.Series:
    """Read adherence_score from the parquet file, row by row.

    The rows must line up with the CSV, so the ids are compared.
    Returns missing values when the parquet file is not there.
    """

    if not parquet_path.is_file():
        return pd.Series([None] * len(df), index=df.index, dtype=object)

    parquet = pd.read_parquet(
        parquet_path,
        columns=["id", "adherence_score"],
    )

    if len(parquet) != len(df) or not (
        parquet["id"].astype(str).to_numpy() == df["id"].astype(str).to_numpy()
    ).all():
        raise ValueError(
            f"Rows of {parquet_path.name} do not line up with the CSV."
        )

    return pd.Series(
        parquet["adherence_score"].astype(bool).to_numpy(),
        index=df.index,
    )


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

    # The answer-level gold label (adherence_score) is not one of the
    # 12 CSV columns, so read it from the parquet file saved next to
    # the CSV by the same download step. It is a gold label only,
    # never a model input.
    if "adherence_score" not in df.columns:
        df["adherence_score"] = _read_adherence(path.with_suffix(".parquet"), df)

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

            # Answer-level gold label, used for answer AUROC only.
            # It is never a model input.
            adherence = getattr(record, "adherence_score", None)

            rows.append(
                {
                    "adherence_score": adherence,
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
            "adherence_score",
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
    """Load Inserted Error Set v2 claims with the team's shared passages.

    Experiment Card, Field 12: every method reads the same cited
    passages, built by Baseline/claimcheck_baseline/evidence.py from
    the filing only (never from the claim or label). No resplitting.

    Columns: claim_id, subset ("general" or "storyboard_b"), split,
    claim, evidence (model inputs), label and error_type (evaluation
    only), and ticker.
    """

    from .shared import load_inserted_error_set

    if split is not None:
        _validate_split(split)

    df = load_inserted_error_set(split)

    if not df["label"].isin(("Supported", "Unsupported")).all():
        raise ValueError("Invalid labels in the Inserted Error Set.")

    return df

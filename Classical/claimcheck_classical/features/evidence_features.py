"""
ClaimCheck Classical Model - Evidence-Aware Features

Evidence-aware feature extraction for:
1. Sentence-level TF-IDF cosine similarity
2. Claim-evidence token coverage
3. Rounding- and scale-tolerant number matching
4. Feature matrix construction
5. Answer-level score aggregation
6. Tuning trial sampling

Input dataframes must come from preprocess_claims() and contain
claim_clean, evidence_clean, and the original evidence column.
The original evidence stores one sentence per line.
"""

from __future__ import annotations

import itertools
import random
import re
from functools import lru_cache

import numpy as np
import pandas as pd

from ..preprocessing import normalize_text
from .numerical_features import (
    compute_numerical_consistency,
    extract_number_details,
)
from .period_features import compute_period_consistency
from .scope_features import compute_scope_consistency
from .text_features import compute_cosine_similarity


# ============================================================
# 1. FEATURE COLUMNS AND CONSTANTS
# ============================================================

# Original seven features from feature_builder.py.
BASE_COLUMNS = [
    "cosine_similarity",
    "numerical_consistency",
    "period_consistency",
    "scope_consistency",
    "numerical_missing",
    "period_missing",
    "scope_missing",
]

# Additional evidence-aware features.
EXTRA_FEATURE_COLUMNS = [
    "max_sentence_cosine",
    "top3_sentence_cosine",
    "claim_token_coverage",
    "fuzzy_number_match",
    "claim_number_count",
    "claim_length",
]

# Complete feature order used by training and evaluation.
FULL_FEATURE_COLUMNS = BASE_COLUMNS + EXTRA_FEATURE_COLUMNS

# Normalized claim text used as the text input of the model.
TEXT_COLUMN = "claim_clean"

# Relative tolerance for fuzzy number matching (1 percent).
NUMBER_RTOL = 0.01

# Scale multipliers applied to evidence values:
# units, thousands, millions, and billions.
SCALES_UP = (1.0, 1e3, 1e6, 1e9)

# Alphabetic word tokens with at least three characters.
TOKEN_PATTERN = re.compile(r"[a-z][a-z\-]{2,}")


# ============================================================
# 2. SENTENCE-LEVEL COSINE SIMILARITY
# ============================================================

def split_evidence(evidence: str) -> list[str]:
    """Split original evidence into normalized sentences."""

    # The original evidence stores one sentence per line.
    sentences = [
        normalize_text(sentence)
        for sentence in str(evidence).split("\n")
    ]

    # Remove empty sentences.
    return [sentence for sentence in sentences if sentence]


def sentence_cosine_features(
    vectorizer,
    claims: list[str],
    evidences: list[str],
    chunk_size: int = 1000,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute maximum and top-3 mean sentence cosine similarity."""

    n = len(claims)

    max_similarity = np.zeros(n)
    top3_similarity = np.zeros(n)

    # Process claims in chunks to limit memory use.
    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)

        # Transform the chunk of claims with the fitted vocabulary.
        claim_vectors = vectorizer.transform(claims[start:stop])

        # Split each evidence passage into sentences and remember
        # which claim each sentence belongs to.
        sentences = []
        owners = []

        for local_index, evidence in enumerate(evidences[start:stop]):
            parts = split_evidence(evidence)

            sentences.extend(parts)
            owners.extend([local_index] * len(parts))

        # Skip chunks without any evidence sentences.
        if not sentences:
            continue

        owners = np.asarray(owners)
        sentence_vectors = vectorizer.transform(sentences)

        # TF-IDF vectors are L2-normalized.
        # Their dot product equals cosine similarity.
        similarities = np.asarray(
            claim_vectors[owners]
            .multiply(sentence_vectors)
            .sum(axis=1)
        ).ravel()

        # Locate the block of sentences belonging to each claim.
        unique_owners, group_starts = np.unique(
            owners,
            return_index=True,
        )

        group_ends = list(group_starts[1:]) + [len(similarities)]

        # Keep the best and the top-3 average sentence similarity.
        for owner, group_start, group_end in zip(
            unique_owners,
            group_starts,
            group_ends,
        ):
            ranked = np.sort(
                similarities[group_start:group_end]
            )[::-1]

            max_similarity[start + owner] = ranked[0]
            top3_similarity[start + owner] = ranked[:3].mean()

    return max_similarity, top3_similarity


# ============================================================
# 3. TOKEN COVERAGE
# ============================================================

@lru_cache(maxsize=4096)
def _evidence_tokens(evidence: str) -> frozenset:
    """Extract unique word tokens from evidence (cached)."""

    return frozenset(
        TOKEN_PATTERN.findall(normalize_text(evidence))
    )


def token_coverage(claim_clean: str, evidence: str) -> float:
    """Compute the share of claim tokens found in the evidence."""

    # Extract unique word tokens from the claim.
    claim_tokens = set(TOKEN_PATTERN.findall(claim_clean))

    # A claim without word tokens has no coverage.
    if not claim_tokens:
        return 0.0

    # Count claim tokens that also appear in the evidence.
    shared = claim_tokens & _evidence_tokens(evidence)

    return len(shared) / len(claim_tokens)


# ============================================================
# 4. FUZZY NUMBER MATCHING
# ============================================================

def _financial_values(text: str) -> list[float]:
    """Extract absolute numeric values, excluding reporting years."""

    values = []

    for item in extract_number_details(text):
        value = abs(float(item["value"]))

        # Plain integers between 1900 and 2100 are treated as years.
        is_year = (
            item["type"] == "number"
            and value.is_integer()
            and 1900 <= value <= 2100
        )

        if not is_year:
            values.append(value)

    return values


@lru_cache(maxsize=4096)
def _evidence_values(evidence: str) -> np.ndarray:
    """Extract numeric values from evidence (cached)."""

    return np.asarray(
        _financial_values(normalize_text(evidence)),
        dtype=float,
    )


def fuzzy_number_match(
    claim_clean: str,
    evidence: str,
) -> tuple[float, int]:
    """Match claim numbers to evidence with rounding and scale tolerance.

    Returns:
        Fraction of claim numbers found in the evidence,
        and the number of claim numbers.

    The match fraction is 0.0 when the claim has no numbers.
    The returned count distinguishes that case.
    """

    # Extract nonzero claim numbers.
    claim_values = [
        value
        for value in _financial_values(claim_clean)
        if value != 0
    ]

    # No claim numbers means nothing to match.
    if not claim_values:
        return 0.0, 0

    evidence_values = _evidence_values(evidence)

    # No evidence numbers means no claim number can match.
    if evidence_values.size == 0:
        return 0.0, len(claim_values)

    matched = 0

    for value in claim_values:

        # Large claim values are also compared after rescaling the
        # evidence, because tables often omit units such as millions.
        scales = SCALES_UP if value >= 1000 else (1.0,)

        # A claim number matches when any evidence number
        # agrees within the relative tolerance at any scale.
        found = any(
            np.any(
                np.isclose(
                    evidence_values * scale,
                    value,
                    rtol=NUMBER_RTOL,
                    atol=0.0,
                )
            )
            for scale in scales
        )

        if found:
            matched += 1

    return matched / len(claim_values), len(claim_values)


# ============================================================
# 5. FEATURE MATRIX CONSTRUCTION
# ============================================================

def _check_columns(df: pd.DataFrame) -> None:
    """Check columns required for evidence-aware features."""

    for column in ("claim_clean", "evidence_clean", "evidence"):
        if column not in df.columns:
            raise ValueError(
                f"Missing required column: {column}. "
                "Pass the output of preprocess_claims()."
            )


def build_static_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build features that do not depend on the TF-IDF vectorizer.

    These features are identical for every TF-IDF configuration,
    so they can be computed once per split and reused across trials.
    """

    _check_columns(df)

    # Read normalized claims and both evidence versions.
    claims = df["claim_clean"].fillna("").astype(str).tolist()
    clean_evidence = df["evidence_clean"].fillna("").astype(str).tolist()
    raw_evidence = df["evidence"].fillna("").astype(str).tolist()

    # Calculate the three financial consistency scores.
    scores = {
        # Numbers read the original evidence, one line at a time,
        # with the team's shared normalizer.
        "numerical": [
            compute_numerical_consistency(claim, evidence)
            for claim, evidence in zip(claims, raw_evidence)
        ],
        "period": [
            compute_period_consistency(claim, evidence)
            for claim, evidence in zip(claims, clean_evidence)
        ],
        # Scope reads the original evidence, one line at a time.
        "scope": [
            compute_scope_consistency(claim, evidence)
            for claim, evidence in zip(claims, raw_evidence)
        ],
    }

    features = pd.DataFrame(index=df.index)

    # Add missing indicators before replacing NaN values.
    # This matches feature_builder.build_feature_matrix.
    for name, values in scores.items():
        series = pd.Series(
            values,
            index=df.index,
            dtype=float,
        )

        features[f"{name}_consistency"] = series.fillna(0.0)
        features[f"{name}_missing"] = series.isna().astype(int)

    # Share of claim tokens that appear in the evidence.
    features["claim_token_coverage"] = [
        token_coverage(claim, evidence)
        for claim, evidence in zip(claims, raw_evidence)
    ]

    # Rounding- and scale-tolerant number matching.
    fuzzy = [
        fuzzy_number_match(claim, evidence)
        for claim, evidence in zip(claims, raw_evidence)
    ]

    features["fuzzy_number_match"] = [
        match for match, _ in fuzzy
    ]

    # Log-scaled counts keep long claims from dominating.
    features["claim_number_count"] = np.log1p(
        [count for _, count in fuzzy]
    )

    features["claim_length"] = np.log1p(
        [len(claim.split()) for claim in claims]
    )

    return features


def build_tfidf_features(
    df: pd.DataFrame,
    vectorizer,
    chunk_size: int = 1000,
) -> pd.DataFrame:
    """Build features that depend on the fitted TF-IDF vectorizer."""

    _check_columns(df)

    # Read normalized claims and both evidence versions.
    claims = df["claim_clean"].fillna("").astype(str).tolist()
    clean_evidence = df["evidence_clean"].fillna("").astype(str).tolist()
    raw_evidence = df["evidence"].fillna("").astype(str).tolist()

    # Cosine similarity against the complete evidence passage.
    whole_similarity = compute_cosine_similarity(
        vectorizer,
        claims,
        clean_evidence,
    )

    # Cosine similarity against individual evidence sentences.
    max_similarity, top3_similarity = sentence_cosine_features(
        vectorizer,
        claims,
        raw_evidence,
        chunk_size=chunk_size,
    )

    return pd.DataFrame(
        {
            "cosine_similarity": np.asarray(whole_similarity).ravel(),
            "max_sentence_cosine": max_similarity,
            "top3_sentence_cosine": top3_similarity,
        },
        index=df.index,
    )


def build_full_feature_matrix(
    df: pd.DataFrame,
    vectorizer,
    static: pd.DataFrame | None = None,
    chunk_size: int = 1000,
) -> pd.DataFrame:
    """Build all thirteen features in a fixed column order.

    Pass static features from build_static_features() to reuse
    the slow vectorizer-independent part across tuning trials.
    """

    # Compute static features only when they are not provided.
    if static is None:
        static = build_static_features(df)

    # Compute features that depend on the fitted vectorizer.
    tfidf = build_tfidf_features(
        df,
        vectorizer,
        chunk_size=chunk_size,
    )

    features = pd.concat([tfidf, static], axis=1)

    # Keep the same feature order for all dataset splits.
    return features[FULL_FEATURE_COLUMNS]


def with_claim_text(
    features: pd.DataFrame,
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Attach normalized claim text as the model's text input."""

    if TEXT_COLUMN not in df.columns:
        raise ValueError(
            f"Missing required column: {TEXT_COLUMN}. "
            "Pass the output of preprocess_claims()."
        )

    # Copy so the numeric feature matrix is not modified in place.
    result = features.copy()

    # Align by position to avoid index mismatches.
    result[TEXT_COLUMN] = df[TEXT_COLUMN].fillna("").astype(str).to_numpy()

    return result


# ============================================================
# 6. ANSWER-LEVEL SCORE AGGREGATION
# ============================================================

ANSWER_AGGREGATIONS = ("min", "mean")


def aggregate_answer_scores(
    df: pd.DataFrame,
    p_supported,
    key_columns: tuple[str, ...] = ("subset", "answer_row"),
    method: str = "min",
) -> pd.DataFrame:
    """Combine claim scores into one answer score.

    method "min" uses the lowest claim support probability and
    "mean" uses the average. The Experiment Card (Field 6) says the
    rule is chosen on the validation split; see
    select_answer_aggregation().

    Gold: an answer is positive (1) when adherence_score is False,
    the same gold the baseline uses. When the column is missing, the
    old rule "any claim is Unsupported" is used. Both are returned,
    so the two can be compared.

    A higher answer_score means the answer is more likely to
    contain an Unsupported claim.
    """

    if method not in ANSWER_AGGREGATIONS:
        raise ValueError(
            f"method must be one of {ANSWER_AGGREGATIONS}."
        )

    columns = list(key_columns) + ["label"]
    has_adherence = "adherence_score" in df.columns

    if has_adherence:
        columns.append("adherence_score")

    table = df[columns].reset_index(drop=True).copy()

    table["p_supported"] = np.asarray(
        p_supported,
        dtype=float,
    )

    grouped = table.groupby(list(key_columns), sort=False)

    answers = grouped.agg(
        agg_p_supported=("p_supported", method),
        y_any_claim=(
            "label",
            lambda labels: int((labels == "Unsupported").any()),
        ),
    ).reset_index()

    if has_adherence:
        adherence = grouped["adherence_score"].first().reset_index(drop=True)

        if adherence.isna().any():
            raise ValueError("adherence_score is missing for some answers.")

        # The CSV may store booleans as text.
        adherence = adherence.map(
            lambda value: str(value).strip().lower() in ("true", "1", "1.0")
        )

        answers["y_adherence"] = (~adherence).astype(int).to_numpy()
        answers["y_true"] = answers["y_adherence"]
    else:
        answers["y_true"] = answers["y_any_claim"]

    answers["answer_score"] = 1.0 - answers["agg_p_supported"]

    # Old column name, kept so existing callers still work.
    if method == "min":
        answers["min_p_supported"] = answers["agg_p_supported"]

    return answers


def select_answer_aggregation(
    validation_df: pd.DataFrame,
    p_supported,
    key_columns: tuple[str, ...] = ("subset", "answer_row"),
) -> dict:
    """Choose the answer aggregation rule on validation data only.

    Returns the AUROC of each rule, the chosen rule, and how many
    answers have a different gold label under adherence_score and
    under "any claim is Unsupported".
    """

    from sklearn.metrics import roc_auc_score

    scores = {}
    disagreements = None

    for method in ANSWER_AGGREGATIONS:
        answers = aggregate_answer_scores(
            validation_df, p_supported, key_columns, method=method
        )
        scores[method] = float(
            roc_auc_score(answers["y_true"], answers["answer_score"])
        )

        if "y_adherence" in answers.columns:
            disagreements = int(
                (answers["y_adherence"] != answers["y_any_claim"]).sum()
            )

    # Ties go to "min", the rule named in the Experiment Card.
    chosen = max(ANSWER_AGGREGATIONS, key=lambda m: (scores[m], m == "min"))

    return {
        "chosen_method": chosen,
        "validation_answer_auroc": scores,
        "gold": "adherence_score" if disagreements is not None else "any_claim",
        "gold_disagreements": disagreements,
    }


# ============================================================
# 7. TUNING TRIAL SAMPLING
# ============================================================

def sample_trials(
    ngram_ranges,
    vocab_sizes,
    c_values,
    max_trials: int = 20,
    seed: int = 42,
) -> list[tuple]:
    """Randomly sample at most max_trials hyperparameter combinations."""

    # Build every n-gram, vocabulary size, and C combination.
    grid = list(
        itertools.product(
            ngram_ranges,
            vocab_sizes,
            c_values,
        )
    )

    # Shuffle with a fixed seed for reproducibility.
    random.Random(seed).shuffle(grid)

    return grid[:max_trials]

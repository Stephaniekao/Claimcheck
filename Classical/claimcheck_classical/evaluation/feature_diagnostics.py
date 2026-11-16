"""
ClaimCheck Classical Model - Feature Diagnostics

Diagnostics for evidence-aware features on RAGBench training claims:
1. Single-feature ROC AUC by subset
2. Feature means by label
3. Sample claims with their best-matching evidence sentence

Only the official training splits are used.
All Unsupported claims are kept, and Supported claims are sampled.
Sampling Supported claims does not change a feature's ROC AUC
in expectation, because ROC AUC compares the two classes pairwise.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from claimcheck_classical.data_loader import load_ragbench_claims
from claimcheck_classical.preprocessing import preprocess_claims
from claimcheck_classical.features.text_features import (
    create_tfidf_vectorizer,
)
from claimcheck_classical.features.evidence_features import (
    FULL_FEATURE_COLUMNS,
    build_full_feature_matrix,
    build_static_features,
    split_evidence,
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

SUBSETS = ["FinQA", "TATQA"]

DEFAULT_SUPPORTED_SAMPLE = 20000
DEFAULT_EXAMPLES = 8
DEFAULT_SEED = 42

MAX_TEXT_CHARS = 300

# Features shown next to each example claim.
EXAMPLE_FEATURES = [
    "numerical_consistency",
    "fuzzy_number_match",
    "claim_token_coverage",
    "max_sentence_cosine",
    "claim_number_count",
]


# ============================================================
# 2. LOAD SAMPLE AND BUILD FEATURES
# ============================================================

def load_sample(
    supported_sample: int,
    seed: int,
) -> pd.DataFrame:
    """Keep all Unsupported claims and sample Supported claims."""

    frames = []

    for subset in SUBSETS:

        # Load the original training split only.
        df = load_ragbench_claims(
            subset=subset,
            split="train",
        )

        unsupported = df[df["label"] == "Unsupported"]
        supported = df[df["label"] == "Supported"]

        # Sample Supported claims with a fixed seed.
        supported = supported.sample(
            n=min(supported_sample, len(supported)),
            random_state=seed,
        )

        print(
            f"{subset}: {len(unsupported)} Unsupported, "
            f"{len(supported)} Supported sampled"
        )

        frames.append(unsupported)
        frames.append(supported)

    return pd.concat(frames, ignore_index=True)


def build_features(df: pd.DataFrame):
    """Fit TF-IDF and build all thirteen features."""

    # Normalize claim and evidence text.
    processed = preprocess_claims(df)

    # Fit TF-IDF on the sampled training text only.
    vectorizer = create_tfidf_vectorizer()

    vectorizer.fit(
        processed["claim_clean"].tolist()
        + processed["evidence_clean"].tolist()
    )

    # Compute features once with the shared static part.
    static = build_static_features(processed)

    features = build_full_feature_matrix(
        processed,
        vectorizer,
        static=static,
    )

    # Attach labels and subsets for analysis.
    result = processed.reset_index(drop=True).copy()

    features = features.reset_index(drop=True)

    for column in FULL_FEATURE_COLUMNS:
        result[column] = features[column]

    result["is_unsupported"] = (
        result["label"] == "Unsupported"
    ).astype(int)

    return result, vectorizer


# ============================================================
# 3. SINGLE-FEATURE ROC AUC
# ============================================================

def single_feature_auc(df: pd.DataFrame) -> pd.DataFrame:
    """Compute each feature's ROC AUC for detecting Unsupported.

    AUC above 0.5: the feature is higher for Unsupported claims.
    AUC below 0.5: the feature is lower for Unsupported claims.
    Strength is the distance from 0.5, whatever the direction.
    """

    def auc(frame: pd.DataFrame, column: str) -> float:

        # A constant feature or a single class has no ranking.
        if (
            frame["is_unsupported"].nunique() < 2
            or frame[column].nunique() < 2
        ):
            return np.nan

        return float(
            roc_auc_score(
                frame["is_unsupported"],
                frame[column],
            )
        )

    rows = []

    for column in FULL_FEATURE_COLUMNS:

        row = {
            "feature": column,
            "auc_all": auc(df, column),
        }

        # Compute the same AUC inside each subset.
        for subset in SUBSETS:
            row[f"auc_{subset}"] = auc(
                df[df["subset"] == subset],
                column,
            )

        row["strength"] = abs(row["auc_all"] - 0.5)

        rows.append(row)

    table = pd.DataFrame(rows)

    return table.sort_values(
        "strength",
        ascending=False,
    ).reset_index(drop=True)


# ============================================================
# 4. FEATURE MEANS BY LABEL
# ============================================================

def means_by_label(df: pd.DataFrame) -> pd.DataFrame:
    """Compare mean feature values for each label and subset."""

    table = (
        df.groupby(["subset", "label"])[FULL_FEATURE_COLUMNS]
        .mean()
        .T
    )

    return table


# ============================================================
# 5. EXAMPLE CLAIMS
# ============================================================

def best_evidence_sentence(
    vectorizer,
    claim_clean: str,
    evidence: str,
) -> tuple[str, float]:
    """Find the evidence sentence most similar to the claim."""

    sentences = split_evidence(evidence)

    if not sentences:
        return "", 0.0

    claim_vector = vectorizer.transform([claim_clean])
    sentence_vectors = vectorizer.transform(sentences)

    # TF-IDF vectors are L2-normalized, so dot product is cosine.
    similarities = (
        sentence_vectors @ claim_vector.T
    ).toarray().ravel()

    best = int(np.argmax(similarities))

    return sentences[best], float(similarities[best])


def print_examples(
    df: pd.DataFrame,
    vectorizer,
    label: str,
    count: int,
    seed: int,
) -> None:
    """Print random claims of one label with their best evidence."""

    pool = df[df["label"] == label]

    if pool.empty:
        return

    sample = pool.sample(
        n=min(count, len(pool)),
        random_state=seed,
    )

    print(f"\n========== {label.upper()} EXAMPLES ==========")

    for index, (_, row) in enumerate(sample.iterrows(), start=1):

        sentence, score = best_evidence_sentence(
            vectorizer,
            row["claim_clean"],
            row["evidence"],
        )

        print(f"\n--- {label} {index} ({row['subset']}) ---")
        print(f"CLAIM: {row['claim'][:MAX_TEXT_CHARS]}")
        print(
            f"BEST EVIDENCE (cosine {score:.3f}): "
            f"{sentence[:MAX_TEXT_CHARS]}"
        )

        features = ", ".join(
            f"{name}={row[name]:.2f}"
            for name in EXAMPLE_FEATURES
        )

        print(f"FEATURES: {features}")


# ============================================================
# 6. COMMAND LINE INTERFACE
# ============================================================

def main() -> None:
    """Run feature diagnostics from the command line."""

    parser = argparse.ArgumentParser(
        description="Diagnose ClaimCheck feature signal."
    )

    parser.add_argument(
        "--supported-sample",
        type=int,
        default=DEFAULT_SUPPORTED_SAMPLE,
        help="Supported claims sampled per subset.",
    )

    parser.add_argument(
        "--examples",
        type=int,
        default=DEFAULT_EXAMPLES,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )

    args = parser.parse_args()

    print("========== LOADING SAMPLE ==========")

    df = load_sample(
        supported_sample=args.supported_sample,
        seed=args.seed,
    )

    print("\n========== BUILDING FEATURES ==========")

    df, vectorizer = build_features(df)

    print("\n========== SINGLE-FEATURE ROC AUC ==========")
    print("(above 0.5: higher for Unsupported, below 0.5: lower)")
    print(
        single_feature_auc(df)
        .round(3)
        .to_string(index=False)
    )

    print("\n========== FEATURE MEANS BY LABEL ==========")
    print(means_by_label(df).round(3).to_string())

    for label in ("Unsupported", "Supported"):
        print_examples(
            df,
            vectorizer,
            label,
            args.examples,
            args.seed,
        )


if __name__ == "__main__":
    main()

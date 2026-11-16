
"""
ClaimCheck Classical Model - Text Features

Text feature extraction for:
1. TF-IDF vectorization
2. Claim-evidence cosine similarity

The TF-IDF vocabulary is fitted on training data only.
Claims and evidence share the same vocabulary.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


# ============================================================
# 1. TF-IDF VECTORIZER
# ============================================================

def create_tfidf_vectorizer(
    ngram_range: tuple[int, int] = (1, 2),
    max_features: int = 10000,
) -> TfidfVectorizer:
    """Create a TF-IDF vectorizer without fitting it."""

    vectorizer = TfidfVectorizer(
        ngram_range=ngram_range,
        max_features=max_features,
        lowercase=False,
        norm="l2",
    )

    return vectorizer



# ============================================================
# 2. CLAIM-EVIDENCE COSINE SIMILARITY
# ============================================================

def compute_cosine_similarity(
    vectorizer: TfidfVectorizer,
    claims: list[str],
    evidence: list[str],
) -> np.ndarray:
    """Compute row-wise TF-IDF cosine similarity."""

    # Each claim must have one corresponding evidence passage.
    if len(claims) != len(evidence):
        raise ValueError(
            "Claims and evidence must have the same length."
        )

    # Transform both texts using the same fitted vocabulary.
    claim_vectors = vectorizer.transform(claims)
    evidence_vectors = vectorizer.transform(evidence)

    # TF-IDF vectors are L2-normalized.
    # Their dot product equals cosine similarity.
    similarities = claim_vectors.multiply(
        evidence_vectors
    ).sum(axis=1)

    # Return one similarity score per claim-evidence pair.
    return np.asarray(similarities).ravel()

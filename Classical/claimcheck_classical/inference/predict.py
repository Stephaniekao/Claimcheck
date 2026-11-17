"""Run three-label inference using a trained ClaimCheck classical model."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from claimcheck_classical.model_io import load_model
from claimcheck_classical.preprocessing import normalize_text, preprocess_claims
from claimcheck_classical.shared import normalizer_cannot_read
from claimcheck_classical.features.evidence_features import (
    build_full_feature_matrix,
    with_claim_text,
    FULL_FEATURE_COLUMNS,
)


NO_PASSAGE_REASON = "no cited passage"
UNREADABLE_NUMBER_REASON = "normalizer could not read a number"
NO_THRESHOLD_REASON = "no feasible thresholds"


def classify_support_probability(
    p_supported: float,
    lower_threshold: float,
    upper_threshold: float,
) -> str:
    """Convert a support probability into a three-label decision."""

    if not (
        np.isfinite(lower_threshold)
        and np.isfinite(upper_threshold)
        and 0 <= lower_threshold < upper_threshold <= 1
    ):
        raise ValueError(
            "Thresholds must satisfy 0 <= lower < upper <= 1."
        )

    if not np.isfinite(p_supported) or not 0 <= p_supported <= 1:
        raise ValueError(
            "Support probability must be between 0 and 1."
        )

    if p_supported < lower_threshold:
        return "Unsupported"

    if p_supported > upper_threshold:
        return "Supported"

    return "Uncertain"


def load_frozen_model(metadata_path) -> tuple:
    """Load the model, vectorizer, and thresholds saved by tune.py.

    The metadata file sits next to the model file and records the
    thresholds selected on validation data.
    """

    metadata_path = Path(metadata_path)

    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Metadata file not found: {metadata_path}"
        )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    for key in ("model_file", "thresholds"):
        if key not in metadata:
            raise ValueError(f"Metadata is missing: {key}")

    model, vectorizer = load_model(
        metadata_path.with_name(metadata["model_file"])
    )

    return model, vectorizer, metadata["thresholds"]


def predict_with_frozen_model(
    claims: pd.DataFrame,
    metadata_path,
) -> pd.DataFrame:
    """Predict with the model and thresholds saved by tune.py.

    If no feasible thresholds were found on validation data,
    every claim is labeled Uncertain.
    """

    model, vectorizer, thresholds = load_frozen_model(metadata_path)

    return predict_claims(
        claims=claims,
        model=model,
        vectorizer=vectorizer,
        lower_threshold=thresholds.get("lower_threshold"),
        upper_threshold=thresholds.get("upper_threshold"),
        all_uncertain=not thresholds["feasible"],
    )


def predict_claims(
    claims: pd.DataFrame,
    model,
    vectorizer,
    lower_threshold: float | None = None,
    upper_threshold: float | None = None,
    all_uncertain: bool = False,
) -> pd.DataFrame:
    """Predict Supported, Unsupported, or Uncertain for claim-evidence pairs.

    The model is trained with binary labels:
        Supported = 0
        Unsupported = 1

    The final three-label decision uses P(Supported):
        P(Supported) < lower_threshold  -> Unsupported
        P(Supported) > upper_threshold  -> Supported
        Otherwise                       -> Uncertain

    The thresholds must be the values selected on validation data.
    Use predict_with_frozen_model() to read them from the saved file.

    A claim without a cited passage is labeled Uncertain instead of
    being guessed. When all_uncertain is True, every claim is labeled
    Uncertain, which is the fallback when no thresholds were feasible.
    """

    # Thresholds are required unless every claim is Uncertain.
    if not all_uncertain:
        if lower_threshold is None or upper_threshold is None:
            raise ValueError(
                "Thresholds are required. Pass the thresholds selected "
                "on validation data or use predict_with_frozen_model()."
            )

        if not (
            np.isfinite(lower_threshold)
            and np.isfinite(upper_threshold)
            and 0 <= lower_threshold < upper_threshold <= 1
        ):
            raise ValueError(
                "Thresholds must satisfy 0 <= lower < upper <= 1."
            )

    if not {"claim", "evidence"}.issubset(claims.columns):
        raise ValueError(
            "Input must contain claim and evidence columns."
        )

    if claims.empty:
        raise ValueError(
            "Input must contain at least one claim."
        )

    claim_text = claims["claim"].fillna("").astype(str)

    if claim_text.str.strip().eq("").any():
        raise ValueError(
            "Claims cannot be missing or empty."
        )

    # Preserve the original input without modifying it.
    input_df = claims.reset_index(drop=True).copy()

    # Find claims that have a cited passage.
    evidence_text = input_df["evidence"].fillna("").astype(str)
    has_passage = evidence_text.str.strip().ne("").to_numpy()

    n = len(input_df)

    p_supported = np.full(n, np.nan)
    p_unsupported = np.full(n, np.nan)

    features = pd.DataFrame(
        np.nan,
        index=range(n),
        columns=FULL_FEATURE_COLUMNS,
    )

    if has_passage.any():

        # Only claims with a passage are scored by the model.
        scored = input_df.loc[has_passage, ["claim", "evidence"]].copy()
        scored = scored.reset_index(drop=True)
        scored["claim"] = scored["claim"].astype(str)
        scored["evidence"] = scored["evidence"].astype(str)

        # Apply the same preprocessing used during training.
        processed = preprocess_claims(scored)

        # Transform with the fitted TF-IDF vectorizer.
        # Never fit or refit during inference.
        # The original evidence should store one sentence per line
        # so that sentence-level similarity can be computed.
        X = build_full_feature_matrix(
            processed,
            vectorizer,
        )

        if not np.isfinite(X.to_numpy()).all():
            raise ValueError(
                "Inference features contain NaN or infinity."
            )

        # Ensure consistent feature order.
        X = X[FULL_FEATURE_COLUMNS]

        if 1 not in model.classes_:
            raise ValueError(
                "Model does not contain Unsupported class."
            )

        unsupported_index = list(model.classes_).index(1)

        # The model also reads the normalized claim text.
        X_model = with_claim_text(X, processed)

        scored_unsupported = model.predict_proba(
            X_model
        )[:, unsupported_index]

        if not np.isfinite(scored_unsupported).all():
            raise ValueError(
                "Predicted probabilities contain NaN or infinity."
            )

        p_unsupported[has_passage] = scored_unsupported
        p_supported[has_passage] = 1.0 - scored_unsupported
        features.loc[has_passage, :] = X.to_numpy()

    # Claims with a number the normalizer cannot read.
    unreadable = np.array([
        normalizer_cannot_read(normalize_text(text))
        for text in claim_text.tolist()
    ])

    predicted_labels = []
    reasons = []

    for index in range(n):

        # No cited passage: do not guess.
        if not has_passage[index]:
            predicted_labels.append("Uncertain")
            reasons.append(NO_PASSAGE_REASON)

        # Number the normalizer cannot read: do not guess
        # (Experiment Card, Field 12).
        elif unreadable[index]:
            predicted_labels.append("Uncertain")
            reasons.append(UNREADABLE_NUMBER_REASON)

        # No feasible thresholds: every claim is Uncertain.
        elif all_uncertain:
            predicted_labels.append("Uncertain")
            reasons.append(NO_THRESHOLD_REASON)

        else:
            predicted_labels.append(
                classify_support_probability(
                    float(p_supported[index]),
                    lower_threshold,
                    upper_threshold,
                )
            )
            reasons.append("")

    result = input_df.copy()

    result["predicted_label"] = predicted_labels
    result["reason"] = reasons

    result["p_supported"] = p_supported
    result["p_unsupported"] = p_unsupported

    result["lower_threshold"] = (
        np.nan if lower_threshold is None else lower_threshold
    )
    result["upper_threshold"] = (
        np.nan if upper_threshold is None else upper_threshold
    )

    # Include all thirteen features for explainability.
    # Claims without a passage have missing feature values.
    result = pd.concat(
        [
            result,
            features,
        ],
        axis=1,
    )

    return result


def predict_single_claim(
    claim: str,
    evidence: str,
    model,
    vectorizer,
    lower_threshold: float | None = None,
    upper_threshold: float | None = None,
    all_uncertain: bool = False,
) -> dict:
    """Predict one claim-evidence pair using three-label inference."""

    if not isinstance(claim, str) or not claim.strip():
        raise ValueError(
            "Claim must be a non-empty string."
        )

    # An empty evidence string is allowed. The claim becomes Uncertain.
    if not isinstance(evidence, str):
        raise ValueError(
            "Evidence must be a string."
        )

    input_df = pd.DataFrame({
        "claim": [claim],
        "evidence": [evidence],
    })

    result = predict_claims(
        claims=input_df,
        model=model,
        vectorizer=vectorizer,
        lower_threshold=lower_threshold,
        upper_threshold=upper_threshold,
        all_uncertain=all_uncertain,
    )

    return result.iloc[0].to_dict()

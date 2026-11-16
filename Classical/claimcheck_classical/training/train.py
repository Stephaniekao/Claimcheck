"""Train the ClaimCheck classical Logistic Regression model."""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from claimcheck_classical.config import (
    CLAIM_TEXT_MIN_DF,
    CLAIM_TEXT_TOKEN_PATTERN,
    MODEL_CONFIG,
    RANDOM_SEED,
    USE_CLAIM_TEXT,
    USE_EXTRA_FEATURES,
)
from claimcheck_classical.data_loader import (
    load_ragbench_claims,
)
from claimcheck_classical.preprocessing import (
    preprocess_claims,
)
from claimcheck_classical.features.text_features import (
    create_tfidf_vectorizer,
)
from claimcheck_classical.features.evidence_features import (
    build_full_feature_matrix,
    with_claim_text,
    BASE_COLUMNS,
    FULL_FEATURE_COLUMNS,
    TEXT_COLUMN,
)


# ============================================================
# 1. TRAINING CONFIGURATION
# ============================================================

DEFAULT_SAMPLE_SIZE = 1000
DEFAULT_SAMPLING = "random"
DEFAULT_UNSUPPORTED_RATIO = 0.5

SUBSETS = ["FinQA", "TATQA"]

LABEL_MAPPING = {
    "Supported": 0,
    "Unsupported": 1,
}


# ============================================================
# 2. LOAD TRAINING DATA
# ============================================================

def load_training_data(
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    sampling: str = DEFAULT_SAMPLING,
    unsupported_ratio: float = DEFAULT_UNSUPPORTED_RATIO,
    full_data: bool = False,
) -> pd.DataFrame:
    """Load training claims from RAGBench FinQA and TATQA.

    When full_data=True:
        Load every claim from the official training splits.
        Do not perform sampling.

    When full_data=False:
        Preserve the existing random or stratified sampling.
    """

    if not isinstance(full_data, bool):
        raise ValueError("full_data must be a boolean.")

    if not full_data:
        if sample_size <= 0:
            raise ValueError("Sample size must be positive.")

        if sampling not in {"random", "stratified"}:
            raise ValueError(
                "Sampling must be 'random' or 'stratified'."
            )

        if not 0 < unsupported_ratio < 1:
            raise ValueError(
                "Unsupported ratio must be between 0 and 1."
            )

    frames = []

    for subset in SUBSETS:

        df = load_ragbench_claims(
            subset=subset,
            split="train",
        )

        if full_data:
            # Preserve all training claims and original labels.
            sampled = df.copy()

        else:
            n = min(sample_size, len(df))

            if sampling == "random":

                sampled = df.sample(
                    n=n,
                    random_state=RANDOM_SEED,
                )

            else:

                supported = df[
                    df["label"] == "Supported"
                ]

                unsupported = df[
                    df["label"] == "Unsupported"
                ]

                n_unsupported = round(
                    n * unsupported_ratio
                )

                n_supported = n - n_unsupported

                if (
                    n_supported > len(supported)
                    or n_unsupported > len(unsupported)
                ):
                    raise ValueError(
                        f"{subset}: Not enough claims for "
                        "the requested stratified sample."
                    )

                sampled_supported = supported.sample(
                    n=n_supported,
                    random_state=RANDOM_SEED,
                )

                sampled_unsupported = unsupported.sample(
                    n=n_unsupported,
                    random_state=RANDOM_SEED,
                )

                sampled = pd.concat(
                    [
                        sampled_supported,
                        sampled_unsupported,
                    ],
                    ignore_index=True,
                )

                sampled = sampled.sample(
                    frac=1,
                    random_state=RANDOM_SEED,
                )

        frames.append(sampled)

        print(
            f"{subset}: {len(sampled)} training claims"
        )

        print(
            sampled["label"]
            .value_counts()
            .to_string()
        )

    combined = pd.concat(
        frames,
        ignore_index=True,
    )

    print(
        f"\nTotal training claims: {len(combined)}"
    )

    return combined


# ============================================================
# 3. PREPARE FEATURES
# ============================================================

def prepare_training_features(
    df: pd.DataFrame,
    ngram_range: tuple[int, int] = (1, 2),
    max_features: int = 10000,
):
    """Fit TF-IDF and create the 13-feature training matrix."""

    processed = preprocess_claims(df)

    vectorizer = create_tfidf_vectorizer(
        ngram_range=ngram_range,
        max_features=max_features,
    )

    training_text = (
        processed["claim_clean"].tolist()
        + processed["evidence_clean"].tolist()
    )

    vectorizer.fit(training_text)

    X = build_full_feature_matrix(
        processed,
        vectorizer,
    )

    y = processed["label"].map(LABEL_MAPPING)

    if y.isna().any():
        raise ValueError(
            "Unknown labels found in training data."
        )

    y = y.astype(int)

    if not np.isfinite(X.to_numpy()).all():
        raise ValueError(
            "Training features contain NaN or infinity."
        )

    # Attach claim text after the numeric check.
    X = with_claim_text(X, processed)

    return X, y, vectorizer


# ============================================================
# 4. TRAIN LOGISTIC REGRESSION
# ============================================================

def model_numeric_columns() -> list[str]:
    """Return the numeric features used by the model.

    The feature matrix always holds all thirteen features.
    The model reads the seven card features unless
    USE_EXTRA_FEATURES is enabled in config.py.
    """

    if USE_EXTRA_FEATURES:
        return list(FULL_FEATURE_COLUMNS)

    return list(BASE_COLUMNS)


def train_logistic_regression(
    X: pd.DataFrame,
    y: pd.Series,
    C: float | None = None,
    ngram_range: tuple[int, int] = (1, 2),
    max_features: int = 10000,
    random_state: int | None = None,
) -> Pipeline:
    """Train scaled numeric features plus claim TF-IDF with L2 LR.

    X must contain the thirteen numeric features and, when
    USE_CLAIM_TEXT is True, the normalized claim text column.

    C overrides MODEL_CONFIG when given (used by the tuning search).
    ngram_range and max_features configure the claim text TF-IDF.
    Scaling matters because the numeric features mix 0-1 scores,
    0/1 flags and log-scale counts, and L2 regularization is
    scale-sensitive.
    """

    if y.nunique() < 2:
        raise ValueError(
            "Training data must contain both classes."
        )

    config = dict(MODEL_CONFIG)

    if C is not None:
        config["C"] = float(C)

    # The seed run (Field 5) changes the solver seed as well.
    if random_state is not None:
        config["random_state"] = int(random_state)

    # Standardize the numeric features.
    transformers = [
        ("numeric", StandardScaler(), model_numeric_columns()),
    ]

    # Add the claim text bag of words when enabled.
    if USE_CLAIM_TEXT:
        transformers.append((
            "claim_text",
            TfidfVectorizer(
                ngram_range=ngram_range,
                max_features=max_features,
                min_df=CLAIM_TEXT_MIN_DF,
                token_pattern=CLAIM_TEXT_TOKEN_PATTERN,
                lowercase=False,
                norm="l2",
            ),
            TEXT_COLUMN,
        ))

    # Columns that are not listed (such as unused text) are dropped.
    preprocessor = ColumnTransformer(
        transformers,
        sparse_threshold=1.0,
    )

    model = Pipeline([
        ("features", preprocessor),
        ("lr", LogisticRegression(**config)),
    ])

    model.fit(X, y)

    return model


def coefficient_table(model: Pipeline) -> pd.DataFrame:
    """List every model coefficient with its feature group."""

    names = model.named_steps["features"].get_feature_names_out()
    coefficients = model.named_steps["lr"].coef_[0]

    # Names look like "numeric__cosine_similarity".
    groups = [name.split("__", 1)[0] for name in names]
    features = [name.split("__", 1)[1] for name in names]

    return pd.DataFrame({
        "group": groups,
        "feature": features,
        "coefficient": coefficients,
    })


# ============================================================
# 5. DISPLAY TRAINING SUMMARY
# ============================================================

def print_training_summary(
    model: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
) -> None:
    """Print model information and training predictions."""

    print("\n========== TRAINING SUMMARY ==========")

    print(f"Training claims: {len(X)}")
    print(f"Numeric feature count: {len(model_numeric_columns())}")
    print(f"Extra features enabled: {USE_EXTRA_FEATURES}")
    print(f"Claim text TF-IDF: {USE_CLAIM_TEXT}")

    print("\nLabel distribution:")
    print(y.value_counts().sort_index().to_string())

    print("\nNumeric feature columns:")
    for column in model_numeric_columns():
        print(column)

    print("\nModel classes:")
    print(model.classes_)

    unsupported_index = list(
        model.classes_
    ).index(1)

    probabilities = model.predict_proba(X)[
        :,
        unsupported_index,
    ]

    print("\nP(Unsupported) summary:")
    print(
        pd.Series(probabilities)
        .describe()
        .round(4)
        .to_string()
    )

    print("\nModel coefficients:")

    coefficients = coefficient_table(model)

    numeric = coefficients[coefficients["group"] == "numeric"]

    print(
        numeric[["feature", "coefficient"]]
        .round(4)
        .to_string(index=False)
    )

    text = coefficients[coefficients["group"] == "claim_text"]

    if not text.empty:
        ranked = text.sort_values("coefficient")

        print("\nClaim n-grams toward Unsupported:")
        print(
            ranked.tail(10)[::-1][["feature", "coefficient"]]
            .round(3)
            .to_string(index=False)
        )

        print("\nClaim n-grams toward Supported:")
        print(
            ranked.head(10)[["feature", "coefficient"]]
            .round(3)
            .to_string(index=False)
        )

    print("\nTraining completed successfully.")


# ============================================================
# 6. COMMAND LINE INTERFACE
# ============================================================

def main() -> None:
    """Run a Logistic Regression training experiment."""

    parser = argparse.ArgumentParser(
        description="Train ClaimCheck Logistic Regression."
    )

    parser.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help="Maximum training claims per subset.",
    )

    parser.add_argument(
        "--sampling",
        choices=["random", "stratified"],
        default=DEFAULT_SAMPLING,
        help="Training sampling strategy.",
    )

    parser.add_argument(
        "--unsupported-ratio",
        type=float,
        default=DEFAULT_UNSUPPORTED_RATIO,
        help="Unsupported fraction for stratified sampling.",
    )

    parser.add_argument(
        "--full-data",
        action="store_true",
        help="Use all claims from the official training splits.",
    )

    args = parser.parse_args()

    print("========== LOADING TRAINING DATA ==========")

    df = load_training_data(
        sample_size=args.sample_size,
        sampling=args.sampling,
        unsupported_ratio=args.unsupported_ratio,
        full_data=args.full_data,
    )

    print("\n========== PREPARING FEATURES ==========")

    X, y, vectorizer = prepare_training_features(df)

    print(
        "Numeric feature matrix shape: "
        f"{X[FULL_FEATURE_COLUMNS].shape}"
    )
    print(f"Claim text column attached: {TEXT_COLUMN in X.columns}")

    print("\n========== TRAINING MODEL ==========")

    model = train_logistic_regression(X, y)

    print_training_summary(
        model,
        X,
        y,
    )


if __name__ == "__main__":
    main()
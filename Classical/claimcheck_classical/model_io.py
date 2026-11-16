"""Save and load trained ClaimCheck classical models."""

from pathlib import Path

import joblib


def save_model(model, vectorizer, path):
    """Save the trained classifier and TF-IDF vectorizer."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    artifact = {
        "model": model,
        "vectorizer": vectorizer,
    }

    joblib.dump(artifact, path)

    return path


def load_model(path):
    """Load the trained classifier and TF-IDF vectorizer."""

    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Model file not found: {path}"
        )

    artifact = joblib.load(path)

    if not isinstance(artifact, dict):
        raise ValueError("Invalid model artifact.")

    if "model" not in artifact or "vectorizer" not in artifact:
        raise ValueError(
            "Model artifact must contain model and vectorizer."
        )

    return artifact["model"], artifact["vectorizer"]

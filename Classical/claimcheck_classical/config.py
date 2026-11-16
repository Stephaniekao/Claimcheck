"""
ClaimCheck Classical Model - Configuration

Centralized configuration for:
1. Project paths
2. Reproducibility
3. Logistic Regression
4. TF-IDF hyperparameter tuning
5. Threshold selection
6. Evaluation
7. Model and experiment outputs
"""

from pathlib import Path


# ============================================================
# 1. PROJECT PATHS
# ============================================================

CONFIG_FILE = Path(__file__).resolve()

PACKAGE_DIR = CONFIG_FILE.parent
CLASSICAL_DIR = PACKAGE_DIR.parent
PROJECT_ROOT = CLASSICAL_DIR.parent

# Data directory
DATA_DIR = PROJECT_ROOT / "Data"

# Classical model directories
MODELS_DIR = CLASSICAL_DIR / "models"
OUTPUTS_DIR = CLASSICAL_DIR / "outputs"


# ============================================================
# 2. REPRODUCIBILITY
# ============================================================

RANDOM_SEED = 42

# Maximum number of hyperparameter trials
MAX_TRIALS = 20


# ============================================================
# 3. MODEL CONFIGURATION
# ============================================================

MODEL_NAME = "LogisticRegression"

# Experiment card configuration (Field 3):
# cosine similarity, numerical consistency, reporting period
# consistency, and financial scope consistency.
# Both extensions below are False by default.
# Enable them only after the team agrees to extend the card.

# Add the six extra numeric features (sentence-level cosine,
# token coverage, fuzzy number match, claim number count,
# and claim length) to the model input.
USE_EXTRA_FEATURES = False

# Add claim text TF-IDF to the model input.
USE_CLAIM_TEXT = False

# Minimum number of claims containing an n-gram.
CLAIM_TEXT_MIN_DF = 2

# Alphabetic words only, so the text model learns wording
# rather than memorizing individual numbers or years.
CLAIM_TEXT_TOKEN_PATTERN = r"(?u)\b[a-z][a-z]+\b"

# L2 regularization is the default penalty of LogisticRegression.
# The penalty is not set here because newer scikit-learn versions
# deprecate the argument.
MODEL_CONFIG = {
    "solver": "liblinear",
    "max_iter": 2000,
    "random_state": RANDOM_SEED,
}


# ============================================================
# 4. HYPERPARAMETER SEARCH SPACE
# ============================================================

# TF-IDF n-gram candidates
TFIDF_NGRAM_RANGE = [
    (1, 1),
    (1, 2),
    (1, 3),
]

# Maximum TF-IDF vocabulary size
TFIDF_MAX_FEATURES = [
    5000,
    10000,
    20000,
]

# Logistic Regression regularization strength
LR_C_VALUES = [
    0.01,
    0.1,
    1.0,
    10.0,
]

# Hyperparameter tuning settings
TUNING_CONFIG = {
    "max_trials": MAX_TRIALS,
    "selection_metric": "f1",
}


# ============================================================
# 5. THRESHOLD SELECTION
# ============================================================

# Threshold candidates are evaluated on validation data only.
# Valid pairs must satisfy low < high.

THRESHOLD_CONFIG = {
    "low_candidates": [
        round(i * 0.05, 2)
        for i in range(1, 11)
    ],
    "high_candidates": [
        round(i * 0.05, 2)
        for i in range(10, 20)
    ],
    "selection_metric": "f1",
}


# ============================================================
# 6. EVALUATION CONFIGURATION
# ============================================================

EVALUATION_CONFIG = {
    "positive_label": "Unsupported",
    "target_recall": 0.80,
    "ece_bins": 10,
    "save_predictions": True,
    "save_plots": True,
}


# ============================================================
# 7. MODEL OUTPUT PATHS
# ============================================================

MODEL_PATH = MODELS_DIR / "logistic_regression.joblib"
THRESHOLDS_PATH = MODELS_DIR / "thresholds.json"
METADATA_PATH = MODELS_DIR / "metadata.json"


# ============================================================
# 8. EVALUATION OUTPUT DIRECTORIES
# ============================================================

PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
METRICS_DIR = OUTPUTS_DIR / "metrics"
PLOTS_DIR = OUTPUTS_DIR / "plots"

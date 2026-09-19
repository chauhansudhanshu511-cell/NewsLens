"""Central configuration for NewsLens.

Every path, constant and label definition lives here so training, inference,
the web app and the tests all agree with each other.
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_DATASET_PATH = DATA_DIR / "WELFake_Dataset.csv"


def get_model_dir() -> Path:
    """Folder that holds the trained artifacts.

    It can be overridden with the NEWSLENS_MODEL_DIR environment variable
    (the tests use this to point the app at a temporary folder).
    """
    override = os.environ.get("NEWSLENS_MODEL_DIR")
    return Path(override) if override else PROJECT_ROOT / "models"


MODEL_FILENAME = "newslens_pipeline.joblib"
METRICS_FILENAME = "metrics.json"
METADATA_FILENAME = "metadata.json"

# ---------------------------------------------------------------------------
# Reproducibility and splitting
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
TEST_SIZE = 0.15        # share of groups kept for the final, held-out test set
VALIDATION_SIZE = 0.15  # share of groups used for model selection

# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------
# Internal labels used by the model everywhere in this project.
# "fake" is the positive class (1), because detecting it is the goal.
LABEL_REAL = 0
LABEL_FAKE = 1
LABEL_NAMES = {LABEL_REAL: "real", LABEL_FAKE: "fake"}
DISPLAY_LABELS = {LABEL_REAL: "Likely real", LABEL_FAKE: "Likely fake"}

# ---------------------------------------------------------------------------
# Dataset: WELFake (Verma, Agrawal & Prodan, 2021) - see data/README.md
# ---------------------------------------------------------------------------
DATASET_INFO = {
    "name": "WELFake dataset for fake news detection in text data",
    "short_name": "WELFake",
    "authors": "Pawan Kumar Verma, Prateek Agrawal, Radu Prodan",
    "source_url": "https://zenodo.org/records/4561253",
    "doi": "10.5281/zenodo.4561253",
    "license": "CC BY 4.0 (Creative Commons Attribution 4.0 International)",
    "file_name": "WELFake_Dataset.csv",
    "zenodo_md5": "73c9675a4b3d09f86a6933d0b8d7d908",
    "paper": (
        "Verma, P. K., Agrawal, P., & Prodan, R. (2021). WELFake: Word Embedding Over "
        "Linguistic Features for Fake News Detection. IEEE Transactions on Computational "
        "Social Systems. doi:10.1109/TCSS.2021.3068519"
    ),
}

# Column names in the raw CSV.
RAW_TITLE_COL = "title"
RAW_TEXT_COL = "text"
RAW_LABEL_COL = "label"

# Mapping from the raw WELFake "label" value to our internal label.
# The Zenodo page describes the column as "0 = fake and 1 = real". The file
# itself behaves the other way round: articles with Reuters datelines (real
# news taken from Reuters) carry label 0. Run
# `python -m src.inspect_dataset` to see this check on your copy.
# Details and evidence are in data/README.md.
RAW_LABEL_MAP = {0: LABEL_REAL, 1: LABEL_FAKE}

# ---------------------------------------------------------------------------
# Input validation for the web app
# ---------------------------------------------------------------------------
MIN_CHARS = 20            # below this the input is rejected
MIN_WORDS = 4             # below this the input is rejected
SHORT_TEXT_WORDS = 60     # below this a "short text, less reliable" warning is shown
MAX_CHARS = 30_000        # above this the input is rejected

# ---------------------------------------------------------------------------
# Model hyperparameters
# ---------------------------------------------------------------------------
TFIDF_PARAMS = {
    "lowercase": True,
    "strip_accents": "unicode",
    "ngram_range": (1, 2),
    "min_df": 3,
    "max_df": 0.9,
    "max_features": 100_000,
    "sublinear_tf": True,
}

# Candidate hyperparameters, compared on the validation set only.
LOGREG_C_VALUES = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
NB_ALPHA_VALUES = [0.01, 0.05, 0.1, 0.5, 1.0]

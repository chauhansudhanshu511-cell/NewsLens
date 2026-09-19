"""Shared test fixtures.

IMPORTANT: the tiny model built here is trained on a handful of invented
sentences. It exists only to test the plumbing (saving, loading, output
structure). It is NOT a fake-news detector and must never be deployed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pytest
import sklearn

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src.model import LOGREG, build_pipeline  # noqa: E402

REAL_STYLE = [
    "The committee approved the annual budget on Tuesday after a public hearing.",
    "Officials said the new bridge will open to traffic next month, according to a statement.",
    "The central bank kept interest rates unchanged, citing stable inflation figures.",
    "Lawmakers debated the education bill and scheduled a final vote for next week.",
    "The company reported quarterly revenue in line with analyst expectations.",
    "The ministry said talks with regional partners would continue in the spring.",
]
FAKE_STYLE = [
    "SHOCKING secret they don't want you to know, share before it gets deleted!!!",
    "You won't believe this miracle cure the elites are hiding from everyone!!!",
    "BREAKING: insiders reveal the shocking truth the media refuses to report!!!",
    "Wake up people, this secret plot exposes everything, share now before it's gone!",
    "Unbelievable shocking video proves they lied to you all along, must watch!!!",
    "The shocking truth about the secret elites will blow your mind, share it now!",
]


@pytest.fixture(scope="session")
def toy_data():
    texts = REAL_STYLE + FAKE_STYLE
    labels = [config.LABEL_REAL] * len(REAL_STYLE) + [config.LABEL_FAKE] * len(FAKE_STYLE)
    return texts, labels


@pytest.fixture(scope="session")
def toy_pipeline(toy_data):
    texts, labels = toy_data
    pipe = build_pipeline(LOGREG, 10.0, tfidf_overrides={"min_df": 1, "max_df": 1.0})
    pipe.fit(texts, labels)
    return pipe


@pytest.fixture()
def fixture_model_dir(tmp_path, toy_pipeline):
    """A models/ folder containing the toy pipeline and fixture metadata (no metrics)."""
    joblib.dump(toy_pipeline, tmp_path / config.MODEL_FILENAME)
    (tmp_path / config.METADATA_FILENAME).write_text(json.dumps({
        "project": "NewsLens",
        "is_test_fixture": True,
        "model_name": "TEST FIXTURE - not a real model",
        "sklearn_version": sklearn.__version__,
        "vocabulary_size": len(toy_pipeline.named_steps["tfidf"].vocabulary_),
        "dataset": {"name": "synthetic test sentences"},
    }))
    return tmp_path

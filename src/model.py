"""Builds the scikit-learn Pipelines used by NewsLens.

Each pipeline is: clean text -> TF-IDF -> classifier.
Because TF-IDF sits *inside* the pipeline, calling ``fit`` on the training
set learns the vocabulary and IDF weights from training data only, and the
saved pipeline applies exactly the same steps at prediction time.
"""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from src import config
from src.preprocessing import make_cleaner

LOGREG = "logistic_regression"
NAIVE_BAYES = "multinomial_nb"


def build_vectorizer(**overrides) -> TfidfVectorizer:
    params = {**config.TFIDF_PARAMS, **overrides}
    return TfidfVectorizer(dtype=np.float32, **params)


def build_pipeline(kind: str, param: float, tfidf_overrides: dict | None = None) -> Pipeline:
    """Create an *unfitted* pipeline.

    kind  : "logistic_regression" (param = C) or "multinomial_nb" (param = alpha)
    """
    if kind == LOGREG:
        clf = LogisticRegression(C=param, solver="liblinear", max_iter=2000,
                                 random_state=config.RANDOM_SEED)
    elif kind == NAIVE_BAYES:
        clf = MultinomialNB(alpha=param)
    else:
        raise ValueError(f"Unknown model kind: {kind!r}")

    return Pipeline([
        ("clean", make_cleaner()),
        ("tfidf", build_vectorizer(**(tfidf_overrides or {}))),
        ("clf", clf),
    ])


def describe(kind: str, param: float) -> str:
    if kind == LOGREG:
        return f"Logistic Regression (C={param:g})"
    return f"Multinomial Naive Bayes (alpha={param:g})"


def slim_for_saving(pipeline: Pipeline) -> Pipeline:
    """Drop the vectorizer's ``stop_words_`` set (only used for inspection).

    It lists every term that was cut by min_df/max_df/max_features and can be
    many megabytes. Removing it does not change predictions.
    """
    vec = pipeline.named_steps.get("tfidf")
    if vec is not None and hasattr(vec, "stop_words_"):
        vec.stop_words_ = None
    return pipeline

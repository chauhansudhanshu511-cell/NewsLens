"""Loading the trained artifacts and making predictions.

Security note: ``joblib.load`` can execute code hidden in a malicious file.
Only load model files you created yourself or that come from a source you
trust (for example, the artifact you committed to your own repository).
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import sklearn

from src import config
from src.preprocessing import clean_text


class ModelNotFoundError(FileNotFoundError):
    """Raised when the trained pipeline has not been created yet."""


@dataclass
class ModelBundle:
    pipeline: object
    metadata: dict
    metrics: dict | None


@dataclass
class Prediction:
    label: int                 # 0 = real, 1 = fake
    label_name: str            # "real" / "fake"
    display_label: str         # "Likely real" / "Likely fake"
    prob_fake: float           # model estimate, NOT the chance the news is false
    prob_real: float

    @property
    def confidence(self) -> float:
        return max(self.prob_fake, self.prob_real)


def missing_model_message(model_dir: Path) -> str:
    return (
        f"No trained model found in '{model_dir}'. Train one first:\n"
        "  1. Download WELFake_Dataset.csv into the data/ folder (see data/README.md)\n"
        "  2. Run:  python -m src.train\n"
        f"This creates {config.MODEL_FILENAME}, {config.METADATA_FILENAME} and "
        f"{config.METRICS_FILENAME}."
    )


def load_bundle(model_dir: Path | None = None) -> ModelBundle:
    """Load pipeline + metadata (+ metrics if present). Never falls back to a dummy model."""
    model_dir = Path(model_dir) if model_dir else config.get_model_dir()
    model_path = model_dir / config.MODEL_FILENAME
    meta_path = model_dir / config.METADATA_FILENAME
    if not model_path.exists() or not meta_path.exists():
        raise ModelNotFoundError(missing_model_message(model_dir))

    metadata = json.loads(meta_path.read_text())
    trained_with = metadata.get("sklearn_version")
    if trained_with and trained_with != sklearn.__version__:
        warnings.warn(
            f"Model was trained with scikit-learn {trained_with} but {sklearn.__version__} "
            "is installed. Install the pinned versions from requirements.txt.",
            stacklevel=2,
        )
    pipeline = joblib.load(model_path)

    metrics_path = model_dir / config.METRICS_FILENAME
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else None
    return ModelBundle(pipeline=pipeline, metadata=metadata, metrics=metrics)


def _fake_column(pipeline) -> int:
    classes = list(pipeline.classes_)
    if config.LABEL_FAKE not in classes or config.LABEL_REAL not in classes:
        raise ValueError(f"Unexpected model classes {classes}; expected [0, 1].")
    return classes.index(config.LABEL_FAKE)


def predict_proba_fake(pipeline, texts: list[str]) -> np.ndarray:
    probs = pipeline.predict_proba(list(texts))
    return probs[:, _fake_column(pipeline)]


def predict(pipeline, text: str) -> Prediction:
    """Predict one text. The 0.5 threshold matches the pipeline's own predict()."""
    p_fake = float(predict_proba_fake(pipeline, [text])[0])
    label = config.LABEL_FAKE if p_fake >= 0.5 else config.LABEL_REAL
    return Prediction(
        label=label,
        label_name=config.LABEL_NAMES[label],
        display_label=config.DISPLAY_LABELS[label],
        prob_fake=p_fake,
        prob_real=1.0 - p_fake,
    )


def explain_terms(pipeline, text: str, top_n: int = 8) -> dict | None:
    """Terms that pushed this Logistic Regression prediction toward fake or real.

    For Logistic Regression the log-odds of "fake" is exactly
        intercept + sum_j (tfidf_j(text) * coef_j)
    so tfidf_j * coef_j is each term's exact contribution. Positive values push
    toward "fake", negative toward "real". Returns None for other classifiers.
    """
    clf = pipeline.named_steps.get("clf")
    vec = pipeline.named_steps.get("tfidf")
    if clf is None or vec is None or not hasattr(clf, "coef_"):
        return None

    # For binary models, coef_ points toward classes_[1]. We require that to be "fake".
    if list(clf.classes_) != [config.LABEL_REAL, config.LABEL_FAKE]:
        return None
    x = vec.transform([clean_text(text)])
    coef = clf.coef_[0]
    idx = x.indices
    contrib = x.data * coef[idx]
    names = vec.get_feature_names_out()
    order = np.argsort(contrib)
    fake = [(str(names[idx[i]]), float(contrib[i])) for i in order[::-1] if contrib[i] > 0][:top_n]
    real = [(str(names[idx[i]]), float(contrib[i])) for i in order if contrib[i] < 0][:top_n]
    return {
        "toward_fake": fake,
        "toward_real": real,
        "intercept": float(clf.intercept_[0]),
        "total_log_odds": float(clf.decision_function(x)[0]),
    }

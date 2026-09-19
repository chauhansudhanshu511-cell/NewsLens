"""Metric helpers: accuracy, precision, recall, F1 and the confusion matrix."""

from __future__ import annotations

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src import config


def compute_metrics(y_true, y_pred) -> dict:
    """Return JSON-friendly metrics. "fake" (label 1) is the positive class.

    confusion_matrix rows = actual label, columns = predicted label,
    both ordered [real, fake].
    """
    labels = [config.LABEL_REAL, config.LABEL_FAKE]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "n_samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_fake": float(precision_score(y_true, y_pred, pos_label=config.LABEL_FAKE,
                                                zero_division=0)),
        "recall_fake": float(recall_score(y_true, y_pred, pos_label=config.LABEL_FAKE,
                                          zero_division=0)),
        "f1_fake": float(f1_score(y_true, y_pred, pos_label=config.LABEL_FAKE, zero_division=0)),
        "precision_real": float(precision_score(y_true, y_pred, pos_label=config.LABEL_REAL,
                                                zero_division=0)),
        "recall_real": float(recall_score(y_true, y_pred, pos_label=config.LABEL_REAL,
                                          zero_division=0)),
        "f1_real": float(f1_score(y_true, y_pred, pos_label=config.LABEL_REAL, zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": {
            "labels": [config.LABEL_NAMES[l] for l in labels],
            "matrix": cm.tolist(),
            "layout": "rows = actual, columns = predicted",
        },
    }


def format_metrics(name: str, m: dict) -> str:
    """Readable one-block summary for the terminal."""
    cm = m["confusion_matrix"]["matrix"]
    return (
        f"{name}  (n={m['n_samples']})\n"
        f"  accuracy        {m['accuracy']:.4f}\n"
        f"  precision/recall/F1 (fake)  {m['precision_fake']:.4f} / {m['recall_fake']:.4f} / "
        f"{m['f1_fake']:.4f}\n"
        f"  precision/recall/F1 (real)  {m['precision_real']:.4f} / {m['recall_real']:.4f} / "
        f"{m['f1_real']:.4f}\n"
        f"  macro F1        {m['f1_macro']:.4f}\n"
        f"  confusion matrix [rows actual real/fake, cols predicted real/fake]\n"
        f"    {cm[0]}\n    {cm[1]}"
    )

"""Train, select and evaluate the NewsLens model.

Run from the project root:

    python -m src.train                       # uses data/WELFake_Dataset.csv
    python -m src.train --data path/to/file.csv

Steps
-----
1. Load the CSV through the WELFake schema adapter.
2. Clean it, drop empty articles, remove duplicates, group near-duplicates.
3. Split by group into train (70%) / validation (15%) / test (15%), seed 42.
4. Fit every candidate on TRAIN and score it on VALIDATION:
   - Logistic Regression (main model) for several C values
   - Multinomial Naive Bayes (baseline) for several alpha values
5. Pick the Logistic Regression setting with the best validation macro-F1.
6. Refit that setting on TRAIN + VALIDATION, then evaluate it ONCE on TEST.
7. Save the fitted pipeline, metrics.json and metadata.json to models/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from src import config
from src.data import load_raw_csv, prepare_dataset, split_by_group
from src.evaluation import compute_metrics, format_metrics
from src.model import LOGREG, NAIVE_BAYES, build_pipeline, describe, slim_for_saving


def file_md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def top_terms(pipeline, n: int = 25) -> dict:
    """Largest positive (-> fake) and negative (-> real) LR coefficients."""
    vec = pipeline.named_steps["tfidf"]
    coef = pipeline.named_steps["clf"].coef_[0]
    vocab = vec.get_feature_names_out()
    order = np.argsort(coef)
    return {
        "toward_fake": [[str(vocab[i]), round(float(coef[i]), 3)] for i in order[::-1][:n]],
        "toward_real": [[str(vocab[i]), round(float(coef[i]), 3)] for i in order[:n]],
    }


def evaluate_candidates(train: pd.DataFrame, val: pd.DataFrame) -> list[dict]:
    """Score every candidate on the validation set.

    All candidates share the same cleaning + TF-IDF settings, so those two
    steps are fitted ONCE on the training data and reused (much faster, and
    exactly equivalent to fitting each full Pipeline separately). The
    validation text is only ever transformed, never fitted on.
    """
    t0 = time.time()
    features = build_pipeline(LOGREG, 1.0)[:-1]          # clean -> tfidf, no classifier
    x_train = features.fit_transform(train["document"])
    x_val = features.transform(val["document"])
    print(f"  TF-IDF fitted on train: {x_train.shape[1]:,} features "
          f"({time.time() - t0:.0f}s)", flush=True)

    results = []
    grid = [(LOGREG, c) for c in config.LOGREG_C_VALUES] + \
           [(NAIVE_BAYES, a) for a in config.NB_ALPHA_VALUES]
    for kind, param in grid:
        start = time.time()
        clf = build_pipeline(kind, param).named_steps["clf"]
        clf.fit(x_train, train["label"])
        metrics = compute_metrics(val["label"], clf.predict(x_val))
        results.append({"kind": kind, "param": param, "name": describe(kind, param),
                        "validation": metrics, "fit_seconds": round(time.time() - start, 1)})
        print(f"  {describe(kind, param):<40} val macro-F1 {metrics['f1_macro']:.4f}  "
              f"acc {metrics['accuracy']:.4f}  ({results[-1]['fit_seconds']}s)", flush=True)
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train and evaluate NewsLens.")
    parser.add_argument("--data", type=Path, default=config.DEFAULT_DATASET_PATH,
                        help="Path to WELFake_Dataset.csv")
    parser.add_argument("--model-dir", type=Path, default=None,
                        help="Where to save artifacts (default: models/)")
    args = parser.parse_args(argv)
    model_dir = args.model_dir or config.get_model_dir()
    model_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.data} ...", flush=True)
    raw = load_raw_csv(args.data)
    prepared = prepare_dataset(raw)
    df, stats = prepared.frame, prepared.stats
    print("Dataset preparation:", json.dumps(stats, indent=2), flush=True)

    train, val, test = split_by_group(df)
    for name, part in [("train", train), ("validation", val), ("test", test)]:
        stats[f"{name}_rows"] = int(len(part))
        stats[f"{name}_fake_share"] = round(float(part["label"].mean()), 4)
    # Sanity check: the grouped split must not share any group.
    assert not (set(train["group"]) & set(val["group"])), "group leak train/val"
    assert not (set(train["group"]) & set(test["group"])), "group leak train/test"
    assert not (set(val["group"]) & set(test["group"])), "group leak val/test"
    print(f"Split sizes: train={len(train)}  validation={len(val)}  test={len(test)}", flush=True)

    print("\nModel selection on the VALIDATION set:", flush=True)
    candidates = evaluate_candidates(train, val)

    lr_candidates = [c for c in candidates if c["kind"] == LOGREG]
    nb_candidates = [c for c in candidates if c["kind"] == NAIVE_BAYES]
    best_lr = max(lr_candidates, key=lambda c: c["validation"]["f1_macro"])
    best_nb = max(nb_candidates, key=lambda c: c["validation"]["f1_macro"])
    print(f"\nSelected main model: {best_lr['name']}")
    print(f"Best baseline:       {best_nb['name']}", flush=True)

    # Refit the selected setting on train + validation, then test ONCE.
    trainval = pd.concat([train, val], ignore_index=True)
    final = build_pipeline(LOGREG, best_lr["param"])
    final.fit(trainval["document"], trainval["label"])
    test_metrics = compute_metrics(test["label"], final.predict(test["document"]))
    print("\n" + format_metrics("FINAL MODEL - held-out TEST set", test_metrics))

    # Extra diagnostic on the same test rows: headline only (no body text).
    has_title = test["title"].str.strip().str.len() > 0
    headline_metrics = compute_metrics(test.loc[has_title, "label"],
                                       final.predict(test.loc[has_title, "title"]))
    print("\n" + format_metrics("Diagnostic - TEST headlines only", headline_metrics))

    slim_for_saving(final)
    model_path = model_dir / config.MODEL_FILENAME
    joblib.dump(final, model_path, compress=3)

    trained_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    metrics = {
        "status": "evaluated",
        "trained_at_utc": trained_at,
        "selection_metric": "validation macro F1",
        "final_model": best_lr["name"],
        "baseline_model": best_nb["name"],
        "validation_candidates": candidates,
        "test": test_metrics,
        "test_headlines_only": headline_metrics,
        "dataset_stats": stats,
        "top_terms": top_terms(final),
    }
    metadata = {
        "project": "NewsLens",
        "is_test_fixture": False,
        "model_name": best_lr["name"],
        "model_kind": LOGREG,
        "hyperparameters": {"C": best_lr["param"], **{k: (list(v) if isinstance(v, tuple) else v)
                                                      for k, v in config.TFIDF_PARAMS.items()}},
        "trained_on": "train + validation (after selection on validation)",
        "trained_at_utc": trained_at,
        "random_seed": config.RANDOM_SEED,
        "dataset": {**config.DATASET_INFO, "local_file_md5": file_md5(args.data)},
        "raw_label_mapping": {str(k): config.LABEL_NAMES[v] for k, v in config.RAW_LABEL_MAP.items()},
        "label_names": {str(k): v for k, v in config.LABEL_NAMES.items()},
        "positive_class": "fake",
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
        "vocabulary_size": int(len(final.named_steps["tfidf"].vocabulary_)),
    }
    (model_dir / config.METRICS_FILENAME).write_text(json.dumps(metrics, indent=2))
    (model_dir / config.METADATA_FILENAME).write_text(json.dumps(metadata, indent=2))
    size_mb = model_path.stat().st_size / 1e6
    print(f"\nSaved {model_path} ({size_mb:.1f} MB), metrics.json and metadata.json to {model_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

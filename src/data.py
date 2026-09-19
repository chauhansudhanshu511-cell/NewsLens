"""Loading, cleaning, de-duplicating and splitting the WELFake dataset.

Leakage prevention (documented in the README):

* Exact duplicates are removed after an aggressive normalisation
  (lower-case, no punctuation/URLs/publisher markers). If the same normalised
  article appears with *both* labels, every copy is dropped because the
  label is unreliable.
* Near-duplicates are grouped, not deleted: two articles end up in the same
  group if they share the same normalised headline or the same opening
  ~300 characters of body text (typical for re-posted or lightly edited
  stories). Groups are then assigned *as a whole* to train, validation or
  test, so a story and its copy can never sit on both sides of a split.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from src import config
from src.preprocessing import (
    clean_text,
    combine_title_and_text,
    normalize_cleaned,
    normalize_for_dedup,
)

MIN_CLEAN_WORDS = 5          # documents shorter than this after cleaning are dropped
TITLE_KEY_MIN_WORDS = 5      # headlines shorter than this are too generic to group on
BODY_PREFIX_CHARS = 300      # length of the body "fingerprint" used for grouping
BODY_PREFIX_MIN_CHARS = 150  # bodies shorter than this are not fingerprinted


class DatasetError(ValueError):
    """Raised when the CSV is missing or does not match the expected schema."""


@dataclass
class PreparedData:
    frame: pd.DataFrame
    stats: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Loading (schema adapter)
# ---------------------------------------------------------------------------
def load_raw_csv(path: str | Path) -> pd.DataFrame:
    """Read the WELFake CSV and adapt it to the columns this project uses.

    Expected raw columns: an unnamed serial-number column, ``title``, ``text``
    and ``label`` (0/1). Column-name case and surrounding spaces are ignored.
    Returns a frame with ``title``, ``text`` and ``label`` (internal labels,
    see ``config.RAW_LABEL_MAP``). Rows with a missing/unknown label are
    dropped here and counted in ``df.attrs["dropped_bad_label"]``.
    """
    path = Path(path)
    if not path.exists():
        raise DatasetError(
            f"Dataset not found at '{path}'. Download WELFake_Dataset.csv from "
            f"{config.DATASET_INFO['source_url']} and place it in the data/ folder "
            "(see data/README.md)."
        )

    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower() for c in df.columns]

    missing = [c for c in (config.RAW_TITLE_COL, config.RAW_TEXT_COL, config.RAW_LABEL_COL)
               if c not in df.columns]
    if missing:
        raise DatasetError(
            f"The CSV is missing required column(s) {missing}. Found columns: "
            f"{list(df.columns)}. Expected WELFake columns: title, text, label."
        )

    raw_label = pd.to_numeric(df[config.RAW_LABEL_COL], errors="coerce")
    mapped = raw_label.map(config.RAW_LABEL_MAP)
    good = mapped.notna()

    out = pd.DataFrame({
        "title": df.loc[good, config.RAW_TITLE_COL],
        "text": df.loc[good, config.RAW_TEXT_COL],
        "label": mapped[good].astype(int),
    }).reset_index(drop=True)
    out.attrs["rows_raw"] = int(len(df))
    out.attrs["dropped_bad_label"] = int((~good).sum())
    return out


# ---------------------------------------------------------------------------
# Cleaning, de-duplication and grouping
# ---------------------------------------------------------------------------
class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def assign_near_duplicate_groups(titles: list[str], bodies: list[str]) -> list[int]:
    """Give every row a group id; rows sharing a title key or body key share a group."""
    n = len(titles)
    uf = _UnionFind(n)
    first_seen: dict[tuple[str, str], int] = {}

    for i in range(n):
        keys = []
        t = normalize_for_dedup(titles[i])
        if len(t.split()) >= TITLE_KEY_MIN_WORDS:
            keys.append(("title", t))
        b = normalize_for_dedup(bodies[i])
        if len(b) >= BODY_PREFIX_MIN_CHARS:
            keys.append(("body", b[:BODY_PREFIX_CHARS]))
        for key in keys:
            if key in first_seen:
                uf.union(first_seen[key], i)
            else:
                first_seen[key] = i

    return [uf.find(i) for i in range(n)]


def prepare_dataset(raw: pd.DataFrame) -> PreparedData:
    """Clean, de-duplicate and group the adapted dataset."""
    stats: dict = {
        "rows_raw": int(raw.attrs.get("rows_raw", len(raw))),
        "dropped_bad_label": int(raw.attrs.get("dropped_bad_label", 0)),
    }
    df = raw.copy()
    df["title"] = df["title"].fillna("").astype(str)
    df["text"] = df["text"].fillna("").astype(str)

    # 1. Build the document (headline + body) and drop empty/near-empty ones.
    df["document"] = [combine_title_and_text(t, x) for t, x in zip(df["title"], df["text"])]
    cleaned = df["document"].map(clean_text)
    empty = cleaned.map(lambda s: len(s.split())) < MIN_CLEAN_WORDS
    stats["dropped_empty_or_tiny"] = int(empty.sum())
    df = df[~empty].reset_index(drop=True)
    cleaned = cleaned[~empty].reset_index(drop=True)

    # 2. Exact duplicates after normalisation.
    df["dedup_key"] = cleaned.map(normalize_cleaned)
    labels_per_key = df.groupby("dedup_key")["label"].nunique()
    conflicting = set(labels_per_key[labels_per_key > 1].index)
    is_conflict = df["dedup_key"].isin(conflicting)
    stats["dropped_conflicting_label_duplicates"] = int(is_conflict.sum())
    df = df[~is_conflict]
    before = len(df)
    df = df.drop_duplicates(subset="dedup_key", keep="first").reset_index(drop=True)
    stats["dropped_exact_duplicates"] = int(before - len(df))

    # 3. Near-duplicate groups (kept together during splitting).
    df["group"] = assign_near_duplicate_groups(df["title"].tolist(), df["text"].tolist())
    stats["rows_after_cleaning"] = int(len(df))
    stats["near_duplicate_groups"] = int(df["group"].nunique())
    stats["rows_in_multi_row_groups"] = int(df["group"].duplicated(keep=False).sum())
    stats["class_counts"] = {config.LABEL_NAMES[k]: int(v)
                             for k, v in df["label"].value_counts().sort_index().items()}

    df = df.drop(columns=["dedup_key"])
    return PreparedData(frame=df, stats=stats)


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------
def split_by_group(df: pd.DataFrame, seed: int = config.RANDOM_SEED,
                   test_size: float = config.TEST_SIZE,
                   val_size: float = config.VALIDATION_SIZE):
    """Split into train / validation / test so that no group spans two splits."""
    outer = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    trainval_idx, test_idx = next(outer.split(df, groups=df["group"]))
    trainval, test = df.iloc[trainval_idx], df.iloc[test_idx]

    # val_size is a share of the whole dataset; convert it to a share of train+val.
    inner = GroupShuffleSplit(n_splits=1, test_size=val_size / (1 - test_size),
                              random_state=seed)
    train_idx, val_idx = next(inner.split(trainval, groups=trainval["group"]))
    train, val = trainval.iloc[train_idx], trainval.iloc[val_idx]
    return (train.reset_index(drop=True), val.reset_index(drop=True),
            test.reset_index(drop=True))

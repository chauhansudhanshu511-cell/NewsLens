"""Quick look at the raw dataset, including a label-direction sanity check.

    python -m src.inspect_dataset
    python -m src.inspect_dataset --data path/to/WELFake_Dataset.csv

Why the check exists: the WELFake page describes the label as "0 = fake and
1 = real", but the file behaves the other way round. Real news that came from
Reuters carries a "(Reuters)" dateline, so the raw label value holding almost
all Reuters datelines is the *real* label. This script prints that evidence
for your copy of the file so you never have to take the mapping on trust.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src import config


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=config.DEFAULT_DATASET_PATH)
    args = parser.parse_args(argv)
    if not args.data.exists():
        raise SystemExit(f"Not found: {args.data}. See data/README.md for download steps.")

    df = pd.read_csv(args.data)
    df.columns = [str(c).strip().lower() for c in df.columns]
    print(f"Rows: {len(df):,}   Columns: {list(df.columns)}")
    print("\nMissing values per column:\n" + df.isna().sum().to_string())
    print("\nRaw label counts:\n" + df["label"].value_counts(dropna=False).to_string())

    text = df["text"].fillna("").astype(str)
    reuters = text.str.contains(r"\(Reuters\)", regex=True)
    print("\nShare of articles containing a '(Reuters)' dateline, per raw label:")
    for raw_value, share in reuters.groupby(df["label"]).mean().items():
        print(f"  raw label {raw_value}: {share:.1%}")

    print("\nCurrent mapping in src/config.py (raw -> internal):")
    for raw_value, internal in config.RAW_LABEL_MAP.items():
        print(f"  {raw_value} -> {config.LABEL_NAMES[internal]}")

    shares = reuters.groupby(df["label"]).mean()
    reuters_heavy = int(shares.idxmax())
    mapped = config.LABEL_NAMES.get(config.RAW_LABEL_MAP.get(reuters_heavy, -1), "?")
    verdict = "consistent" if mapped == "real" else "INCONSISTENT - check RAW_LABEL_MAP"
    print(f"\nRaw label {reuters_heavy} holds most Reuters datelines and is mapped to "
          f"'{mapped}': {verdict}.")


if __name__ == "__main__":
    main()

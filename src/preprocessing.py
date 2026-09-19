"""Text preprocessing shared by training and inference.

Two kinds of cleaning happen here:

1. ``clean_text`` - the text the model actually sees. It removes URLs,
   social-media handles and a small, documented list of publisher markers
   (for example the "WASHINGTON (Reuters) -" dateline). Those markers would
   let the model "cheat": it could learn "contains the word Reuters -> real"
   instead of learning anything about the writing itself.

2. ``normalize_for_dedup`` - an aggressive normalisation used only to find
   duplicate and near-duplicate articles before the data is split.

The same ``clean_text`` function runs inside the saved scikit-learn Pipeline,
so training and inference always apply identical preprocessing.
"""

from __future__ import annotations

import re
from typing import Iterable

import numpy as np
from sklearn.preprocessing import FunctionTransformer

# --- patterns used by clean_text ----------------------------------------------
_URL_RE = re.compile(r"(https?://\S+|www\.\S+|pic\.twitter\.com/\S+|\b\S+\.(com|org|net)/\S*)", re.I)
_HANDLE_RE = re.compile(r"(?<!\w)@\w+")
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")

# Leading news-agency dateline, e.g. "WASHINGTON (Reuters) - " or "LONDON (Reuters) -".
_DATELINE_RE = re.compile(
    r"^\s*[A-Z][A-Za-z .,'/&-]{0,80}\((?:Reuters|AP|AFP)\)\s*[-–—]+\s*",
)

# Publisher names and photo-credit boilerplate that identify the *source*
# rather than the *content*. Removed everywhere, case-insensitive.
PUBLISHER_MARKERS = [
    r"\(\s*reuters\s*\)",
    r"\breuters\b",
    r"\b21st\s+century\s+wire\b",
    r"\b21wire\b",
    r"\bfeatured\s+image\s+(?:via|by|credit)\b[^\n.]*",
    r"\b(?:photo|image)\s+(?:via|by|credit)\b[^\n.]*",
    r"\bgetty\s+images\b",
    r"\bbreitbart\b",
    r"\bfollow\s+[^.\n]{0,60}?\s+on\s+twitter\b",   # author sign-off line
    r"\bread\s+more\s*:?",
    r"\bfor\s+(?:the\s+)?entire\s+story\b",
    r"\bpic\W{0,3}twitter\W{0,3}com\S*",          # left-over tweet-embed fragments
    r"\bhttps?\b\W*",
]
_MARKER_RE = re.compile("|".join(PUBLISHER_MARKERS), re.I)
# Lower-case substrings; if none is present, _MARKER_RE cannot match.
_MARKER_TRIGGERS = ("reuters", "century", "21wire", "image", "photo", "getty", "breitbart",
                    "twitter", "read more", "entire story", "http")
_WHITESPACE_RE = re.compile(r"\s+")

# --- patterns used by normalize_for_dedup ------------------------------------
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def _to_str(value: object) -> str:
    """Convert a possibly-missing value (None, NaN, number) to a string."""
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    return str(value)


def combine_title_and_text(title: object, text: object) -> str:
    """Join headline and body into one string (either part may be missing)."""
    parts = [_to_str(title).strip(), _to_str(text).strip()]
    return " ".join(p for p in parts if p)


def clean_text(text: object) -> str:
    """Clean one document for the model. Safe on None/NaN and empty strings."""
    s = _to_str(text)
    low = s.lower()
    # Cheap substring checks skip regexes that cannot match (big speed-up on 70k articles).
    if "(" in s:
        s = _DATELINE_RE.sub(" ", s)
    if any(k in low for k in ("http", "www.", ".com", ".org", ".net")):
        s = _URL_RE.sub(" ", s)
    if "@" in s:
        s = _EMAIL_RE.sub(" ", s)
        s = _HANDLE_RE.sub(" ", s)
    if any(k in low for k in _MARKER_TRIGGERS):
        s = _MARKER_RE.sub(" ", s)
    return " ".join(s.split())


def clean_texts(texts: Iterable[object]) -> list[str]:
    """Vectorised helper used inside the Pipeline (FunctionTransformer)."""
    return [clean_text(t) for t in texts]


def make_cleaner() -> FunctionTransformer:
    """A Pipeline step that applies ``clean_text`` to every document.

    ``clean_texts`` is a module-level function, so joblib can pickle it by
    reference. That is why the saved pipeline needs this project's ``src``
    package to be importable when it is loaded.
    """
    return FunctionTransformer(clean_texts, validate=False)


def normalize_for_dedup(text: object) -> str:
    """Lowercase, drop URLs/markers/punctuation and collapse whitespace.

    Two articles that differ only in capitalisation, punctuation, spacing,
    links or a publisher dateline get the same normalised string.
    """
    return normalize_cleaned(clean_text(text))


def normalize_cleaned(cleaned: str) -> str:
    """Second half of ``normalize_for_dedup`` for text already passed through clean_text."""
    return " ".join(_NON_ALNUM_RE.sub(" ", cleaned.lower()).split())


def word_count(text: object) -> int:
    return len(_to_str(text).split())

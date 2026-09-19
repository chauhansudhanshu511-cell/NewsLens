"""Checks on user input before it reaches the model."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from src import config

_LETTER_RE = re.compile(r"[A-Za-z]")


@dataclass
class ValidationResult:
    ok: bool                                  # False -> do not run the model
    error: str | None = None                  # why the input was rejected
    warnings: list[str] = field(default_factory=list)
    word_count: int = 0
    char_count: int = 0


def validate_input(text: str | None) -> ValidationResult:
    """Reject empty, too-short, too-long or non-text input; warn on short text."""
    s = (text or "").strip()
    words = s.split()
    res = ValidationResult(ok=False, word_count=len(words), char_count=len(s))

    if not s:
        res.error = "Please paste a news headline or article first."
        return res
    if len(s) > config.MAX_CHARS:
        res.error = (f"The text is too long ({len(s):,} characters). Please paste at most "
                     f"{config.MAX_CHARS:,} characters, for example the headline and the "
                     "first part of the article.")
        return res
    if len(s) < config.MIN_CHARS or len(words) < config.MIN_WORDS:
        res.error = (f"The text is too short to analyse. Please enter at least "
                     f"{config.MIN_WORDS} words and {config.MIN_CHARS} characters.")
        return res
    letters = len(_LETTER_RE.findall(s))
    if letters < 0.5 * len(s.replace(" ", "")):
        res.error = ("This does not look like English text. Please paste an English "
                     "news headline or article.")
        return res

    res.ok = True
    if len(words) < config.SHORT_TEXT_WORDS:
        res.warnings.append(
            f"Short input ({len(words)} words). The model was trained mostly on full "
            "articles, so predictions on headlines or short snippets are less reliable."
        )
    return res

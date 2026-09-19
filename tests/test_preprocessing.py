import math

import pandas as pd

from src import config
from src.data import assign_near_duplicate_groups, prepare_dataset, split_by_group
from src.preprocessing import clean_text, combine_title_and_text, normalize_for_dedup


def test_clean_text_handles_missing_values():
    assert clean_text(None) == ""
    assert clean_text(float("nan")) == ""
    assert clean_text("") == ""


def test_clean_text_removes_publisher_markers_and_urls():
    raw = ("WASHINGTON (Reuters) - The Senate passed the bill, Reuters reported. "
           "More at https://example.com/story and via @someone. Featured image via Getty Images")
    out = clean_text(raw)
    assert "reuters" not in out.lower()
    assert "http" not in out and "example.com" not in out
    assert "@someone" not in out
    assert "getty" not in out.lower()
    assert out.startswith("The Senate passed the bill")


def test_clean_text_removes_signoff_boilerplate():
    raw = ("The vote was close. Follow Jane Doe on Twitter. Read more: Breitbart News. "
           "pic twitter com/abc123 and https //t co/xyz")
    out = clean_text(raw).lower()
    for gone in ("follow jane doe on twitter", "read more", "breitbart", "pic twitter", "https"):
        assert gone not in out
    assert out.startswith("the vote was close.")


def test_combine_title_and_text_skips_missing_parts():
    assert combine_title_and_text("Title", "Body") == "Title Body"
    assert combine_title_and_text(None, "Body") == "Body"
    assert combine_title_and_text("Title", math.nan) == "Title"


def test_normalize_for_dedup_ignores_case_punctuation_and_spacing():
    a = normalize_for_dedup("Breaking:  The Vote PASSED!  https://x.com/a")
    b = normalize_for_dedup("breaking the vote passed")
    assert a == b == "breaking the vote passed"


def _frame(rows):
    df = pd.DataFrame(rows, columns=["title", "text", "label"])
    df.attrs["rows_raw"] = len(df)
    return df


BODY = "officials said the plan would be reviewed by the committee next week " * 5


def test_prepare_dataset_drops_empty_duplicates_and_conflicts():
    df = _frame([
        ["Vote passes", BODY, 0],
        ["VOTE PASSES!", BODY.upper(), 0],                  # exact duplicate after normalising
        ["", "", 1],                                         # empty
        [None, None, 0],                                     # missing
        ["Same story", "identical text with conflicting labels here ok", 0],
        ["Same story", "identical text with conflicting labels here ok", 1],
        ["Different", "a completely different article about the weather today", 1],
    ])
    prepared = prepare_dataset(df)
    s = prepared.stats
    assert s["dropped_empty_or_tiny"] == 2
    assert s["dropped_conflicting_label_duplicates"] == 2
    assert s["dropped_exact_duplicates"] == 1
    assert len(prepared.frame) == 2


def test_near_duplicates_share_a_group():
    titles = ["Senate passes new budget bill after long debate",
              "Senate passes new budget bill after long debate!",   # same title
              "Something else entirely happened in the city today",
              "Unrelated"]
    bodies = [BODY + " version one", BODY + " edited ending", "short", BODY + " tail"]
    groups = assign_near_duplicate_groups(titles, bodies)
    assert groups[0] == groups[1]           # same headline
    assert groups[0] == groups[3]           # same opening body text
    assert groups[2] != groups[0]


def test_split_has_no_group_overlap():
    rows = []
    for i in range(300):
        rows.append({"document": f"doc {i}", "label": i % 2, "group": i // 3})
    df = pd.DataFrame(rows)
    train, val, test = split_by_group(df, seed=config.RANDOM_SEED)
    assert len(train) + len(val) + len(test) == len(df)
    assert not set(train.group) & set(val.group)
    assert not set(train.group) & set(test.group)
    assert not set(val.group) & set(test.group)
    # Same seed -> same split.
    train2, _, _ = split_by_group(df, seed=config.RANDOM_SEED)
    assert train["document"].tolist() == train2["document"].tolist()

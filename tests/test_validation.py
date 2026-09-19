from src import config
from src.validation import validate_input


def test_empty_and_whitespace_rejected():
    for value in ["", "   ", "\n\t", None]:
        res = validate_input(value)
        assert not res.ok
        assert "paste" in res.error.lower()


def test_too_short_rejected():
    res = validate_input("Fake news")
    assert not res.ok
    assert "too short" in res.error.lower()


def test_too_long_rejected():
    res = validate_input("word " * (config.MAX_CHARS // 5 + 10))
    assert not res.ok
    assert "too long" in res.error.lower()


def test_non_text_rejected():
    res = validate_input("1234 5678 9012 3456 7890 !!!! ???? ####")
    assert not res.ok


def test_headline_accepted_with_short_warning():
    res = validate_input("Researchers publish new study on sleep and memory in teenagers")
    assert res.ok
    assert res.warnings and "less reliable" in res.warnings[0]


def test_full_article_accepted_without_warning():
    article = " ".join(["The council discussed the budget proposal in detail."] * 12)
    res = validate_input(article)
    assert res.ok
    assert res.warnings == []
    assert res.word_count >= config.SHORT_TEXT_WORDS
